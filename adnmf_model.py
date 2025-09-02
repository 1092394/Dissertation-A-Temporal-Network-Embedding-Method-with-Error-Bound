# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence, Union
import time
from algorithms import *
from algorithms import _build_U_current_full

EmbedMode = Literal["aca", "cols", "randomized_svd"] # cols: generilsed Nystrom
Event = Union[Tuple[Literal["node"], np.ndarray], Tuple[Literal["edge"], np.ndarray, np.ndarray]]


@dataclass
class ANDMFModel:
    """
    Attributes
    ----------
    U0 : (n0, k) ndarray
        Top-k eigenvectors of S0 = D0^{-1/2} A0 D0^{-1/2}.
    Lambda : (k,) ndarray
        Top-k eigenvalues of S0.
    D : (n,) ndarray
        Current degree vector (updated online).
    weights : (T,) ndarray
        Filter weights w_r.
    b_neg : float
        Negative-sampling parameter.
    d_embed : int
        Target embedding dimension for the final export.
    tau_clip : float
        Lower clipping for M before elementwise log.
    embed_mode : {"aca", "cols", "randomized_svd"}
        Backend used by estimate_embedding_logmhat.
    ML_diag : (n,) ndarray of left scalings; MR : (k,k) right rotation;
    delta_rows_idx : (t,) int64 indices; delta_rows_val : (t,k) row updates.
    vol : float
        Graph volume, updated online (assumes undirected).
    rng_seed : int
        Seed forwarded to randomized routines.
    """

    U0: np.ndarray
    Lambda: np.ndarray
    D: np.ndarray

    weights: np.ndarray
    b_neg: float

    d_embed: int
    tau_clip: float = 1.0
    embed_mode: EmbedMode = "aca"
    rng_seed: int = 0


    n_cols: Optional[int] = None
    oversample: Optional[int] = None
    n_iter: int = 2
    n_oversamples: Optional[int] = None
    build_chunk: int = 4096

    # Internal lazy state
    ML_diag: np.ndarray = field(init=False)
    MR: np.ndarray = field(init=False)
    delta_rows_idx: np.ndarray = field(init=False)
    delta_rows_val: np.ndarray = field(init=False)
    vol: float = field(init=False)
    n: int = field(init=False)
    k: int = field(init=False)

    def __post_init__(self):
        self.U0 = np.asarray(self.U0, dtype=float)
        self.Lambda = np.asarray(self.Lambda, dtype=float)
        self.D = np.asarray(self.D, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        assert self.U0.ndim == 2 and self.Lambda.ndim == 1
        n0, k = self.U0.shape
        assert self.Lambda.shape[0] == k, "Lambda dimension must match U0"
        assert self.D.shape[0] == n0, "D must have length n0"
        self.k = k
        self.n = int(n0)

        self.ML_diag = np.ones(self.n, dtype=float)
        self.MR = np.eye(self.k, dtype=float)
        self.delta_rows_idx = np.zeros((0,), dtype=np.int64)
        self.delta_rows_val = np.zeros((0, self.k), dtype=float)
        self.vol = float(np.sum(self.D))

        # Fill estimator defaults
        if self.n_cols is None:
            self.n_cols = self.d_embed
        if self.oversample is None:
            self.oversample = max(1, self.n_cols // 2)
        if self.n_oversamples is None:
            self.n_oversamples = max(1, self.d_embed // 2)

    @classmethod
    def from_adjacency(
        cls,
        A0: np.ndarray,
        k: int,
        weights: np.ndarray,
        b_neg: float,
        d_embed: int,
        *,
        tau_clip: float = 1.0,
        embed_mode: EmbedMode = "randomized_svd",
        rng_seed: int = 0,
        use_normalized: bool = True,
    ) -> "ANDMFModel":

        A0 = np.asarray(A0, dtype=float)
        A0 = 0.5 * (A0 + A0.T)
        D0 = A0.sum(axis=1)
        if use_normalized:
            invsqrt = 1.0 / np.sqrt(np.maximum(D0, 1e-12))
            S0 = (invsqrt[:, None] * A0) * invsqrt[None, :]
        else:
            S0 = A0
        w, V = np.linalg.eigh(S0)
        idx = np.argsort(w)[::-1][:k]
        U0 = V[:, idx]
        Lambda = w[idx]
        return cls(U0=U0, Lambda=Lambda, D=D0, weights=weights, b_neg=b_neg,
                   d_embed=d_embed, tau_clip=tau_clip, embed_mode=embed_mode,
                   rng_seed=rng_seed)

    @classmethod
    def from_eigendecomp(
        cls,
        U0: np.ndarray,
        Lambda: np.ndarray,
        D0: np.ndarray,
        weights: np.ndarray,
        b_neg: float,
        d_embed: int,
        **kwargs,
    ) -> "ANDMFModel":
        return cls(U0=U0, Lambda=Lambda, D=D0, weights=weights, b_neg=b_neg,
                   d_embed=d_embed, **kwargs)

    def update_node(self, b: np.ndarray, *, undirected: bool = True) -> None:
        b = np.asarray(b, dtype=float)
        assert b.shape[0] == self.n, "b must have length equal to current n"
        (ML_t_idx, ML_t_val, MR_t,
         d_idx_t, d_val_t, self.Lambda, Ind) = node_update(
            self.U0, self.ML_diag, self.MR, self.delta_rows_idx, self.delta_rows_val,
            self.Lambda, self.D, b
        )
        self.ML_diag, self.MR, self.delta_rows_idx, self.delta_rows_val, n_next = fast_compose_one_step(
            ML_t_idx, ML_t_val, MR_t, d_idx_t, d_val_t,
            self.ML_diag, self.MR, self.delta_rows_idx, self.delta_rows_val,
            True, self.n
        )
        # Degree & volume updates
        self.D = np.concatenate([self.D, np.array([0.0])], axis=0)
        if Ind.size > 0:
            self.D[Ind] += b[Ind]
        self.D[n_next - 1] = float(np.sum(b))
        if undirected:
            self.vol += float(2.0 * np.sum(b))
        else:
            self.vol += float(np.sum(b))
        self.n = int(n_next)

    def update_edge(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        assert x.shape[0] == self.n and y.shape[0] == self.n
        (ML_t_idx, ML_t_val, MR_t,
         d_idx_t, d_val_t, self.Lambda, Ind) = edge_update(
            self.U0, self.ML_diag, self.MR, self.delta_rows_idx, self.delta_rows_val,
            self.Lambda, self.D, x, y
        )
        self.ML_diag, self.MR, self.delta_rows_idx, self.delta_rows_val, n_next = fast_compose_one_step(
            ML_t_idx, ML_t_val, MR_t, d_idx_t, d_val_t,
            self.ML_diag, self.MR, self.delta_rows_idx, self.delta_rows_val,
            False, self.n
        )
        # Degree and volume updates
        delta_d = float(np.sum(y)) * x + float(np.sum(x)) * y
        self.D = self.D + delta_d
        self.vol += float(np.sum(delta_d))
        self.n = int(n_next)
        
    def step(self, ev: Event) -> None:
        if ev[0] == "node":
            self.update_node(ev[1])
        elif ev[0] == "edge":
            self.update_edge(ev[1], ev[2])
        else:
            raise ValueError(f"Unknown event type {ev[0]}")

    def run(self, events: Iterable[Event], Tmax: Optional[int] = None) -> None:
        t0 = int(time.time() * 1000)
        # counter = 0
        for t, ev in enumerate(events, start=1):
            self.step(ev)
            #counter += 1
            if Tmax is not None and t >= Tmax:
                break
        t1 = int(time.time() * 1000)
        # Average update time

        self.update_time_avg = (t0-t1)#/counter
        print("Update time used:")
        # print("Average update time (s per event):")
        # print(self.update_time_avg)
        print(t1-t0)

    def get_embedding(self, *, embed_mode: Optional[EmbedMode] = None) -> np.ndarray:
        mode = embed_mode or self.embed_mode
        return estimate_embedding_logmhat(
            self.U0, self.ML_diag, self.MR, self.delta_rows_idx, self.delta_rows_val,
            self.Lambda, self.D, self.weights, self.vol, self.b_neg,
            d_embed=self.d_embed, tau_clip=self.tau_clip, embed_mode=mode,
            # GN (cols) params
            r=self.n_cols, oversample=self.oversample, rng_seed=self.rng_seed,
            # randomized_svd params
            n_iter=self.n_iter, n_oversamples=self.n_oversamples, build_chunk=self.build_chunk,
        )

    def get_U_rows(self, rows: Sequence[int]) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        # Use the same partial builder used inside node/edge updates
        from algorithms import _compose_rows  # local import to avoid namespace clutter
        return _compose_rows(self.U0, self.ML_diag, self.MR,
                             self.delta_rows_idx, self.delta_rows_val,
                             rows)

    def materialize_U(self) -> np.ndarray:
        return _build_U_current_full(self.U0, self.ML_diag, self.MR,
                                     self.delta_rows_idx, self.delta_rows_val,
                                     n=self.n)

    def state_dict(self) -> dict:
        return dict(
            U0=self.U0, Lambda=self.Lambda.copy(), D=self.D.copy(),
            ML_diag=self.ML_diag.copy(), MR=self.MR.copy(),
            delta_rows_idx=self.delta_rows_idx.copy(),
            delta_rows_val=self.delta_rows_val.copy(),
            vol=float(self.vol), n=int(self.n), k=int(self.k),
            weights=self.weights.copy(), b_neg=float(self.b_neg),
            d_embed=int(self.d_embed), tau_clip=float(self.tau_clip),
            embed_mode=self.embed_mode, rng_seed=int(self.rng_seed),
            n_cols=int(self.n_cols), oversample=int(self.oversample),
            n_iter=int(self.n_iter), n_oversamples=int(self.n_oversamples),
            build_chunk=int(self.build_chunk),
        )



if __name__ == "__main__":
    # A small example, suggested by GPT
    n0, k = 2000, 7
    A0 = (np.random.rand(n0, n0) > 0.7).astype(float)
    A0 = np.triu(A0, 1); A0 = 3*(A0 + A0.T)

    T = 10
    weights = np.ones(T) / float(T)
    b_neg = 1.0

    model = ANDMFModel.from_adjacency(A0, k=k, weights=weights, b_neg=b_neg, d_embed=k,
                                       tau_clip=1.0, embed_mode="randomized_svd")

    b = np.zeros(model.n); b[1] = 1.0; b[3] = 1.0
    model.update_node(b)

    x = np.zeros(model.n); y = np.zeros(model.n)
    x[0] = 1.0; y[2] = 1.0
    model.update_edge(x, y)

    E = model.get_embedding(embed_mode="randomized_svd")
    print("E shape:", E.shape)
    print(E)

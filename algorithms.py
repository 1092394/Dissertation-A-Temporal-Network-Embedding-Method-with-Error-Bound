
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular
from typing import Tuple
from numba import jit, prange, float64, int32

import numpy as np


def make_netmf_accessors(D, U, f_lambda, volG, b, eps_inner=1e-12):
    D = np.asarray(D, dtype=float)
    U = np.asarray(U, dtype=float)
    f_lambda = np.asarray(f_lambda, dtype=float)

    assert D.ndim == 1 and U.ndim == 2 and f_lambda.ndim == 1
    n, r = U.shape
    assert D.shape[0] == n and f_lambda.shape[0] == r

    logD = np.log(D)
    const = np.log(volG / b)

    # U_f[i, k] = U[i, k] * f_lambda[k]
    U_f = U * f_lambda  # broadcasting along columns

    def _inner_from_row(i):
        # vector w[j] = sum_k U[j, k] * (U[i, k] * f_lambda[k])  = (U @ U_f[i,:])
        return U @ U_f[i, :]

    def row_fn(i):
        w = _inner_from_row(i)
        # H[i, j] = const - 1/2 logD[i] - 1/2 logD[j] + log(max(w[j], eps))
        return const - 0.5 * logD[i] - 0.5 * logD + np.log(np.maximum(w, eps_inner))

    def col_fn(j):
        # symmetry: col j uses the same inner product with i<->j
        w = U @ U_f[j, :]
        return const - 0.5 * logD - 0.5 * logD[j] + np.log(np.maximum(w, eps_inner))

    return row_fn, col_fn


def frobenius_norm(x):
    return np.linalg.norm(x)


def find_max_vector_element(v, Ik):
    """
    Find the maximum element in the vector v excluding the elements at the indices in Ik.
    """
    mask = np.ones(len(v), dtype=bool)
    mask[Ik] = False
    max_index = np.argmax(np.abs(v[mask]))
    return np.abs(v[mask][max_index]), np.arange(len(v))[mask][max_index]


###########################################
##        ACA  classic algorithm         ##
###########################################

def aca(tol, max_rank, min_pivot, D, Ut, Lambda, weights, volG, b):
    """
    This function is based on Adaptive Cross Approximation: ACA and ACA-GP by vyastreb on
    https://github.com/vyastreb/ACA?tab=readme-ov-file
    """
    f_lambda = _f_of_Lambda(Lambda, weights)
    n = Ut.shape[0]
    U = np.zeros((n, max_rank), dtype=np.float64)
    V = np.zeros((max_rank, n), dtype=np.float64)
    ranks = 0
    Jk = []
    Ik = []
    history = np.zeros((max_rank, 4), dtype=np.float64)

    # Start algorithm
    # 1. Find randomly a column
    j1 = np.random.randint(0, n)
    # 2. Find the maximal elemnt
    row_fn, col_fn = make_netmf_accessors(D, Ut, f_lambda, volG, b)
    u1 = col_fn(j1)
    _, i1 = find_max_vector_element(u1, Ik)
    v1 = row_fn(i1)
    Ik.append(i1)
    Jk.append(j1)
    pivot = v1[j1]
    if abs(pivot) < min_pivot:
        print("/ ACA Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
        return U, V, 0., ranks, Jk, Ik, history
        # raise ValueError("Pivot is too small: " +str(pivot))
    sign_pivot = np.sign(pivot)
    sqrt_pivot = np.sqrt(np.abs(pivot))
    # 4. Evaluate the associated row and column of the matrix
    U[:, 0] = sign_pivot * u1 / sqrt_pivot
    V[0, :] = v1 / sqrt_pivot

    # Compute matrix norm
    R_norm = frobenius_norm(u1) * frobenius_norm(v1) / np.abs(pivot)
    M_norm = R_norm
    history[0] = np.array([R_norm, M_norm, R_norm / M_norm, pivot])

    ranks = 1
    # Main loop
    while R_norm > tol * M_norm and ranks < max_rank:
        # Version 1: Get a random column which is not in Jk
        j_k = Jk[0]
        while j_k in Jk:
            j_k = np.random.randint(0, n)
            # _, j_k = find_max_vector_element(v_k, Jk)
        # Version 2: Search for maximal entry in v_k
        # _, j_k = find_max_vector_element(v_k, Jk)

        Jk.append(j_k)
        # Extract the column
        u_k = col_fn(j_k)
        # Remove the contribution of the previous ranks
        u_k -= np.dot(U[:, :ranks], V[:ranks, j_k])

        # Find the maximal element
        _, i_k = find_max_vector_element(u_k, Ik)
        # Store the column
        Ik.append(i_k)
        # Compute the new row and column
        v_k = row_fn(i_k)
        # Remove the contribution of the previous ranks
        v_k -= np.dot(U[i_k, :ranks], V[:ranks, :])

        pivot = u_k[i_k]
        if abs(pivot) < min_pivot:
            print("/ ACA Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
            return U, V, 0., ranks, Jk, Ik, history
        sign_pivot = np.sign(pivot)
        sqrt_pivot = np.sqrt(np.abs(pivot))

        U[:, ranks] = u_k * sign_pivot / sqrt_pivot
        V[ranks, :] = v_k / sqrt_pivot

        # Compute residual norm
        u_k_norm = frobenius_norm(u_k)
        v_k_norm = frobenius_norm(v_k)
        R_norm = float64(u_k_norm * v_k_norm / np.abs(pivot))

        # Approximate matrix norm
        cross_term = 2 * np.dot(np.dot(U[:, :ranks].T, u_k), np.dot(v_k, V[:ranks, :].T)) / pivot
        M_norm = float64(np.sqrt(M_norm ** 2 + R_norm ** 2 + cross_term))

        # Increment the rank
        ranks += 1
        history[ranks - 1] = np.array([R_norm, M_norm, R_norm / M_norm, pivot])

    U_contiguous = np.ascontiguousarray(U[:, :ranks])
    V_contiguous = np.ascontiguousarray(V[:ranks, :])

    return U_contiguous, V_contiguous  # , R_norm / M_norm, ranks, Jk, Ik, history


def _chol_psd(A: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Cholesky with automatic jitter; raise if it still fails."""
    A = 0.5 * (A + A.T)
    eps = jitter
    for _ in range(7):
        try:
            return cholesky(A, lower=False, check_finite=False)
        except Exception:
            A = A + eps * np.eye(A.shape[0], dtype=A.dtype)
            eps *= 10.0
    raise "Cholesky failed."


def _topk_eig_sym(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k eigenpairs of a symmetric matrix (descending by eigenvalue)."""
    A = 0.5 * (A + A.T)
    w, V = eigh(A, check_finite=False)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx], w[idx]


def _compose_rows(U0, ML_diag, MR, delta_rows_idx, delta_rows_val, rows_idx):
    """
    Build only the requested rows of current Ut:
      Ut[rows] = ML_diag[rows] * (U0[rows] @ MR) + ΔM[rows]
    ΔM given as sparse row-list (delta_rows_idx, delta_rows_val).
    """
    rows_idx = np.asarray(rows_idx, dtype=np.int64)
    n0, k = U0.shape
    U_part = np.zeros((rows_idx.size, k), dtype=U0.dtype)

    # Base from U0 for old nodes
    mask_old = rows_idx < n0
    if np.any(mask_old):
        idx_old = rows_idx[mask_old]
        base = (U0[idx_old, :] @ MR)
        base *= ML_diag[idx_old, None]
        U_part[mask_old, :] = base

    # Add ΔM
    if delta_rows_idx.size > 0:
        pos = {int(r): j for j, r in enumerate(rows_idx.tolist())}
        for r, v in zip(delta_rows_idx.tolist(), delta_rows_val):
            j = pos.get(int(r))
            if j is not None:
                U_part[j, :] += v
    return U_part  # (len(rows_idx), k)


def _add_merge_rows(delta_rows_idx, delta_rows_val, add_idx, add_val):
    if add_idx.size == 0:
        return delta_rows_idx, delta_rows_val
    if delta_rows_idx.size == 0:
        return add_idx.copy(), add_val.copy()

    pos = {int(r): j for j, r in enumerate(delta_rows_idx.tolist())}
    rows_new, vals_new = [], []
    for r, v in zip(add_idx.tolist(), add_val):
        j = pos.get(int(r), None)
        if j is None:
            rows_new.append(int(r))
            vals_new.append(v)
        else:
            delta_rows_val[j, :] += v
    if rows_new:
        delta_rows_idx = np.concatenate([delta_rows_idx, np.asarray(rows_new, np.int64)], axis=0)
        delta_rows_val = np.vstack([delta_rows_val, np.vstack(vals_new)])
    return delta_rows_idx, delta_rows_val

def node_update(U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
                Lambda, D, b):
    n, k = U0.shape
    Ind = np.flatnonzero(b != 0)
    if Ind.size == 0:
        return (np.array([], np.int64), np.array([], float),
                np.eye(k), np.array([], np.int64), np.empty((0, k)),
                Lambda.copy(), Ind)

    Di = D[Ind]
    bi = b[Ind]
    Di_tilde = Di + bi

    Wi = np.sqrt(Di / np.maximum(Di_tilde, 1e-12))
    # Normalized b-hat to avoid scaling issues
    bhat_i = bi / np.maximum(np.sqrt(Di_tilde), 1e-12)
    denom_b = max(float(np.sum(bi)), 1e-12)
    bhat_i = bhat_i / np.sqrt(denom_b)

    U_rows = _compose_rows(U0, ML_diag, MR, delta_rows_idx, delta_rows_val, Ind)

    # G = I - Σ g_i u_i u_i^T, where g_i = bi / (di + bi)
    gi = bi / np.maximum(Di_tilde, 1e-12)
    G = np.eye(k)
    for j in range(Ind.size):
        uj = U_rows[j, :]
        G -= gi[j] * np.outer(uj, uj)

    # c = Σ (Wi * bhat_i) u_i
    coeff = Wi * bhat_i
    c = (coeff[:, None] * U_rows).sum(axis=0)  # (k,)

    R = _chol_psd(G)
    z = solve_triangular(R.T, c, lower=True, check_finite=False)
    # alpha^2 = ||b̂||^2 - ||z||^2
    bhat_norm2 = float(np.sum((bi * bi) / np.maximum(Di_tilde, 1e-12)) / denom_b)
    alpha2 = max(bhat_norm2 - float(z @ z), 0.0)
    alpha = np.sqrt(alpha2)
    den_alpha = max(alpha, 1e-8)

    # Core (k+2)×(k+2)
    RLRT = R @ np.diag(Lambda) @ R.T
    Ctil = np.zeros((k + 2, k + 2))
    Ctil[:k, :k] = RLRT
    Ctil[:k, k] = z
    Ctil[k, :k] = z
    Ctil[k, k + 1] = alpha
    Ctil[k + 1, k] = alpha

    UC, gamma = _topk_eig_sym(Ctil, k)
    Lambda_next = gamma

    # MR_t = R^{-1} UC[:k,:] - (R^{-1} z) * q^T / alpha
    MR_left = solve_triangular(R, UC[:k, :], lower=False, check_finite=False)
    y = solve_triangular(R, z, lower=False, check_finite=False)
    q = UC[k, :]
    MR_t = MR_left - y[:, None] * (q[None, :] / den_alpha)

    # ΔM_t: affected old rows + new node row (index n)
    vec = q / den_alpha
    delta_rows_idx_t = []
    delta_rows_val_t = []
    for j in range(Ind.size):
        if bhat_i[j] == 0:
            continue
        delta_rows_idx_t.append(int(Ind[j]))
        delta_rows_val_t.append(float(bhat_i[j]) * vec)
    delta_rows_idx_t.append(int(n))  # new node
    delta_rows_val_t.append(UC[k + 1, :])
    delta_rows_idx_t = np.asarray(delta_rows_idx_t, dtype=np.int64)
    delta_rows_val_t = np.vstack(delta_rows_val_t)

    ML_t_idx = Ind.astype(np.int64)
    ML_t_val = Wi

    return ML_t_idx, ML_t_val, MR_t, delta_rows_idx_t, delta_rows_val_t, Lambda_next, Ind

def edge_update(U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
                Lambda, D, x, y):
    n, k = U0.shape
    Ind = np.union1d(np.flatnonzero(x != 0), np.flatnonzero(y != 0)).astype(np.int64)
    if Ind.size == 0:
        return (np.array([], np.int64), np.array([], float),
                np.eye(k), np.array([], np.int64), np.empty((0, k)),
                Lambda.copy(), Ind)

    xi = x[Ind]
    yi = y[Ind]
    sx, sy = float(np.sum(xi)), float(np.sum(yi))
    delta_d_full = sy * x + sx * y
    Di = D[Ind]
    Di1 = Di + delta_d_full[Ind]

    Wi = np.sqrt(Di / np.maximum(Di1, 1e-12))
    x_tilde = xi / np.maximum(np.sqrt(Di1), 1e-12)
    y_tilde = yi / np.maximum(np.sqrt(Di1), 1e-12)

    U_rows = _compose_rows(U0, ML_diag, MR, delta_rows_idx, delta_rows_val, Ind)

    # G = I - Σ g_i u_i u_i^T, where g_i = Δd_i / (d_i + Δd_i)
    gi = delta_d_full[Ind] / np.maximum(Di1, 1e-12)
    G = np.eye(k)
    for j in range(Ind.size):
        uj = U_rows[j, :]
        G -= gi[j] * np.outer(uj, uj)

    # c = [Σ W_i x̃_i u_i, Σ W_i ỹ_i u_i], T = [[Σ x̃^2, Σ x̃ỹ],[., Σ ỹ^2]]
    c1 = (Wi * x_tilde)[:, None] * U_rows
    c2 = (Wi * y_tilde)[:, None] * U_rows
    c = np.stack([c1.sum(axis=0), c2.sum(axis=0)], axis=1)  # (k,2)
    T11 = float(np.dot(x_tilde, x_tilde))
    T22 = float(np.dot(y_tilde, y_tilde))
    T12 = float(np.dot(x_tilde, y_tilde))
    T = np.array([[T11, T12], [T12, T22]])

    R = _chol_psd(G)
    z = solve_triangular(R.T, c, lower=True, check_finite=False)  # (k,2)
    H = T - z.T @ z  # 2×2

    # 2×2 PSD factor RX from eigen-decomp
    wH, VH = eigh(0.5 * (H + H.T), check_finite=False)
    RX = (np.sqrt(np.clip(wH, 0.0, None))[None, :] * VH.T)  # 2×2

    RLRT = R @ np.diag(Lambda) @ R.T
    Ctil = np.zeros((k + 2, k + 2))
    Ctil[:k, :k] = RLRT
    Ubar = np.vstack([c, RX])  # (k+2, 2)
    S2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    Ctil += Ubar @ S2 @ Ubar.T

    UC, gamma = _topk_eig_sym(Ctil, k)
    Lambda_next = gamma

    MR_left = solve_triangular(R, UC[:k, :], lower=False, check_finite=False)
    tail = UC[k:k + 2, :]  # 2×k
    RX_pinv = np.linalg.pinv(RX)  # 2×2
    Y = RX_pinv @ tail  # 2×k
    correction = solve_triangular(R, z, lower=False, check_finite=False)  # k×2
    MR_t = MR_left - correction @ Y

    # ΔM_t (only rows in Ind are non-zero)
    coeff = (np.stack([x_tilde, y_tilde], axis=1) @ Y)  # (m,k)
    delta_rows_idx_t = Ind.copy()
    delta_rows_val_t = coeff

    ML_t_idx = Ind
    ML_t_val = Wi

    return ML_t_idx, ML_t_val, MR_t, delta_rows_idx_t, delta_rows_val_t, Lambda_next, Ind


def fast_compose_one_step(ML_t_idx, ML_t_val, MR_t,
                          delta_rows_idx_t, delta_rows_val_t,
                          ML_diag, MR, delta_rows_idx, delta_rows_val,
                          is_node_step, n_current):
    if ML_t_idx.size > 0:
        ML_diag[ML_t_idx] *= ML_t_val
        if delta_rows_idx.size > 0:
            pos = {int(r): j for j, r in enumerate(delta_rows_idx.tolist())}
            for i, w in zip(ML_t_idx.tolist(), ML_t_val.tolist()):
                j = pos.get(i, None)
                if j is not None:
                    delta_rows_val[j, :] *= w

    if delta_rows_idx.size > 0:
        delta_rows_val = delta_rows_val @ MR_t

    delta_rows_idx, delta_rows_val = _add_merge_rows(
        delta_rows_idx, delta_rows_val, delta_rows_idx_t, delta_rows_val_t
    )

    n_next = n_current + 1 if is_node_step else n_current
    if is_node_step:
        ML_diag = np.concatenate([ML_diag, np.array([0.0])], axis=0)

    MR = MR @ MR_t
    return ML_diag, MR, delta_rows_idx, delta_rows_val, n_next

def _f_of_Lambda(Lambda: np.ndarray, weights: np.ndarray) -> np.ndarray:
    T = int(weights.size)
    lam = Lambda
    acc = np.zeros_like(lam)
    p = np.ones_like(lam)
    for r in range(1, T + 1):
        p = p * lam
        acc += float(weights[r - 1]) * p
    return acc


def _build_U_current_full(U0, ML_diag, MR, delta_rows_idx, delta_rows_val, n=None):
    n0, k = U0.shape
    if n is None:
        n = max(ML_diag.shape[0],
                (int(np.max(delta_rows_idx)) + 1) if delta_rows_idx.size else n0)

    U = np.zeros((n, k), dtype=U0.dtype)

    m = min(n0, n)
    if m > 0:
        U_old = (U0[:m, :] @ MR)
        U_old *= ML_diag[:m, None]
        U[:m, :] = U_old

    if delta_rows_idx.size > 0:
        for r, v in zip(delta_rows_idx.tolist(), delta_rows_val):
            if 0 <= int(r) < n:
                U[int(r), :] += v
    return U


def _oracle_logM_cols(U, Lambda, D, weights, vol_over_b, idx_cols, tau_clip=1.0):
    """
    Return selected columns of H:
        H = log( max( (vol/b) * D^{-1/2} U f(Λ) U^T D^{-1/2}, tau_clip ) ).
    """
    fLam = _f_of_Lambda(Lambda, weights)  # (k,)
    UJ = U[idx_cols, :]  # s×k
    R = (fLam[:, None] * UJ.T)  # k×s
    M_cols = U @ R  # n×s

    invsqrtD = 1.0 / np.sqrt(np.maximum(D, 1e-12))
    M_cols *= invsqrtD[:, None]
    M_cols *= invsqrtD[idx_cols][None, :]
    M_cols *= float(vol_over_b)

    if tau_clip is not None and tau_clip > 0:
        np.maximum(M_cols, float(tau_clip), out=M_cols)
    np.log(M_cols, out=M_cols)
    return M_cols


def oracle_logM_cols_factored(U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
                              Lambda, D, weights, vol_over_b, idx_cols,
                              tau_clip: float = 1.0):
    idx_cols = np.asarray(idx_cols, dtype=np.int64)
    U0 = np.asarray(U0);
    ML_diag = np.asarray(ML_diag);
    MR = np.asarray(MR)
    Lambda = np.asarray(Lambda);
    D = np.asarray(D);
    weights = np.asarray(weights)
    n0, k = U0.shape
    n = int(ML_diag.shape[0])
    s = idx_cols.size

    T = int(weights.size)
    lam = Lambda
    fLam = np.zeros_like(lam)
    p = np.ones_like(lam)
    for r in range(1, T + 1):
        p = p * lam
        fLam += float(weights[r - 1]) * p

    def _compose_rows_fixed(rows_idx):
        rows_idx = np.asarray(rows_idx, dtype=np.int64)
        U_part = np.zeros((rows_idx.size, k), dtype=U0.dtype)
        mask_old = rows_idx < n0
        if np.any(mask_old):
            idx_old = rows_idx[mask_old]
            base = (U0[idx_old, :] @ MR)
            base *= ML_diag[idx_old, None]
            U_part[mask_old, :] = base
        if delta_rows_idx.size > 0:
            pos = {int(r): j for j, r in enumerate(rows_idx.tolist())}
            for r, v in zip(delta_rows_idx.tolist(), delta_rows_val):
                j = pos.get(int(r))
                if j is not None:
                    U_part[j, :] += v
        return U_part

    UJ = _compose_rows_fixed(idx_cols)
    R = (fLam[:, None] * UJ.T)
    K = MR @ R
    UR_left = U0 @ K

    M_cols = np.zeros((n, s), dtype=U0.dtype)
    if n0 > 0:
        M_cols[:n0, :] = (ML_diag[:n0, None] * UR_left)
    if delta_rows_idx.size > 0:
        temp = delta_rows_val @ R
        for p, r in enumerate(delta_rows_idx.tolist()):
            if 0 <= int(r) < n:
                M_cols[int(r), :] += temp[p, :]
    eps = 1e-12
    invsqrtD = 1.0 / np.sqrt(np.maximum(D, eps))
    M_cols *= invsqrtD[:, None]
    M_cols *= invsqrtD[idx_cols][None, :]
    M_cols *= float(vol_over_b)

    if tau_clip is not None and tau_clip > 0:
        np.maximum(M_cols, float(tau_clip), out=M_cols)
    np.log(M_cols, out=M_cols)  # H[:, idx_cols]
    return M_cols


def _GN_cols_embedding_symmetric_factored(
        U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
        Lambda, D, weights, vol_over_b,
        d_embed: int, r: int, tau_clip: float = 1.0, rng_seed: int = 0):
    print("Start Esitmating")
    n = int(ML_diag.shape[0])
    rng = np.random.default_rng(rng_seed)
    idx_cols = rng.choice(n, size=int(r), replace=False)

    AX = oracle_logM_cols_factored(
        U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
        Lambda, D, weights, vol_over_b, idx_cols, tau_clip=tau_clip
    )
    W = AX[idx_cols, :]

    W = 0.5 * (W + W.T)
    w, Uloc = np.linalg.eigh(W)
    order = np.argsort(np.abs(w))[::-1][:d_embed]
    w = w[order];
    Uloc = Uloc[:, order]
    absv = np.abs(w)
    invsqrt = np.zeros_like(w)
    pos = absv > 0
    invsqrt[pos] = 1.0 / np.sqrt(absv[pos])
    E = AX @ (Uloc * invsqrt[None, :])  # n×d
    print("End estimating")
    return E


def _randomized_svd_embedding(U, Lambda, D, weights, vol_over_b,
                              d_embed, tau_clip=1.0, n_iter=2, n_oversamples=10,
                              rng_seed: int = 0, build_chunk=4096):

    from sklearn.utils.extmath import randomized_svd
    n = U.shape[0]

    H = np.empty((n, n))
    cols = np.arange(n, dtype=np.int64)
    for s in range(0, n, build_chunk):
        J = cols[s:s + build_chunk]
        H[:, s:s + build_chunk] = _oracle_logM_cols(
            U, Lambda, D, weights, vol_over_b, J, tau_clip=tau_clip
        )

    U_svd, s_svd, Vt_svd = randomized_svd(
        H, n_components=d_embed, n_iter=n_iter, n_oversamples=n_oversamples,
        random_state=int(rng_seed)
    )
    E = U_svd * np.sqrt(s_svd)[None, :]
    return E


def estimate_embedding_logmhat(
        U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
        Lambda, D, weights, vol, b_neg,
        d_embed, tau_clip=1.0, embed_mode="randomized_svd",
        r=None, oversample=None,
        n_iter=2, n_oversamples=10, rng_seed=0, build_chunk=4096,
):

    print("Start estimating")
    vol_over_b = float(vol / max(b_neg, 1e-12))

    if embed_mode == "cols":
        return _GN_cols_embedding_symmetric_factored(
            U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
            Lambda, D, weights, vol_over_b,
            d_embed=d_embed, r=r, tau_clip=tau_clip, rng_seed=rng_seed
        )

    if embed_mode == "aca":
        # ACA on H using factorized oracles; returns embedding E ∈ R^{n×d_embed}
        return estimate_embedding_logmhat_aca(
            U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
            Lambda, D, weights, vol, b_neg,
            d_embed=d_embed, tol=1e-4, max_rank=max(d_embed, 64), tau_clip=tau_clip, rng_seed=rng_seed
        )

    if embed_mode == "randomized_svd":
        n = int(D.shape[0])
        U = _build_U_current_full(U0, ML_diag, MR, delta_rows_idx, delta_rows_val, n)
        # scikit-learn randomized_svd parameters per docs: n_iter, n_oversamples, random_state.
        return _randomized_svd_embedding(
            U, Lambda, D, weights, vol_over_b,
            d_embed=d_embed, tau_clip=tau_clip,
            n_iter=n_iter, n_oversamples=n_oversamples, rng_seed=rng_seed,
            build_chunk=build_chunk
        )

    raise ValueError(f"Unknown embed_mode: {embed_mode}. Use 'cols' or 'randomized_svd'.")


def andmf(events, A0_eigvecs, A0_eigvals, D0,
          weights, b_neg, Tmax,
          d_embed, n_cols=None, embed_mode="cols", rng_seed=0):

    U0 = A0_eigvecs
    Lambda = A0_eigvals
    n, k = U0.shape

    ML_diag = np.ones(n)
    MR = np.eye(k)
    delta_rows_idx = np.zeros((0,), dtype=np.int64)
    delta_rows_val = np.zeros((0, k))
    D = D0.copy()
    vol = float(np.sum(D))

    step = 0
    for ev in events:
        if ev[0] == 'node':
            b = ev[1]
            ML_t_idx, ML_t_val, MR_t, d_idx_t, d_val_t, Lambda, Ind = node_update(
                U0, ML_diag, MR, delta_rows_idx, delta_rows_val, Lambda, D, b
            )
            ML_diag, MR, delta_rows_idx, delta_rows_val, n = fast_compose_one_step(
                ML_t_idx, ML_t_val, MR_t, d_idx_t, d_val_t,
                ML_diag, MR, delta_rows_idx, delta_rows_val, True, n
            )
            # Degree & volume updates
            D = np.concatenate([D, np.array([0.0])], axis=0)
            if Ind.size > 0:
                D[Ind] += b[Ind]
            D[n - 1] = float(np.sum(b))
            vol += float(2.0 * np.sum(b))  # undirected; adjust if directed
        else:
            x = ev[1]
            y = ev[2]
            ML_t_idx, ML_t_val, MR_t, d_idx_t, d_val_t, Lambda, Ind = edge_update(
                U0, ML_diag, MR, delta_rows_idx, delta_rows_val, Lambda, D, x, y
            )
            ML_diag, MR, delta_rows_idx, delta_rows_val, n = fast_compose_one_step(
                ML_t_idx, ML_t_val, MR_t, d_idx_t, d_val_t,
                ML_diag, MR, delta_rows_idx, delta_rows_val, False, n
            )
            delta_d = float(np.sum(y)) * x + float(np.sum(x)) * y
            D += delta_d
            vol += float(np.sum(delta_d))

        step += 1
        if step >= Tmax:
            break

    r_for_cols = (n_cols if n_cols is not None else d_embed)
    E = estimate_embedding_logmhat(
        U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
        Lambda, D, weights, vol, b_neg,
        d_embed=d_embed, embed_mode=embed_mode,
        r=r_for_cols, oversample=max(1, r_for_cols // 2),
        rng_seed=rng_seed,
        # randomized_svd params (used only if embed_mode='randomized_svd')
        n_iter=2, n_oversamples=max(1, r_for_cols // 2), build_chunk=4096,
    )
    return E

def _find_max_except(v, excluded_idx):
    mask = np.ones(v.size, dtype=bool)
    if len(excluded_idx):
        mask[np.asarray(excluded_idx, dtype=int)] = False
    if not np.any(mask):
        return 0.0, -1
    j = int(np.argmax(np.abs(v[mask])))
    idx_all = np.arange(v.size)[mask]
    return float(np.abs(v[mask][j])), int(idx_all[j])


def aca_from_oracle(n, row_fn, col_fn, tol, max_rank, min_pivot=1e-14):
    L = np.zeros((n, max_rank), dtype=float)
    R = np.zeros((max_rank, n), dtype=float)
    Jk, Ik, ranks = [], [], 0

    j1 = np.random.randint(0, n)
    col = col_fn(j1)
    _, i1 = _find_max_except(col, Ik)
    row = row_fn(i1)

    pivot = row[j1]
    if abs(pivot) < min_pivot:
        return L[:, :0], R[:0, :]

    sgn = np.sign(pivot) if pivot != 0 else 1.0
    sq = np.sqrt(abs(pivot))
    L[:, 0] = sgn * col / sq
    R[0, :] = row / sq

    Rnorm = np.linalg.norm(col) * np.linalg.norm(row) / abs(pivot)
    Mnorm = Rnorm
    ranks = 1
    Jk.append(j1);
    Ik.append(i1)

    while (Rnorm > tol * max(Mnorm, 1e-18)) and ranks < max_rank:
        j_k = Jk[0]
        while j_k in Jk:
            j_k = np.random.randint(0, n)
        Jk.append(j_k)

        col = col_fn(j_k) - L[:, :ranks] @ R[:ranks, j_k]
        _, i_k = _find_max_except(col, Ik)
        Ik.append(i_k)

        row = row_fn(i_k) - L[i_k, :ranks] @ R[:ranks, :]

        pivot = col[i_k]
        if abs(pivot) < min_pivot:
            break

        sgn = np.sign(pivot)
        sq = np.sqrt(abs(pivot))
        L[:, ranks] = sgn * col / sq
        R[ranks, :] = row / sq

        u_k = col;
        v_k = row
        Rnorm = np.linalg.norm(u_k) * np.linalg.norm(v_k) / abs(pivot)
        cross = 2.0 * (L[:, :ranks].T @ u_k) @ (v_k @ R[:ranks, :].T) / pivot
        Mnorm = float(np.sqrt(Mnorm ** 2 + Rnorm ** 2 + cross))
        ranks += 1

    return L[:, :ranks], R[:ranks, :]


def embedding_from_lowrank(L, R, d_embed):
    QL, RL = np.linalg.qr(L, mode='reduced')
    QR, RR = np.linalg.qr(R.T, mode='reduced')
    S = RL @ RR.T
    Us, s, VsT = np.linalg.svd(S, full_matrices=False)
    d = min(d_embed, s.size)
    E = (QL @ Us[:, :d]) * np.sqrt(s[:d])[None, :]
    return E


def estimate_embedding_logmhat_aca(
        U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
        Lambda, D, weights, vol, b_neg,
        d_embed, tol=1e-6, max_rank=None, tau_clip=1.0, rng_seed=0
):
    n = int(ML_diag.shape[0])
    if max_rank is None:
        max_rank = max(d_embed, 32)
    vol_over_b = float(vol / max(b_neg, 1e-12))

    def col_fn(j):
        cols = oracle_logM_cols_factored(
            U0, ML_diag, MR, delta_rows_idx, delta_rows_val,
            Lambda, D, weights, vol_over_b, idx_cols=np.asarray([j], dtype=np.int64),
            tau_clip=tau_clip
        )
        return cols[:, 0].copy()

    def row_fn(i):
        # symmetry: H is symmetric, row = column
        return col_fn(i).copy()

    np.random.seed(rng_seed)
    L, R = aca_from_oracle(n, row_fn, col_fn, tol=tol, max_rank=max_rank)
    if L.shape[1] == 0:
        return np.zeros((n, d_embed))
    return embedding_from_lowrank(L, R, d_embed=d_embed)


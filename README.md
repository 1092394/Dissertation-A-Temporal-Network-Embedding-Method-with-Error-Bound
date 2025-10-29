# A Temporal Network Embedding Method with Error Bound
This dissertation, supervised by Prof. Renaud Lambiotte, is submitted for the degree of MSc in Mathematical Modelling and Scientific Computing in the University of Oxford. This disseration proposed a new matrix-factorisation based embedding method for temporal (or dynamic) networks. The main results are 

Theorem 6 (One Step Node Update of Normalised Adjacency Matrix)

Let $G_t$ be a symmetric network without isolated vertices at time $t$,
	$A_t\!\in\!\mathbb{R}^{n\times n}$ as its adjacency matrix and
	$D_t=\diag(\sum_j A[i,j])$ as degree matrix. At time $t+1$, a new node is added with incident
	vector $b\in\mathbb{R}^n$ having $m \ll n$ nonzeros, so that
	$$
	A_{t+1}=\begin{pmatrix}A_t & b\\ b^\top & 0\end{pmatrix}.
	$$
	Define $S_t:=D_t^{-1/2}A_tD_t^{-1/2}$, and let a $k$-term orthogonal diagonalisation  approximation of $S_t$ as 
  $$
	\widehat{S}_t^{(k)} = U_t\Lambda_t^{(k)}U_t^\top,\ U_t\in \mathbb{R}^{n\times k},\ \Lambda_t^{(k)} \in \mathbb{R}^{k\times k}
	$$
	with $U_t^\top U_t=I_k$ and $\Lambda_t^{(k)}$ diagonal. 	
	Further let $\tilde D_t:=D_t+$diag$(b)$, $d=\mathbf{1}^\top b$, $W:=\tilde D_t^{-1/2}D_t^{1/2}$, $\hat b:= d^{-1/2}\tilde D_t^{-1/2}b$, $\hat b\not\in$ span$(WU_t)$ and $\|W\|_2 \le \beta_{t,\mathrm{node}}$.
	Then for $S_{t+1}:=D_{t+1}^{-1/2}A_{t+1}D_{t+1}^{-1/2}$ there exists a $k$-term
	orthogonal diagonalisation approximation $\widehat S^{(k)}_{t+1} = U_{t+1}\Lambda_{t+1}^{(k)} U_{t+1}^\top$ and matrices
	$M^L_{t},M^R_{t},\Delta M_{t}$ such that:
	\begin{enumerate}
		\item[(i)]
		The orthogonal matrix $U_{t+1}\in\mathbb{R}^{(n+1)\times k}$ can be updated as
		\begin{equation}\label{eqn:node-update}
			\boxed{
				U_{t+1} = M^L_{t} U_t M^R_{t} + \Delta M_{t}.
			}
		\end{equation}			
		\item[(ii)] The error between $S_{t+1}$ and $\widehat S^{(k)}_{t+1}$ is upper bounded by 
		\begin{equation}\label{eqn:1-step-node-update-error}
			\boxed{
				\|S_{t+1}-\widehat S^{(k)}_{t+1}\|_F\ \le\ \beta_{t,\mathrm{node}}^2\|S_t - \widehat S_t^{(k)}\|_F + \sqrt{2}\|\hat b\|_2.
			}
		\end{equation}

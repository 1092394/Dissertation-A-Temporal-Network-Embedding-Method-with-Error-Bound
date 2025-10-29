# A Temporal Network Embedding Method with Error Bound
This dissertation, supervised by Prof. Renaud Lambiotte, is submitted for the degree of MSc in Mathematical Modelling and Scientific Computing in the University of Oxford. This disseration proposed a new matrix-factorisation based embedding method for temporal (or dynamic) networks. The main results are 

### Theorem (One Step Node Update of NAM)

Let $G_t$ be a symmetric network without isolated vertices at time $t$,  
with adjacency matrix $A_t \in \mathbb{R}^{n\times n}$ and  
degree matrix $D_t = \mathrm{diag}\!\left(\sum_j A[i,j]\right)$.

At time $t+1$, a new node is added with incident vector  
$b \in \mathbb{R}^n$ having $m \ll n$ nonzeros, so that

$$
A_{t+1} =
\begin{pmatrix}
A_t & b \\
b^\top & 0
\end{pmatrix}.
$$

Define  

$$
S_t := D_t^{-1/2} A_t D_t^{-1/2},
$$

and let a $k$-term orthogonal diagonalisation approximation of $S_t$ be  

$$
\widehat{S}_t^{(k)} = U_t \Lambda_t^{(k)} U_t^\top,
\qquad
U_t \in \mathbb{R}^{n\times k},\;
\Lambda_t^{(k)} \in \mathbb{R}^{k\times k},
$$

with $U_t^\top U_t = I_k$ and $\Lambda_t^{(k)}$ diagonal.

Further let  

$$
\tilde D_t := D_t + \mathrm{diag}(b), \qquad
d = \mathbf{1}^\top b, \qquad
W := \tilde D_t^{-1/2} D_t^{1/2}, \qquad
\hat b := d^{-1/2} \tilde D_t^{-1/2} b,
$$

and suppose $\hat b \notin \mathrm{span}(W U_t)$ and $\|W\|_2 \le \beta_{t,\mathrm{node}}$.

Then, for  

$$
S_{t+1} := D_{t+1}^{-1/2} A_{t+1} D_{t+1}^{-1/2},
$$

there exists a $k$-term orthogonal diagonalisation approximation  

$$
\widehat S^{(k)}_{t+1} = U_{t+1} \Lambda_{t+1}^{(k)} U_{t+1}^\top,
$$

and matrices $M^L_t$, $M^R_t$, $\Delta M_t$ such that:

---

#### (i) Node Update Formula

The orthogonal matrix $U_{t+1} \in \mathbb{R}^{(n+1)\times k}$ can be updated as

$$
U_{t+1} = M^L_t\, U_t\, M^R_t + \Delta M_t.
$$

---

#### (ii) Error Bound

The error between $S_{t+1}$ and $\widehat S^{(k)}_{t+1}$ is upper bounded by

$$
\|S_{t+1} - \widehat S^{(k)}_{t+1}\|_F
\le
\beta_{t,\mathrm{node}}^2 \|S_t - \widehat S_t^{(k)}\|_F
+ \sqrt{2}\, \|\hat b\|_2.
$$



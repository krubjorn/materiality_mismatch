"""
Vectorised deterministic CRLD (strategy actor-critic, Barfuss et al. 2019 PRE; pyCRLD `stratAC`)
re-implemented in numpy, plus a corporate 'nature commons' stochastic game with heterogeneous
impact and dependency (materiality mismatch) and four policy levers.

States: 0 = degraded (g), 1 = prosperous (p)  [same ordering as pyCRLD EcologicalPublicGood]
Actions: 0 = invest in nature-positive action (c), 1 = do not (d)
"""
import itertools as it
import numpy as np


# ----------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------
class NatureCommons:
    """
    N firms share an ecosystem that can tip from prosperous to degraded.

    h      : impact shares (sum to 1). Collapse prob per step in p = qc * sum_{non-investors} h_i
             (with h_i = 1/N this reproduces pyCRLD EcologicalPublicGood exactly).
    dep    : relative dependency (financial materiality) of each firm; degraded-state loss m_i = m * dep_i
    f, c   : synergy factor and cost of nature-positive investment (prosperous-state public good)
    Policy levers
    lam    : risk pricing / disclosure intensity -> perceived loss scaled by lam (acts only through dep)
    kappa  : materiality bridging -> each firm internalises kappa x mean loss of all other firms
             (value-chain contracts, lender covenants, insurance linkage: impact made financially material)
    tau    : PES / water-fund transfer -> each investing impactor receives tau*c from dependents
    """

    def __init__(self, h, dep, f=1.2, c=5.0, m=-5.0, qc=0.2, qr=0.01,
                 lam=1.0, kappa=0.0, tau=0.0, payers=None, receivers=None):
        self.h = np.asarray(h, float)
        self.N = len(self.h)
        self.dep = np.asarray(dep, float)
        self.f, self.c, self.m, self.qc, self.qr = f, c, m, qc, qr
        self.lam, self.kappa, self.tau = lam, kappa, tau
        self.payers = [] if payers is None else list(payers)
        self.receivers = [] if receivers is None else list(receivers)
        self.Z, self.M = 2, 2
        self.acts = np.array(list(it.product(range(2), repeat=self.N)))  # J x N
        self.J = len(self.acts)
        self.T = self._T()
        self.R = self._R()
        self.Rbar = np.einsum('sjt,isjt->isj', self.T, self.R)

    def loss(self):
        """Perceived per-step degraded-state payoff for each firm (incl. levers)."""
        base = self.lam * self.m * self.dep
        if self.kappa != 0 and self.N > 1:
            others = (base.sum() - base) / (self.N - 1)
            base = base + self.kappa * others
        return base

    def _T(self):
        T = np.zeros((2, self.J, 2))
        for j, a in enumerate(self.acts):
            pc = self.qc * self.h[a == 1].sum()
            T[1, j, 0], T[1, j, 1] = pc, 1 - pc
            T[0, j, 1], T[0, j, 0] = self.qr, 1 - self.qr
        return T

    def _R(self):
        R = np.zeros((self.N, 2, self.J, 2))
        L = self.loss()
        for j, a in enumerate(self.acts):
            coop = (a == 0)
            pg = self.f * self.c * coop.sum() / self.N
            for i in range(self.N):
                r = pg - (self.c if coop[i] else 0.0)
                # PES transfers among cooperating receivers, paid by payers
                if self.tau > 0 and self.payers:
                    paid = self.tau * self.c * sum(coop[k] for k in self.receivers)
                    if i in self.receivers and coop[i]:
                        r += self.tau * self.c
                    if i in self.payers:
                        r -= paid / len(self.payers)
                R[i, 1, j, 1] = r
                R[i, 1, j, 0] = L[i]
                R[i, 0, j, :] = L[i]
        return R


# ----------------------------------------------------------------------------------------
# Deterministic strategy actor-critic learning dynamics
# ----------------------------------------------------------------------------------------
class CRLD:
    def __init__(self, env, alpha=0.1, gamma=0.99, prefactor=True):
        self.env = env
        N = env.N
        self.alpha = np.broadcast_to(np.asarray(alpha, float), (N,)).copy()
        self.gamma = np.broadcast_to(np.asarray(gamma, float), (N,)).copy()
        self.pre = (1 - self.gamma) if prefactor else np.ones(N)
        acts = env.acts
        # indicator I[i,j,a] = 1 if agent i plays a in joint action j
        self.I = np.zeros((N, env.J, 2))
        for i in range(N):
            self.I[i, np.arange(env.J), acts[:, i]] = 1.0

    def _probs(self, X):
        """X: (B,N,Z,M). Returns P (B,Z,J) and P_minus_i (B,N,Z,J)."""
        acts = self.env.acts
        N = self.env.N
        # per-agent prob of its component of joint action j: (B,N,Z,J)
        pj = np.stack([X[:, i][:, :, acts[:, i]] for i in range(N)], axis=1)
        P = pj.prod(axis=1)
        Pm = np.stack([np.prod(np.delete(pj, i, axis=1), axis=1) for i in range(N)], axis=1)
        return P, Pm

    def values(self, X, P=None):
        env = self.env
        if P is None:
            P, _ = self._probs(X)
        Tss = np.einsum('bsj,sjt->bst', P, env.T)
        Ris = np.einsum('bsj,isj->bis', P, env.Rbar)
        B, N = X.shape[0], env.N
        A = np.eye(2)[None, None] - self.gamma[None, :, None, None] * Tss[:, None]
        V = np.linalg.solve(A, Ris[..., None])[..., 0] * self.pre[None, :, None]
        return V, Tss, Ris

    def td_error(self, X):
        env = self.env
        P, Pm = self._probs(X)
        V, Tss, _ = self.values(X, P)
        Risa = np.einsum('bisj,ija,isj->bisa', Pm, self.I, env.Rbar)
        Tisat = np.einsum('bisj,ija,sjt->bisat', Pm, self.I, env.T)
        NextV = np.einsum('bisat,bit->bisa', Tisat, V)
        E = self.pre[None, :, None, None] * Risa + self.gamma[None, :, None, None] * NextV - V[..., None]
        return E, V, Tss

    def step(self, X):
        E, _, _ = self.td_error(X)
        Y = X * np.exp(self.alpha[None, :, None, None] * E)
        return Y / Y.sum(-1, keepdims=True)

    def run(self, X0, Tmax=6000, tol=1e-6, record=False):
        X = X0.copy()
        B = X.shape[0]
        done = np.zeros(B, bool)
        tconv = np.full(B, Tmax)
        traj = [X.copy()] if record else None
        for t in range(Tmax):
            Xn = self.step(X)
            d = np.abs(Xn - X).max(axis=(1, 2, 3))
            X = np.where(done[:, None, None, None], X, Xn)
            newly = (~done) & (d < tol)
            tconv[newly] = t
            done |= newly
            if record:
                traj.append(X.copy())
            if done.all():
                break
        return (X, tconv, np.array(traj)) if record else (X, tconv)


def stationary_prosperity(model, X):
    """Long-run share of time the ecosystem spends in the prosperous state."""
    _, Tss, _ = model.values(X)
    a = Tss[:, 1, 0]          # p -> g
    b = Tss[:, 0, 1]          # g -> p
    return b / (a + b)


def random_inits(B, N, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1e-3, 1 - 1e-3, size=(B, N, 2))
    X = np.stack([x, 1 - x], axis=-1)
    return X

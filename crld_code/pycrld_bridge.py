"""
Bridge between the corporate nature-commons game and the original pyCRLD library
(github.com/BarfussLab/pyCRLD, commit a370d51).

Key identity: for tau = 0 the corporate game IS pyCRLD's EcologicalPublicGood with
agent-specific collapse leverage qc_i = QC * N * h_i and agent-specific collapse impact
m_i = perceived loss of firm i (risk loading lam, bridging kappa, dependency error).
Payment transfers (tau > 0) need a thin subclass that adds the transfer to the
prosperous-state rewards.

Batching: pyCRLD's jitted `stratAC.step` is vmapped over initial conditions, so every
learning step uses the library's own TD-error computation.
"""
import os, sys
sys.path.insert(0, os.environ.get('PYCRLD_PATH', '/home/claude/ext/pyCRLD'))
import numpy as np
import jax, jax.numpy as jnp
if os.environ.get('CRLD_X64', '0') == '1':
    jax.config.update('jax_enable_x64', True)

from pyCRLD.Environments.EcologicalPublicGood import EcologicalPublicGood
from pyCRLD.Agents.StrategyActorCritic import stratAC
from crld import NatureCommons


def corrected_Omega(self):
    """
    Other-agents summation tensor with ONE canonical ordering of the other agents.

    pyCRLD's `_OtherAgentsActionsSummationTensor` (commit a370d51) accepts every
    permutation of the other agents' indices, so for N >= 3 the strategy-averaged
    quantities Risa, Tisas and NextVisa are summed (N-1)! times (rows of Tisas sum to
    (N-1)!), and permuted terms pair agents with each other's strategies.
    For N = 2 there is one permutation and the original is exact. The only change here is
    requiring the other-agent indices to be the remaining agents in ascending order.
    """
    dim = np.concatenate(([self.N], [self.N] * (self.N - 1), [self.M],
                          [self.M] * self.N, [self.M] * (self.N - 1))).astype(int)
    Omega = np.zeros(dim, int)
    for I in range(self.N):
        notI = tuple(k for k in range(self.N) if k != I)        # canonical order
        for allA in np.ndindex(*([self.M] * self.N)):
            notA = tuple(allA[k] for k in notI)
            Omega[(I,) + notI + (allA[I],) + allA + notA] = 1
    return jnp.array(Omega)


class stratAC_fixed(stratAC):
    """pyCRLD strategy actor-critic with the corrected summation tensor."""
    _OtherAgentsActionsSummationTensor = corrected_Omega


class EPGTransfer(EcologicalPublicGood):
    """EcologicalPublicGood plus payment transfers from payers to cooperating receivers."""

    def __init__(self, *args, tau=0.0, payers=(), receivers=(), **kw):
        self.tau, self.payers, self.receivers = tau, list(payers), list(receivers)
        super().__init__(*args, **kw)

    def _reward(self, i, s, jA, s_):
        r = super()._reward(i, s, jA, s_)
        if self.tau > 0 and self.Sset[s] == 'p' and self.Sset[s_] == 'p':
            coop = [self.Aset[k][jA[k]] == 'c' for k in range(self.N)]
            paid = self.tau * self.c[0] * sum(coop[k] for k in self.receivers)
            if i in self.receivers and coop[i]:
                r += self.tau * self.c[i]
            if i in self.payers:
                r -= paid / len(self.payers)
        return r


def pycrld_env(nc: NatureCommons):
    """Construct the pyCRLD environment equivalent to a NatureCommons instance."""
    qc_i = nc.qc * nc.N * nc.h
    m_i = nc.loss()
    if nc.tau > 0:
        return EPGTransfer(N=nc.N, f=nc.f, c=nc.c, m=m_i, qc=qc_i, qr=nc.qr,
                           tau=nc.tau, payers=nc.payers, receivers=nc.receivers)
    return EcologicalPublicGood(N=nc.N, f=nc.f, c=nc.c, m=m_i, qc=qc_i, qr=nc.qr)


class Runner:
    """Batched pyCRLD learning: vmap of the library's jitted step."""

    def __init__(self, nc, alpha, gamma, fixed=True):
        self.env = pycrld_env(nc)
        cls = stratAC_fixed if fixed else stratAC
        self.agent = cls(env=self.env, learning_rates=alpha, discount_factors=gamma)
        self._vstep = jax.jit(jax.vmap(lambda X: self.agent.step(X)[0]))
        self.N = nc.N

    def step(self, X):
        return self._vstep(X)

    def run(self, X0, Tmax=8000, tol=1e-5, chunk=50):
        X = jnp.asarray(X0)
        B = X.shape[0]
        done = np.zeros(B, bool); tconv = np.full(B, Tmax)
        t = 0
        while t < Tmax and not done.all():
            for _ in range(chunk):
                Xn = self._vstep(X)
                d = np.asarray(jnp.abs(Xn - X).max(axis=(1, 2, 3)))
                newly = (~done) & (d < tol)
                tconv[newly] = t
                done |= newly
                X = jnp.where(jnp.asarray(done)[:, None, None, None], X, Xn)
                t += 1
                if t >= Tmax or done.all():
                    break
        return np.asarray(X, dtype=float), tconv

    def prosperity(self, X):
        """Stationary probability of the prosperous state via pyCRLD's Tss."""
        out = []
        for x in X:
            Tss = np.asarray(self.agent.Tss(jnp.asarray(x)), dtype=float)
            a, b = Tss[1, 0], Tss[0, 1]
            out.append(b / (a + b))
        return np.array(out)


def to_pycrld_R(nc):
    """Reshape our R (N,Z,J,Z) / T (Z,J,Z) to pyCRLD's (N,Z,M..M,Z) / (Z,M..M,Z)."""
    shp = [nc.M] * nc.N
    return (nc.R.reshape(nc.N, nc.Z, *shp, nc.Z), nc.T.reshape(nc.Z, *shp, nc.Z))


# ----------------------------------------------------------------------------------------
# Drop-in replacement for crld.CRLD that runs on pyCRLD (used when BACKEND=pycrld)
# ----------------------------------------------------------------------------------------
from jax import lax


class PyCRLD:
    """Same interface as crld.CRLD (run, step, values), but every learning step is
    pyCRLD's jitted stratAC.step (vmapped over initial conditions)."""

    def __init__(self, env, alpha=0.1, gamma=0.99, prefactor=False, fixed=None):
        assert not prefactor
        if fixed is None:
            fixed = os.environ.get('PYCRLD_STOCK', '0') != '1'
        self.nc = env
        self.penv = pycrld_env(env)
        cls = stratAC_fixed if fixed else stratAC
        self.agent = cls(env=self.penv, learning_rates=alpha, discount_factors=gamma)
        step1 = lambda X: self.agent.step(X)[0]
        self._vstep = jax.jit(jax.vmap(step1))
        self._vTss = jax.jit(jax.vmap(self.agent.Tss))
        self._chunks = {}

    def step(self, X):
        return np.asarray(self._vstep(jnp.asarray(X)), dtype=float)

    def values(self, X):
        Tss = np.asarray(self._vTss(jnp.asarray(X)), dtype=float)
        return None, Tss, None

    def _chunk(self, K, tol):
        key = (K, tol)
        if key not in self._chunks:
            vstep = self._vstep

            def f(X, done, tconv, t0):
                def body(k, c):
                    X, done, tconv = c
                    Xn = vstep(X)
                    d = jnp.abs(Xn - X).max(axis=(1, 2, 3))
                    Xu = jnp.where(done[:, None, None, None], X, Xn)
                    newly = (~done) & (d < tol)
                    tconv = jnp.where(newly, t0 + k, tconv)
                    return Xu, done | newly, tconv
                return lax.fori_loop(0, K, body, (X, done, tconv))
            self._chunks[key] = jax.jit(f)
        return self._chunks[key]

    def run(self, X0, Tmax=6000, tol=1e-6, record=False, K=100):
        assert not record and Tmax % K == 0
        X = jnp.asarray(X0); B = X.shape[0]
        done = jnp.zeros(B, bool); tconv = jnp.full(B, Tmax)
        f = self._chunk(K, tol)
        for t0 in range(0, Tmax, K):
            X, done, tconv = f(X, done, tconv, t0)
            if bool(done.all()):
                break
        return np.asarray(X, dtype=float), np.asarray(tconv)

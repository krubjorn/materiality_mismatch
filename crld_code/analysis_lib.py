"""
Extensions used by the analysis notebook (analysis_notebook.ipynb).

  * TransferNC        alternative payment designs (buy-out, levy-and-rebate, pressure fee)
  * two_group_env     the manuscript's impactor/dependent parameterisation for any even N
  * stability_margins equilibrium-selection diagnostics from TD errors at pure strategies
  * StochasticAC      sample-based independent actor-critic learners in the stochastic game
  * basin_from_firms  calibrate a game from a firm table (impact pressure, revenue at risk)
"""
import numpy as np
from crld import NatureCommons, CRLD, stationary_prosperity, random_inits


# ----------------------------------------------------------------------------------------
# Payment designs
# ----------------------------------------------------------------------------------------
class TransferNC(NatureCommons):
    """
    NatureCommons with one of three budget-balanced payment designs, paid while the
    ecosystem stays prosperous (s = p -> p), with intensity tau (in units of c):

    'buyout'   dependents pay tau*c to each investing impactor (water-fund style; manuscript)
    'rebate'   every investing firm receives tau*c, funded by an equal levy on all firms
    'fee'      each non-investing firm pays tau*c*N*h_i (pressure fee); revenue recycled
               equally to all firms
    """

    def __init__(self, *args, design='buyout', tau_d=0.0, impactors=(0, 1), dependents=(2, 3), **kw):
        self.design, self.tau_d = design, tau_d
        self.impactors, self.dependents = list(impactors), list(dependents)
        super().__init__(*args, **kw)

    def transfers(self, a):
        N, c, t = self.N, self.c, self.tau_d
        coop = (a == 0)
        x = np.zeros(N)
        if t == 0:
            return x
        if self.design == 'buyout':
            paid = t * c * sum(coop[k] for k in self.impactors)
            for k in self.impactors:
                x[k] += t * c * coop[k]
            for k in self.dependents:
                x[k] -= paid / len(self.dependents)
        elif self.design == 'rebate':
            x += t * c * coop
            x -= t * c * coop.sum() / N
        elif self.design == 'fee':
            fee = t * c * N * self.h * (~coop)
            x -= fee
            x += fee.sum() / N
        else:
            raise ValueError(self.design)
        return x

    def _R(self):
        R = super()._R()
        for j, a in enumerate(self.acts):
            R[:, 1, j, 1] += self.transfers(a)
        return R


class TargetedNC(NatureCommons):
    """Materiality bridging applied firm by firm: kappa_vec[i] is the share of the other
    firms' mean loss that firm i internalises (e.g. only firms under a buyer's contract)."""

    def __init__(self, *args, kappa_vec=None, **kw):
        self.kappa_vec = None if kappa_vec is None else np.asarray(kappa_vec, float)
        super().__init__(*args, **kw)

    def loss(self):
        base = self.lam * self.m * self.dep
        if self.kappa_vec is None or self.N < 2:
            return base
        others = (base.sum() - base) / (self.N - 1)
        return base + self.kappa_vec * others


# ----------------------------------------------------------------------------------------
# Manuscript parameterisation for any even N
# ----------------------------------------------------------------------------------------
def two_group(mu, N=4):
    k = N // 2
    h = np.array([(1 + mu) / N] * k + [(1 - mu) / N] * k)
    d = np.array([1 - mu] * k + [1 + mu] * k)
    return h, d


def two_group_env(mu, N=4, qc=0.8, cls=NatureCommons, **kw):
    h, d = two_group(mu, N)
    k = N // 2
    if cls is TransferNC:
        kw.setdefault('impactors', range(k)); kw.setdefault('dependents', range(k, N))
    return cls(h=h, dep=d, qc=qc, **kw)


def evaluate(env, X0, gamma=0.9, alpha=0.05, Tmax=8000, tol=1e-5, engine=CRLD):
    mod = engine(env, alpha=alpha, gamma=gamma, prefactor=False)
    X, tc = mod.run(X0, Tmax=Tmax, tol=tol)
    c = X[:, :, 1, 0]
    return dict(basin=float((c.min(1) > 0.5).mean()), invest=c.mean(0),
                prosp=float(stationary_prosperity(mod, X).mean()),
                tconv=float(np.median(tc)), unconv=int((tc == Tmax).sum()))


# ----------------------------------------------------------------------------------------
# Equilibrium-selection diagnostics
# ----------------------------------------------------------------------------------------
def stability_margins(env, gamma=0.9, eps=1e-6):
    """
    TD-error advantage of investing over not investing, in the prosperous state, for each
    firm, evaluated (i) when nobody invests and (ii) when everybody invests.

    invasion[i]  > 0  -> firm i is pulled out of the non-investment trap
    defection[i] > 0  -> firm i is pulled away from collective investment
    Collective investment is a stable state iff all defection < 0;
    the non-investment trap is stable iff all invasion < 0.
    """
    mod = CRLD(env, alpha=0.05, gamma=gamma, prefactor=False)
    N = env.N

    def at(p_invest):
        X = np.zeros((1, N, 2, 2))
        X[..., 0] = p_invest; X[..., 1] = 1 - p_invest
        E, _, _ = mod.td_error(X)
        return E[0, :, 1, 0] - E[0, :, 1, 1]      # (invest - not) in prosperous state

    return dict(invasion=at(eps), defection=-at(1 - eps))


# ----------------------------------------------------------------------------------------
# Sample-based learners
# ----------------------------------------------------------------------------------------
class StochasticAC:
    """
    Independent tabular actor-critic learners playing the stochastic game by sampling.
    Each firm keeps a critic V_i(s) and softmax policy logits theta_i(s, a).
      delta = r + gamma V(s') - V(s);  V(s) += zeta*delta;  theta(s, a_t) += eta*delta
    Vectorised over B independent runs.
    """

    def __init__(self, env, gamma=0.9, eta=0.01, zeta=0.05, seed=0):
        self.env, self.gamma, self.eta, self.zeta = env, gamma, eta, zeta
        self.rng = np.random.default_rng(seed)
        N = env.N
        self.pow2 = 2 ** np.arange(N - 1, -1, -1)       # joint-action index (itertools order)

    def run(self, B=64, T=200_000, X0=None, record_every=1000):
        env, rng, N = self.env, self.rng, self.env.N
        if X0 is None:
            p = rng.uniform(0.05, 0.95, (B, N, 2))
        else:
            p = X0[:, :, :, 0]
        theta = np.zeros((B, N, 2, 2))
        theta[..., 0] = np.log(p); theta[..., 1] = np.log(1 - p)
        V = np.zeros((B, N, 2))
        s = np.ones(B, int)                            # start prosperous
        T_cum = env.T.cumsum(-1)
        hist_p, hist_s = [], []
        b = np.arange(B)
        prosp_count = np.zeros(B)
        for t in range(T):
            th = theta[b, :, s, :]                     # (B, N, 2)
            pr = np.exp(th - th.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
            a = (rng.random((B, N)) > pr[..., 0]).astype(int)     # 0 invest, 1 not
            j = a @ self.pow2
            s2 = (rng.random(B)[:, None] > T_cum[s, j, :]).sum(-1)
            r = env.R[:, s, j, s2].T                   # (B, N)
            Vs = V[b, :, s]; Vn = V[b, :, s2]
            delta = r + self.gamma * Vn - Vs
            V[b, :, s] += self.zeta * delta
            for i in range(N):
                theta[b, i, s, a[:, i]] += self.eta * delta[:, i]
            prosp_count += (s == 1)
            s = s2
            if (t + 1) % record_every == 0:
                thp = theta[:, :, 1, :]
                pp = np.exp(thp - thp.max(-1, keepdims=True)); pp /= pp.sum(-1, keepdims=True)
                hist_p.append(pp[..., 0].copy()); hist_s.append(prosp_count / record_every)
                prosp_count[:] = 0
        return np.array(hist_p), np.array(hist_s)     # (K, B, N), (K, B)


# ----------------------------------------------------------------------------------------
# Calibration from firm data
# ----------------------------------------------------------------------------------------
def basin_from_firms(df, pressure='pressure', exposure='revenue_at_risk', max_firms=6):
    """
    Turn a firm table for one basin into model inputs.
      h_i = pressure share; d_i = exposure relative to the mean (mean 1)
      mu  = 0.5 * sum |h_i - d_i / sum d|  (total-variation distance)
    Keeps the `max_firms` firms with the largest combined share (CRLD cost grows as 2^N).
    """
    df = df.copy()
    df['_h'] = df[pressure] / df[pressure].sum()
    df['_e'] = df[exposure] / df[exposure].sum()
    df = df.assign(_w=df['_h'] + df['_e']).nlargest(max_firms, '_w')
    h = (df[pressure] / df[pressure].sum()).to_numpy()
    e = (df[exposure] / df[exposure].sum()).to_numpy()
    d = e * len(df)
    mu = 0.5 * np.abs(h - e).sum()
    return h, d, mu, df.drop(columns=['_h', '_e', '_w'])

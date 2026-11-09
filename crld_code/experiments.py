import numpy as np, time
from crld import *
import os
if os.environ.get('BACKEND') == 'pycrld':
    from pycrld_bridge import PyCRLD as CRLD
SUF = os.environ.get('RESULTS_SUFFIX', '')

QC, GAMMA, ALPHA = 0.8, 0.9, 0.05
IMP, DEPS = [0, 1], [2, 3]


def firms(mu, dep_scale=None):
    h = [(1 + mu) / 4] * 2 + [(1 - mu) / 4] * 2
    dep = np.array([1 - mu] * 2 + [1 + mu] * 2, float)
    if dep_scale is not None:
        dep = dep * dep_scale
    return h, dep


def make_env(mu, lam=1.0, kappa=0.0, tau=0.0, dep_scale=None, qc=QC):
    h, dep = firms(mu, dep_scale)
    return NatureCommons(h=h, dep=dep, qc=qc, lam=lam, kappa=kappa, tau=tau,
                         payers=DEPS, receivers=IMP)


def evaluate(env, X0, gamma=GAMMA, Tmax=8000):
    mod = CRLD(env, alpha=ALPHA, gamma=gamma, prefactor=False)
    X, tc = mod.run(X0, Tmax=Tmax, tol=1e-5)
    c = X[:, :, 1, 0]
    # evaluate prosperity under TRUE environment (no perception distortion, no transfers)
    return dict(basin=float((c.min(1) > 0.5).mean()),
                imp=float(c[:, IMP].mean()), dep=float(c[:, DEPS].mean()),
                prosp=float(stationary_prosperity(mod, X).mean()),
                tconv=float(np.median(tc)), unconv=int((tc == Tmax).sum()))


if __name__ == '__main__':
    t0 = time.time()
    X0 = random_inits(64, 4, seed=2)
    out = {}

    # ---------------- E1: lever frontiers across materiality mismatch -----------------
    mus = np.round(np.arange(0, 1.0, 0.1), 2)
    levers = {
        'lam':   np.round(np.linspace(1, 8, 15), 3),
        'kappa': np.round(np.linspace(0, 1, 11), 3),
        'tau':   np.round(np.linspace(0, 1, 11), 3),
        'gamma': np.round(np.array([0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]), 3),
    }
    for name, vals in levers.items():
        B = np.zeros((len(mus), len(vals))); P = B.copy(); Tc = B.copy(); U = B.copy()
        for a, mu in enumerate(mus):
            for b, v in enumerate(vals):
                if name == 'gamma':
                    r = evaluate(make_env(mu), X0, gamma=v, Tmax=15000)
                else:
                    r = evaluate(make_env(mu, **{name: v}), X0)
                B[a, b], P[a, b], Tc[a, b], U[a, b] = r['basin'], r['prosp'], r['tconv'], r['unconv']
        out[f'E1_{name}_vals'] = vals
        out[f'E1_{name}_basin'] = B; out[f'E1_{name}_prosp'] = P
        out[f'E1_{name}_tconv'] = Tc; out[f'E1_{name}_unconv'] = U
        print('E1', name, 'done', round(time.time() - t0), 's; unconverged total', U.sum())
    out['mus'] = mus

    # ---------------- E2: fine sweep at mu = 0.8 for critical transitions ----------------
    for name, vals in {'lam': np.linspace(1, 8, 36), 'kappa': np.linspace(0, 1, 41),
                       'tau': np.linspace(0, 1, 41)}.items():
        res = [evaluate(make_env(0.8, **{name: v}), X0, Tmax=15000) for v in vals]
        out[f'E2_{name}_vals'] = vals
        for k in ['basin', 'prosp', 'tconv', 'imp', 'dep']:
            out[f'E2_{name}_{k}'] = np.array([r[k] for r in res])
    print('E2 done', round(time.time() - t0))

    # ---------------- E3: hysteresis / policy withdrawal (slow ramp) --------------------
    for name, grid in {'kappa': np.linspace(0, 1, 81), 'tau': np.linspace(0, 1, 81)}.items():
        path = np.concatenate([grid, grid[::-1]])
        X = np.full((1, 4, 2, 2), 0.5); X[..., 0] = 0.05; X[..., 1] = 0.95  # start near non-investment
        coop_path = []
        for v in path:
            mod = CRLD(make_env(0.8, **{name: v}), alpha=ALPHA, gamma=GAMMA, prefactor=False)
            for _ in range(150):
                X = mod.step(X)
            coop_path.append(X[0, :, 1, 0].mean())
        out[f'E3_{name}_path'] = path; out[f'E3_{name}_coop'] = np.array(coop_path)
    print('E3 done', round(time.time() - t0))

    # ---------------- E4: temporary scaffold -> does cooperation persist after removal? --
    # PES at tau_on for D periods, then removed (tau = tau_off). Track a single trajectory set.
    X0s = random_inits(64, 4, seed=7)
    sc = {}
    for tau_on, tau_off in [(0.6, 0.0), (0.6, 0.3), (0.0, 0.0)]:
        X = X0s.copy()
        mod_on = CRLD(make_env(0.8, tau=tau_on), alpha=ALPHA, gamma=GAMMA, prefactor=False)
        mod_off = CRLD(make_env(0.8, tau=tau_off), alpha=ALPHA, gamma=GAMMA, prefactor=False)
        series = []
        for t in range(1500):
            X = (mod_on if t < 500 else mod_off).step(X)
            series.append((X[:, :, 1, 0].min(1) > 0.5).mean())
        sc[f'{tau_on}_{tau_off}'] = np.array(series)
    for k, v in sc.items():
        out[f'E4_{k}'] = v
    print('E4 done', round(time.time() - t0))

    # ---------------- E5: information architecture (Burton et al. 2024) -----------------
    # Perceived dependency = true dependency x exp(sigma * eps_i)
    #   private:   eps_i independent
    #   monoculture: eps_i = common shock (single shared rating / AI nature-risk tool)
    #   pooled:    each firm uses the average of all N independent assessments
    rng = np.random.default_rng(11)
    mu5, sigma, D = 0.5, 0.6, 300
    X05 = random_inits(32, 4, seed=5)
    arch = {}
    for label in ['private', 'monoculture', 'pooled']:
        vals = []
        for d in range(D):
            z = rng.standard_normal(); e = rng.standard_normal(4)
            if label == 'private':
                eps = e
            elif label == 'monoculture':
                eps = np.full(4, z)
            else:
                eps = np.full(4, e.mean())  # pooled estimate of a common bias-free signal
            ds = np.exp(sigma * eps - 0.5 * sigma ** 2)  # mean-one multiplicative error
            r = evaluate(make_env(mu5, dep_scale=ds), X05, gamma=0.95)
            vals.append((r['basin'], r['prosp']))
        arch[label] = np.array(vals)
        out[f'E5_{label}'] = arch[label]
        print('E5', label, 'mean basin', arch[label][:, 0].mean().round(3),
              'sd', arch[label][:, 0].std().round(3),
              'P(basin<0.1)', (arch[label][:, 0] < 0.1).mean().round(3))
    # reference without error
    out['E5_truth'] = np.array([evaluate(make_env(mu5), X05, gamma=0.95)['basin']])
    print('E5 truth basin', out['E5_truth'])

    np.savez(f'results{SUF}.npz', **out)
    print('total', round(time.time() - t0), 's')

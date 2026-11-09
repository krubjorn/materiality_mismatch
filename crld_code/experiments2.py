import numpy as np, time
from crld import *
import os
if os.environ.get('BACKEND') == 'pycrld':
    from pycrld_bridge import PyCRLD as CRLD
SUF = os.environ.get('RESULTS_SUFFIX', '')
from experiments import make_env, evaluate, ALPHA, GAMMA, IMP, DEPS

t0 = time.time()
out = {}
X0 = random_inits(64, 4, seed=2)

# ---- E6: policy-mix maps at mu = 0.8 (prosperity + group investment) ----------------
grid = np.round(np.linspace(0, 1, 11), 2)
lamg = np.round(np.linspace(1, 3, 11), 2)
for name, (ax1, v1), (ax2, v2) in [('kt', ('kappa', grid), ('tau', grid)),
                                     ('lk', ('lam', lamg), ('kappa', grid))]:
    P = np.zeros((len(v1), len(v2))); I = P.copy(); D = P.copy(); Bn = P.copy()
    for a, x in enumerate(v1):
        for b, y in enumerate(v2):
            r = evaluate(make_env(0.8, **{ax1: x, ax2: y}), X0)
            P[a, b], I[a, b], D[a, b], Bn[a, b] = r['prosp'], r['imp'], r['dep'], r['basin']
    out[f'E6_{name}_v1'], out[f'E6_{name}_v2'] = v1, v2
    out[f'E6_{name}_prosp'], out[f'E6_{name}_imp'], out[f'E6_{name}_dep'], out[f'E6_{name}_basin'] = P, I, D, Bn
    print('E6', name, round(time.time() - t0))

# ---- E3b: hysteresis with slow ramp, per group ---------------------------------------
for name in ['kappa', 'lam']:
    g = np.linspace(0, 1, 51) if name == 'kappa' else np.linspace(1, 3, 51)
    path = np.concatenate([g, g[::-1]])
    X = np.zeros((1, 4, 2, 2)); X[..., 0] = 0.05; X[..., 1] = 0.95
    imp, dep, pr = [], [], []
    for v in path:
        mod = CRLD(make_env(0.8, **{name: v}), alpha=ALPHA, gamma=GAMMA, prefactor=False)
        for _ in range(600):
            X = mod.step(X)
        imp.append(X[0, IMP, 1, 0].mean()); dep.append(X[0, DEPS, 1, 0].mean())
        pr.append(stationary_prosperity(mod, X)[0])
    out[f'E3b_{name}_path'] = path
    out[f'E3b_{name}_imp'], out[f'E3b_{name}_dep'], out[f'E3b_{name}_prosp'] = map(np.array, (imp, dep, pr))
    print('E3b', name, round(time.time() - t0))

# ---- E4b: temporary scaffold (materiality bridging) then withdrawal --------------------
X0s = random_inits(64, 4, seed=7)
for key, (on, off) in {'k09_0': (0.9, 0.0), 'k09_03': (0.9, 0.3), 'k0_0': (0.0, 0.0),
                       'k05_0': (0.5, 0.0)}.items():
    X = X0s.copy()
    mon = CRLD(make_env(0.8, kappa=on), alpha=ALPHA, gamma=GAMMA, prefactor=False)
    moff = CRLD(make_env(0.8, kappa=off), alpha=ALPHA, gamma=GAMMA, prefactor=False)
    pros, imp, dep = [], [], []
    for t in range(3000):
        mod = mon if t < 1000 else moff
        X = mod.step(X)
        if t % 10 == 0:
            pros.append(stationary_prosperity(mod, X).mean())
            imp.append(X[:, IMP, 1, 0].mean()); dep.append(X[:, DEPS, 1, 0].mean())
    out[f'E4b_{key}_prosp'], out[f'E4b_{key}_imp'], out[f'E4b_{key}_dep'] = map(np.array, (pros, imp, dep))
    print('E4b', key, 'end prosp', pros[-1], 'imp', imp[-1], 'dep', dep[-1])

np.savez(f'results2{SUF}.npz', **out)
print('total', round(time.time() - t0))

import numpy as np
from crld import *
import os
if os.environ.get('BACKEND') == 'pycrld':
    from pycrld_bridge import PyCRLD as CRLD
SUF = os.environ.get('RESULTS_SUFFIX', '')
from experiments import make_env, ALPHA, GAMMA
rng = np.random.default_rng(3)
levels = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5])
out = {'levels': levels}
for name, kw in {'kappa09': dict(kappa=0.9), 'lam30': dict(lam=3.0), 'mix': dict(lam=1.2, kappa=0.4)}.items():
    res = []
    for x0 in levels:
        x = np.clip(rng.uniform(0.5*x0, 1.5*x0, size=(64,4,2)), 1e-3, 1-1e-3)
        X = np.stack([x, 1-x], -1)
        mod = CRLD(make_env(0.8, **kw), alpha=ALPHA, gamma=GAMMA, prefactor=False)
        Xf, tc = mod.run(X, Tmax=10000, tol=1e-5)
        res.append(stationary_prosperity(mod, Xf).mean())
    out[name] = np.array(res); print(name, np.round(res, 2))
np.savez(f'results3{SUF}.npz', **out)

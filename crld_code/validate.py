import numpy as np, time
from crld import *
env = NatureCommons(h=[.5,.5], dep=[1,1])
X0 = random_inits(100, 2, seed=1)
t0=time.time()
for g in [0.6,0.65,0.7,0.72,0.75,0.78,0.8,0.82,0.85,0.88,0.9]:
    mod = CRLD(env, alpha=0.1, gamma=g, prefactor=False)
    X,tc = mod.run(X0, Tmax=20000, tol=1e-5)
    coop = X[:,:,1,0].mean(1)
    print(g, 'fullcoop frac', np.round((coop>0.5).mean(),2), 'median tconv', int(np.median(tc)), 'unconv', (tc==20000).sum())
print(time.time()-t0)

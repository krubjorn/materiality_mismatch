import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from crld import *
import os
if os.environ.get('BACKEND') == 'pycrld':
    from pycrld_bridge import PyCRLD as CRLD
SUF = os.environ.get('RESULTS_SUFFIX', '')
env = NatureCommons(h=[.5,.5], dep=[1,1], qc=0.2)
X0 = random_inits(250, 2, seed=1)
gs = np.round(np.arange(0.5,0.96,0.0125),4); coop=[]; tcs=[]
for g in gs:
    X,tc = CRLD(env, alpha=0.1, gamma=g, prefactor=False).run(X0, Tmax=25000, tol=1e-5)
    coop.append((X[:,:,1,0].mean(1)>0.5).mean()); tcs.append(np.median(tc))
np.savez(f'resultsS1{SUF}.npz', g=gs, coop=coop, t=tcs)
plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
fig,ax=plt.subplots(1,2,figsize=(7.4,2.4),constrained_layout=True)
ax[0].plot(gs,coop,color='#2a78d6',lw=2); ax[0].axvspan(0.7,0.85,color='#e4e3df',zorder=0)
ax[0].set_xlabel('Discount factor γ'); ax[0].set_ylabel('Share converging to\nmutual cooperation'); ax[0].set_title('A',loc='left',weight='bold')
ax[1].plot(gs,tcs,color='#52514e',lw=2); ax[1].axvspan(0.7,0.85,color='#e4e3df',zorder=0)
ax[1].set_xlabel('Discount factor γ'); ax[1].set_ylabel('Median steps to convergence'); ax[1].set_title('B',loc='left',weight='bold')
fig.savefig(f'figs/figS1_replication{SUF}.png'); print(gs[np.argmax(tcs)], max(tcs), [ (g,c) for g,c in zip(gs,coop) if g in (0.7,0.85)])

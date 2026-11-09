import sys, numpy as np, jax.numpy as jnp, jax
from crld import NatureCommons, CRLD, random_inits
from pycrld_bridge import pycrld_env, Runner, to_pycrld_R
from experiments import make_env
print('x64 enabled:', jax.config.jax_enable_x64)
cases = [dict(mu=0.0), dict(mu=0.8), dict(mu=0.8, lam=2.0), dict(mu=0.8, kappa=0.6),
         dict(mu=0.8, tau=0.5), dict(mu=0.5, dep_scale=np.array([1.3, .7, 1.1, .9])), dict(mu=0.8, lam=1.2, kappa=0.4)]
X = random_inits(64, 4, seed=123)
worstT = worstR = worstStep = worstTraj = 0
for kw in cases:
    nc = make_env(**kw)
    e = pycrld_env(nc)
    Rm, Tm = to_pycrld_R(nc)
    dT = np.abs(e.T - Tm).max(); dR = np.abs(e.R - Rm).max()
    mine = CRLD(nc, alpha=0.05, gamma=0.9, prefactor=False)
    r = Runner(nc, alpha=0.05, gamma=0.9, fixed=True)
    dS = np.abs(np.asarray(r.step(jnp.asarray(X)), float) - mine.step(X)).max()
    # 2000-step trajectories
    Xa = X.copy(); Xb = jnp.asarray(X)
    for _ in range(2000):
        Xa = mine.step(Xa); Xb = r.step(Xb)
    dTr = np.abs(np.asarray(Xb, float) - Xa).max()
    worstT, worstR, worstStep, worstTraj = max(worstT, dT), max(worstR, dR), max(worstStep, dS), max(worstTraj, dTr)
    print(kw, f'|dT|={dT:.1e} |dR|={dR:.1e} |dstep|={dS:.1e} |dX after 2000 steps|={dTr:.1e}')
print(f'WORST  T {worstT:.1e}  R {worstR:.1e}  step {worstStep:.1e}  traj {worstTraj:.1e}')

# ---- corrected summation tensor: normalisation for N = 2..5, and N = 2 identical to stock pyCRLD
from pyCRLD.Environments.EcologicalPublicGood import EcologicalPublicGood as EPG
from pyCRLD.Agents.StrategyActorCritic import stratAC
from pycrld_bridge import stratAC_fixed
rng = np.random.default_rng(0)
for N in [2, 3, 4, 5]:
    env = EPG(N=N, f=1.2, c=5, m=-5, qc=0.2, qr=0.01)
    x = rng.uniform(.05, .95, (N, 2)); Xs = jnp.asarray(np.stack([x, 1 - x], -1))
    fx = stratAC_fixed(env=env, learning_rates=0.1, discount_factors=0.9)
    st = stratAC(env=env, learning_rates=0.1, discount_factors=0.9)
    print(f'N={N}: row sums of Tisas  stock {float(st.Tisas(Xs)[0,0,0].sum()):.3f}  corrected {float(fx.Tisas(Xs)[0,0,0].sum()):.3f};'
          f'  |stock - corrected| step {float(jnp.abs(st.step(Xs)[0] - fx.step(Xs)[0]).max()):.1e}')

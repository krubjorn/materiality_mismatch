# Draft issue for github.com/BarfussLab/pyCRLD (not posted)

**Title:** `_OtherAgentsActionsSummationTensor` over-counts other-agent permutations for N ≥ 3 (Tisas rows sum to (N−1)!)

**Version:** main @ a370d51 (20 Nov 2024)

**Summary.** For N ≥ 3 agents, the summation tensor `Omega` built in
`Agents/Base.py::_OtherAgentsActionsSummationTensor` accepts every permutation of the
other agents' indices (the only check is that all indices are distinct). The
strategy-averaged quantities that use `Omega` (`Risa`, `Tisas`, `NextVisa`, and with them
the TD error) are therefore summed (N−1)! times. Permuted terms also pair agent k's
action with agent σ(k)'s strategy. For N = 2 there is one permutation, so all two-agent
results (including the notebooks of Barfuss et al. 2025, PNAS) are unaffected.

**Minimal reproduction**

```python
import numpy as np, jax.numpy as jnp
from pyCRLD.Environments.EcologicalPublicGood import EcologicalPublicGood as EPG
from pyCRLD.Agents.StrategyActorCritic import stratAC

for N in [2, 3, 4, 5]:
    env = EPG(N=N, f=1.2, c=5, m=-5, qc=0.2, qr=0.01)
    ag = stratAC(env=env, learning_rates=0.1, discount_factors=0.9)
    X = jnp.ones((N, 2, 2)) / 2
    print(N, float(ag.Tisas(X)[0, 0, 0].sum()), float(ag.Risa(X)[0, 0, 0]))
# expected: 1.0 and -5.0 for every N
# observed: N=2: 1, -5 | N=3: 2, -10 | N=4: 6, -30 | N=5: 24, -120
```

In the degraded state the reward is the constant `m = -5`, so `Risa` must equal −5 for
any strategy, and rows of `Tisas` must sum to one.

**Suggested fix.** Keep a single canonical ordering of the other agents:

```python
@patch
def _OtherAgentsActionsSummationTensor(self: abase):
    dim = np.concatenate(([self.N], [self.N] * (self.N - 1), [self.M],
                          [self.M] * self.N, [self.M] * (self.N - 1))).astype(int)
    Omega = np.zeros(dim, int)
    for I in range(self.N):
        notI = tuple(k for k in range(self.N) if k != I)   # ascending order only
        for allA in np.ndindex(*([self.M] * self.N)):
            notA = tuple(allA[k] for k in notI)
            Omega[(I,) + notI + (allA[I],) + allA + notA] = 1
    return jnp.array(Omega)
```

With this change:
- rows of `Tisas` sum to 1 for N = 2…5;
- `step` is unchanged for N = 2 (difference 0.0);
- for N = 4, `step` matches an independent brute-force/NumPy implementation of the same
  equations to 1e-15 (float64), and 2,000-step trajectories to 1e-13.

The over-count also makes TD errors large enough that `exp(alpha * TDerror)` overflows in
the default float32, producing NaN strategies for N ≥ 4 in some environments.

Happy to open a PR with the fix and a regression test (`Tisas.sum(-1) == 1` for N = 3, 4).

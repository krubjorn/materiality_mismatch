"""What the unpatched pyCRLD (N=4, (N-1)! over-count) would report for headline configurations."""
import numpy as np, os
from crld import random_inits
import experiments as E
X0 = random_inits(64, 4, seed=2)
cfgs = [('mu=0, accurate pricing', dict(mu=0.0)), ('mu=0.5, accurate pricing', dict(mu=0.5)),
        ('mu=0.8, accurate pricing', dict(mu=0.8)), ('mu=0.8, lam=2.0', dict(mu=0.8, lam=2.0)),
        ('mu=0.8, kappa=0.6', dict(mu=0.8, kappa=0.6)), ('mu=0.8, tau=0.5', dict(mu=0.8, tau=0.5)),
        ('mu=0.8, lam=1.2 + kappa=0.4', dict(mu=0.8, lam=1.2, kappa=0.4))]
rows = []
for lab, kw in cfgs:
    r = E.evaluate(E.make_env(**kw), X0)
    rows.append((lab, r['basin'], r['prosp'], r['unconv']))
    print(f"{lab:32s} basin {r['basin']:.2f}  prosperous {r['prosp']:.2f}  unconverged {r['unconv']}")
np.save(f"stock_sensitivity_{os.environ.get('TAG','x')}.npy", np.array(rows, dtype=object), allow_pickle=True)

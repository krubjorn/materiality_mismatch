# Corporate nature commons: code for "Materiality mismatch traps private nature finance"

Deterministic collective reinforcement-learning dynamics (strategy actor-critic; Barfuss et al. 2019 PRE,
2025 PNAS) applied to a stochastic game of firms with heterogeneous impact (h) and dependency (d).

All reported results are computed with **pyCRLD** (github.com/BarfussLab/pyCRLD, commit a370d51) with a
corrected other-agents summation tensor, and reproduced to machine precision by an independent NumPy
implementation. See `pyCRLD_issue_draft.md` for the N >= 3 issue in the stock library.

## Engines
- `pycrld_bridge.py`  maps the game onto pyCRLD's `EcologicalPublicGood` (firm-specific qc_i = N*qc*h_i, m_i);
                      `EPGTransfer` subclass for payment transfers; `stratAC_fixed` (corrected Omega);
                      `PyCRLD`, a batched drop-in for `crld.CRLD` (jax.vmap of pyCRLD's jitted step)
- `crld.py`           independent NumPy implementation of the same equations

## Checks
- `check_equivalence.py`   tensors, single steps and 2,000-step trajectories, pyCRLD vs NumPy; Tisas row sums N=2..5
- `stock_sensitivity.py`   what the uncorrected library reports for headline configurations
- `compare_numbers.py`     every number quoted in the manuscript from both result sets (writes compare.json)

## Experiments (select engine with environment variables)
    python3 experiments.py                                   # NumPy engine -> results.npz
    BACKEND=pycrld CRLD_X64=1 RESULTS_SUFFIX=_pycrld python3 experiments.py   # pyCRLD -> results_pycrld.npz
- `experiments.py`   E1 lever atlas, E2 thresholds at mu=0.8, E5 information architecture
- `experiments2.py`  E6 policy-mix maps, E3b slow ramp, E4b temporary scaffold
- `experiments3.py`  entrenchment
- `figS1.py`         two-firm replication of PNAS 2025 Fig. 2 (run with PYCRLD_STOCK=1 for the unmodified library)
- `run_pycrld_suite.sh`  the full pyCRLD rerun used for the manuscript
- `figures.py`, `fig1.py`, `build_doc.js`  figures and manuscript

Use double precision (CRLD_X64=1): pyCRLD's default float32 freezes slow strategy drift near basin boundaries.

Requirements: numpy, matplotlib, jax, fastcore (python 3.10+); pyCRLD on PYTHONPATH (or PYCRLD_PATH).
Paper materials of ref. 5: github.com/wbarfuss/collective-cooperative-intelligence (Zenodo 10.5281/zenodo.11611242).

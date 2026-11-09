#!/bin/bash
# Full rerun of every manuscript experiment on pyCRLD (double precision (float64))
cd /home/claude/ncs
export BACKEND=pycrld RESULTS_SUFFIX=_pycrld CRLD_X64=1
echo "S1 (stock, unmodified pyCRLD, N=2)"; PYCRLD_STOCK=1 python3 -u figS1.py 2>&1 | grep -v Warn
echo "experiments3"; python3 -u experiments3.py 2>&1 | grep -v Warn
echo "experiments2"; python3 -u experiments2.py 2>&1 | grep -v Warn
echo "experiments"; python3 -u experiments.py 2>&1 | grep -v Warn
echo SUITE_DONE

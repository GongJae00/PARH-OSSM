"""Run the PARH materializer with the filter's per-observation trust equalized across observations (W5).

Inside the multichannel filter each observation's noise is R_eff = R_t / pi_t, where pi_t is the
reliability-weighted trust of that observation at that frame (parh_ossm.py, "Effective R with prior
trust"). The control replaces pi_t by its mean over the observations at the same frame: the overall
trust the filter places in the observations is unchanged, but no observation is trusted more than
another. Setting pi_t to 1 instead would also raise the overall trust and confound the two. R_t, the
innovation-calibrated noise, is left as it is. The module is re-executed with the patched line before the
materializer imports the class, and its worker processes are forked afterwards.

usage: equal_filter_trust.py <materializer arguments>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import estimate.parh_ossm as parh_module

trust_line = "R_eff = np.asarray(R_t, dtype=np.float64) / np.maximum(np.asarray(pi_t, dtype=np.float64), 1e-6)"
equal_trust_line = "R_eff = np.asarray(R_t, dtype=np.float64) / max(float(np.mean(pi_t)), 1e-6)"
source = open(parh_module.__file__).read()
assert source.count(trust_line) == 1
exec(compile(source.replace(trust_line, equal_trust_line), parh_module.__file__, "exec"), parh_module.__dict__)

import scripts.materialize_calibrated_multifamily_parh_system as materializer

sys.argv = ["materialize_calibrated_multifamily_parh_system.py", *sys.argv[1:]]
sys.exit(materializer.main())

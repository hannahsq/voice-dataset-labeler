"""
test_labeller.py
================
Runner for the labeller regression test suite.

Imports and executes all tests from the split test modules:
  test_labeller_config.py    — R08  LabellerConfig validation
  test_labeller_formants.py  — R01-R03  _vtl_from_formants, _assign_formant_indices
  test_labeller_smoother.py  — R04-R07  VTLSmoother
  test_synth_vowels.py       — R09-R10  synth-based integration

Run individual modules to test a specific area, or run this file to
execute the full suite and print a combined summary.
"""

# test_labeller_registry sets up mocks and populates the shared _results list;
# it must be imported before any test module.
from test_labeller_registry import _print_summary

from test_labeller_config   import test_r08_alpha_clamp
from test_labeller_formants import (
    test_r01_vtl_roundtrip,
    test_r02_nyquist_slots,
    test_r03_slot_ordering,
)
from test_labeller_smoother import (
    test_r04_group_convergence,
    test_r05_smoother_monotonicity,
    test_r06_speaker_anchors_to_group,
    test_r07_override_bypasses_group,
)
from test_synth_vowels import (
    test_r09_synth_vtl_from_assigned,
    test_r10_synth_group_estimate,
)


if __name__ == "__main__":
    test_r01_vtl_roundtrip()
    test_r02_nyquist_slots()
    test_r03_slot_ordering()
    test_r04_group_convergence()
    test_r05_smoother_monotonicity()
    test_r06_speaker_anchors_to_group()
    test_r07_override_bypasses_group()
    test_r08_alpha_clamp()
    test_r09_synth_vtl_from_assigned()
    test_r10_synth_group_estimate()

    n_fail = _print_summary()
    raise SystemExit(0 if n_fail == 0 else 1)

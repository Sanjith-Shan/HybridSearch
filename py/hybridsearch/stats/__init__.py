"""Significance testing for paired per-query IR metrics (M4)."""
from .significance import (  # noqa: F401
    Comparison,
    align,
    compare,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_randomization_test,
)

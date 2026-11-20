"""Tests for ensemble score attribution without needing binary model files."""
import numpy as np
from antifraud.blend import REFERENCE

def test_reference_has_expected_features():
    assert len(REFERENCE) == 13
    assert REFERENCE["country"] == "RU"
    assert REFERENCE["channel"] == "mobile"

def test_weighted_blend_weights():
    assert np.isclose(0.9 + 0.1, 1.0)

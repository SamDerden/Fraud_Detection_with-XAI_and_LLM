from pathlib import Path
import importlib.util
import numpy as np
spec=importlib.util.spec_from_file_location('drift_v3',Path(__file__).resolve().parents[1]/'experiments'/'webinar13_drift_v3.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_psi_identical():
    x=np.arange(100,dtype=float)
    assert module.psi(x,x)==0

def test_psi_shift():
    x=np.arange(100,dtype=float)
    assert module.psi(x,x+200)>0.2

def test_invalid_samples():
    import pytest
    with pytest.raises(ValueError):module.psi([1,2],[3,4])

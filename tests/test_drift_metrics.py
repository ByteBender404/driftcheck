import pytest
import pandas as pd
import numpy as np
from driftcheck.drift_metrics import calculate_psi, get_severity, get_severity_score

def test_psi_identical_numeric():
    # Identical distributions should have PSI ~ 0
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = expected.copy()
    
    psi = calculate_psi(expected, actual, is_categorical=False)
    assert psi < 0.01

def test_psi_shifted_numeric():
    # Shifted distributions should have high PSI
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = pd.Series(np.random.normal(2, 1, 1000))
    
    psi = calculate_psi(expected, actual, is_categorical=False)
    assert psi > 0.25

def test_psi_categorical_identical():
    expected = pd.Series(['A', 'A', 'B', 'C'])
    actual = pd.Series(['A', 'A', 'B', 'C'])
    
    psi = calculate_psi(expected, actual, is_categorical=True)
    assert psi < 0.01
    
def test_psi_categorical_drifted():
    expected = pd.Series(['A'] * 80 + ['B'] * 20)
    actual = pd.Series(['A'] * 20 + ['B'] * 80)
    
    psi = calculate_psi(expected, actual, is_categorical=True)
    assert psi > 0.25

def test_get_severity():
    assert get_severity(0.05) == "Low"
    assert get_severity(0.15) == "Medium"
    assert get_severity(0.3) == "High"

def test_severity_score():
    assert get_severity_score(0) == 0
    assert 0 <= get_severity_score(0.05) <= 33
    assert 34 <= get_severity_score(0.2) <= 66
    assert get_severity_score(0.3) > 66
    assert get_severity_score(1.0) == 100

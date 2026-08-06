import numpy as pd
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare
from typing import Dict, Any, Tuple

def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10, is_categorical: bool = False) -> float:
    """
    Calculates the Population Stability Index (PSI).
    
    WHAT IS PSI?
    PSI is a metric that measures how much a population's distribution has shifted over time.
    It compares an "expected" (old) distribution against an "actual" (new) distribution.
    - PSI < 0.1: Low drift (No significant change)
    - 0.1 <= PSI <= 0.25: Medium drift (Some change, might require monitoring)
    - PSI > 0.25: High drift (Significant change, action recommended)
    
    For numeric data, it divides the expected data into bins (e.g. deciles), counts the proportion
    of expected and actual data in those bins, and compares them.
    For categorical data, the bins are simply the unique categories.
    """
    if is_categorical:
        # Get all unique categories across both expected and actual
        categories = set(expected.dropna().unique()).union(set(actual.dropna().unique()))
        
        expected_counts = expected.value_counts().reindex(list(categories), fill_value=0)
        actual_counts = actual.value_counts().reindex(list(categories), fill_value=0)
    else:
        # Numeric: calculate bins based on expected distribution
        # Use quantiles to get equal-sized bins where possible
        q_bins = np.linspace(0, 1, bins + 1)
        bin_edges = expected.dropna().quantile(q_bins).values
        # Ensure bin edges are strictly increasing to avoid ValueError in digitize/histogram
        bin_edges = np.unique(bin_edges)
        
        if len(bin_edges) < 2:
            # Fallback if too few unique values
            bin_edges = np.histogram_bin_edges(expected.dropna(), bins=bins)
            
        # Add a tiny amount to highest bin edge and subtract from lowest to capture all values
        bin_edges[0] -= 0.001
        bin_edges[-1] += 0.001
        
        expected_counts, _ = np.histogram(expected.dropna(), bins=bin_edges)
        actual_counts, _ = np.histogram(actual.dropna(), bins=bin_edges)
        
        expected_counts = pd.Series(expected_counts)
        actual_counts = pd.Series(actual_counts)

    # Convert counts to proportions
    expected_props = expected_counts / expected_counts.sum()
    actual_props = actual_counts / actual_counts.sum()
    
    # Replace zeros with a tiny number to avoid division by zero or log(0)
    epsilon = 0.0001
    expected_props = expected_props.replace(0, epsilon)
    actual_props = actual_props.replace(0, epsilon)
    
    # Calculate PSI
    psi_values = (actual_props - expected_props) * np.log(actual_props / expected_props)
    
    return float(psi_values.sum())

def get_severity(psi_score: float) -> str:
    """Returns a categorical severity label based on standard PSI thresholds."""
    if psi_score < 0.1:
        return "Low"
    elif psi_score <= 0.25:
        return "Medium"
    else:
        return "High"

def get_severity_score(psi_score: float) -> int:
    """
    Returns a normalized 0-100 severity score.
    Maps [0, 0.1, 0.25, 0.5+] to ~[0, 30, 70, 100].
    """
    if psi_score <= 0:
        return 0
    elif psi_score >= 0.5:
        return 100
    elif psi_score < 0.1:
        return int((psi_score / 0.1) * 33)
    elif psi_score <= 0.25:
        return int(33 + ((psi_score - 0.1) / 0.15) * 33)
    else:
        return int(66 + ((psi_score - 0.25) / 0.25) * 34)

def run_ks_test(expected: pd.Series, actual: pd.Series) -> Dict[str, float]:
    """
    Runs the Kolmogorov-Smirnov (KS) test for numeric distributions.
    
    WHAT IS THE KS TEST?
    It's a statistical test that compares the cumulative distributions of two datasets.
    It returns a 'statistic' (max distance between the curves, 0 to 1) and a 'p-value'.
    A low p-value (e.g. < 0.05) strongly suggests the two datasets come from different distributions.
    """
    # Drop missing values
    e_clean = expected.dropna()
    a_clean = actual.dropna()
    
    if len(e_clean) == 0 or len(a_clean) == 0:
        return {"statistic": 0.0, "p_value": 1.0}
        
    stat, p_val = ks_2samp(e_clean, a_clean)
    return {"statistic": float(stat), "p_value": float(p_val)}

def run_chi_square_test(expected: pd.Series, actual: pd.Series) -> Dict[str, float]:
    """
    Runs the Chi-Squared test for categorical distributions.
    
    WHAT IS THE CHI-SQUARED TEST?
    It compares the observed frequencies of categories (in the actual data) against
    the expected frequencies (from the old data).
    A low p-value (e.g. < 0.05) suggests the categorical distribution has significantly changed.
    """
    e_clean = expected.dropna()
    a_clean = actual.dropna()
    
    if len(e_clean) == 0 or len(a_clean) == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    categories = set(e_clean.unique()).union(set(a_clean.unique()))
    
    # Get raw counts
    e_counts = e_clean.value_counts().reindex(list(categories), fill_value=0)
    a_counts = a_clean.value_counts().reindex(list(categories), fill_value=0)
    
    # Scale expected counts to match actual sample size for the test
    total_expected = e_counts.sum()
    total_actual = a_counts.sum()
    
    if total_expected == 0 or total_actual == 0:
         return {"statistic": 0.0, "p_value": 1.0}
         
    scaling_factor = total_actual / total_expected
    e_counts_scaled = e_counts * scaling_factor
    
    # Add a tiny epsilon to avoid zero expected frequencies
    epsilon = 1e-8
    e_counts_scaled = e_counts_scaled + epsilon
    a_counts = a_counts + epsilon

    stat, p_val = chisquare(f_obs=a_counts, f_exp=e_counts_scaled)
    return {"statistic": float(stat), "p_value": float(p_val)}

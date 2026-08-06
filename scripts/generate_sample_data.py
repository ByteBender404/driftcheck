import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Old Dataset (Expected)
    old_data = {
        "id": range(n_samples),
        "age": np.random.normal(loc=35, scale=10, size=n_samples), # Normal distribution
        "income": np.random.lognormal(mean=10, sigma=1, size=n_samples),
        "category": np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.5, 0.3, 0.2]),
        "status": np.random.choice(['Active', 'Inactive'], size=n_samples, p=[0.8, 0.2]),
    }
    df_old = pd.DataFrame(old_data)
    
    # New Dataset (Actual) - Injecting Drift
    new_data = {
        "id": range(n_samples, n_samples * 2),
        "age": np.random.normal(loc=45, scale=12, size=n_samples), # Drifted: Mean shift + variance
        "income": np.random.lognormal(mean=10, sigma=1, size=n_samples), # No drift
        "category": np.random.choice(['A', 'B', 'C', 'D'], size=n_samples, p=[0.2, 0.2, 0.1, 0.5]), # Drifted: new category and changed probs
        "status": np.random.choice(['Active', 'Inactive'], size=n_samples, p=[0.75, 0.25]), # Low/No drift
    }
    df_new = pd.DataFrame(new_data)
    
    # Introduce some missing values to age in new dataset
    df_new.loc[np.random.choice(n_samples, size=50, replace=False), "age"] = np.nan
    
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    old_path = root_dir / "old_sample.csv"
    new_path = root_dir / "new_sample.csv"
    
    df_old.to_csv(old_path, index=False)
    df_new.to_csv(new_path, index=False)
    
    print(f"Generated {old_path} and {new_path}")

if __name__ == "__main__":
    generate_data()

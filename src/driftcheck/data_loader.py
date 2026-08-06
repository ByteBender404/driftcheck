import pandas as pd
from typing import Tuple, Dict, Any

def load_data(file_path: str) -> pd.DataFrame:
    """Loads a CSV or Parquet file into a pandas DataFrame."""
    if file_path.lower().endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.lower().endswith('.parquet') or file_path.lower().endswith('.pq'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Use .csv or .parquet")

def get_missing_percentages(df: pd.DataFrame) -> Dict[str, float]:
    """Calculates the percentage of missing values for each column."""
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    return missing_percentages.to_dict()

def classify_columns(df: pd.DataFrame, max_categories: int = 100) -> Tuple[list, list, Dict[str, str]]:
    """
    Classifies columns into numeric, categorical, and excluded (identifiers).
    Even numeric types are considered categorical if they have a small number of unique values.
    """
    numeric_cols = []
    categorical_cols = []
    excluded_cols = {}
    
    row_count = len(df)
    id_patterns = {"id", "_id", "uuid", "index"}
    
    for col in df.columns:
        col_lower = col.lower()
        nunique = df[col].nunique()
        
        # Check for identifiers
        if col_lower in id_patterns:
            excluded_cols[col] = "Matches ID naming pattern"
            continue
        elif nunique == row_count and row_count > 0:
            if not pd.api.types.is_float_dtype(df[col]):
                excluded_cols[col] = "Unique value for every row (cardinality == row count)"
                continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's actually categorical pretending to be numeric (e.g. 0/1, ids, etc.)
            if nunique <= max_categories and nunique < row_count * 0.1:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    return numeric_cols, categorical_cols, excluded_cols

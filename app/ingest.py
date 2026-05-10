import pandas as pd
import numpy as np
from pathlib import Path


def load_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use CSV or Excel.")


def load_from_path(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported format: {path.suffix}")



def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return a comprehensive profile of the dataframe."""
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)

    col_profiles = []
    for col in df.columns:
        entry = {
            "column":   col,
            "dtype":    str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "n_null":   int(null_counts[col]),
            "pct_null": float(null_pct[col]),
        }
        if col in num_cols and not df[col].isnull().all():
            entry.update({
                "mean": round(float(df[col].mean()), 4),
                "std":  round(float(df[col].std()),  4),
                "min":  round(float(df[col].min()),  4),
                "max":  round(float(df[col].max()),  4),
            })
        col_profiles.append(entry)

    return {
        "rows":             len(df),
        "cols":             len(df.columns),
        "numeric_cols":     num_cols,
        "cat_cols":         cat_cols,
        "bool_cols":        bool_cols,
        "null_pct_avg":     round(float(null_pct.mean()), 2),
        "null_pct_by_col":  null_pct.to_dict(),
        "null_count_by_col": null_counts.to_dict(),
        "duplicates":       int(df.duplicated().sum()),
        "memory_mb":        round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "col_profiles":     col_profiles,
    }


def validate_target(df: pd.DataFrame, target: str) -> dict:
    """Validate the chosen target column and return statistics."""
    if target not in df.columns:
        return {"valid": False, "error": f"Column '{target}' not found."}
    series = df[target].dropna()
    if len(series) == 0:
        return {"valid": False, "error": "Target column is entirely null."}
    n_unique   = series.nunique()
    null_count = int(df[target].isnull().sum())
    task_hint  = "classification" if (series.dtype == object or n_unique <= 20) else "regression"
    return {
        "valid":       True,
        "n_unique":    n_unique,
        "null_count":  null_count,
        "null_pct":    round(null_count / len(df) * 100, 2),
        "task_hint":   task_hint,
        "value_counts": series.value_counts().head(10).to_dict() if task_hint == "classification" else None,
        "mean": round(float(series.mean()), 4) if task_hint == "regression" else None,
        "std":  round(float(series.std()),  4) if task_hint == "regression" else None,
    }
import pandas as pd
import numpy as np
import pickle
import json
import io
from pathlib import Path
from datetime import datetime


def detect_task_type(series: pd.Series) -> str:
    if series.dtype == object:
        return "classification"
    if series.nunique() <= 20:
        return "classification"
    return "regression"


def get_feature_names(preprocessor, X: pd.DataFrame) -> list:
    """Recover human-readable feature names after ColumnTransformer."""
    names = []
    for t_name, transformer, cols in preprocessor.transformers_:
        if t_name == "remainder":
            continue
        if t_name == "num":
            names.extend(cols)
        elif t_name == "cat":
            try:
                enc = transformer.named_steps["encode"]
                names.extend(enc.get_feature_names_out(cols).tolist())
            except Exception:
                names.extend(cols)
    return names



def results_to_dataframe(results: list, task: str) -> pd.DataFrame:
    exclude = {"_model", "_pred"}
    rows = [{k: v for k, v in r.items() if k not in exclude} for r in results]
    return pd.DataFrame(rows)


def export_predictions(y_test, y_pred, label_encoder=None) -> str:
    """Return predictions as a CSV string."""
    actual = y_test
    predicted = y_pred
    if label_encoder is not None:
        try:
            actual    = label_encoder.inverse_transform(y_test)
            predicted = label_encoder.inverse_transform(y_pred)
        except Exception:
            pass
    df = pd.DataFrame({"actual": actual, "predicted": predicted})
    return df.to_csv(index=False)


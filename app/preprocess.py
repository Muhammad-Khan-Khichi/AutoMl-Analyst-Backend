import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings("ignore")



# ─────────────────────────────────────────────
#  PIPELINE BUILDER
# ─────────────────────────────────────────────

def build_pipeline(df: pd.DataFrame, target: str,test_size: float = 0.2,scaler: str = "standard",random_state: int = 42):
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    valid_mask = ~y.isnull()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    # Encode target for classification
    le = None
    is_classification = (y.dtype == object or y.nunique() <= 20)
    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    # Column type detection
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = X.select_dtypes(include="bool").columns.tolist()

    # Convert bool to int
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(int)
        num_cols = num_cols + bool_cols

    # Choose scaler
    scaler_map = {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "robust":   RobustScaler(),
    }
    chosen_scaler = scaler_map.get(scaler, StandardScaler())

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),("scale",  chosen_scaler)]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
        ]), cat_cols))

    if not transformers:
        raise ValueError("No usable columns found after type detection.")

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    # Stratified split for classification
    stratify = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return preprocessor, X_train, X_test, y_train, y_test, le


# ─────────────────────────────────────────────
#  FEATURE NAME RECOVERY
# ─────────────────────────────────────────────



def summarize_preprocessing(preprocessor, X_train, X_test,
                             X_train_t, feature_names) -> dict:
    return {
        "original_features": X_train.shape[1],
        "transformed_features": X_train_t.shape[1],
        "train_rows": len(X_train),
        "test_rows":  len(X_test),
        "feature_names": feature_names,
    }
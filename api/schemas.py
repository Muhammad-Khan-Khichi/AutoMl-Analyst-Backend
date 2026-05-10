"""
schemas.py — Pydantic request/response models for the FastAPI server.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class TaskType(str, Enum):
    classification = "classification"
    regression     = "regression"


# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single-row prediction request."""
    features: Dict[str, Any] = Field(
        ...,
        description="Feature values as a column_name → value mapping",
        example={"age": 29, "fare": 7.25, "pclass": 3, "sex": "male"},
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": {"age": 29, "fare": 7.25, "pclass": 3, "sex": "male"}
            }
        }


class BatchPredictRequest(BaseModel):
    """Multi-row batch prediction request."""
    rows: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dicts, one per row",
        min_items=1,
        max_items=10000,
    )


class PredictResponse(BaseModel):
    prediction:  Any              # scalar or list
    probability: Optional[Any]   # class probabilities if available
    model_name:  str
    task_type:   str


class BatchPredictResponse(BaseModel):
    predictions:   List[Any]
    probabilities: Optional[List[Any]] = None
    model_name:    str
    task_type:     str
    n_rows:        int


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────

class TrainRequest(BaseModel):
    """Trigger model training from a pre-loaded dataset."""
    dataset_path: str = Field(..., description="Path to CSV/Excel in data/raw/")
    target:       str = Field(..., description="Target column name")
    task:         TaskType = TaskType.classification
    test_size:    float = Field(0.2, ge=0.05, le=0.5)
    cv_folds:     int   = Field(5,   ge=2,    le=10)
    scaler:       str   = Field("standard", pattern="^(standard|minmax|robust)$")
    exclude_cols: List[str] = Field(default_factory=list)


class ModelResult(BaseModel):
    model:      str
    score:      float
    cv_mean:    float
    cv_std:     float
    train_time: float


class TrainResponse(BaseModel):
    status:          str
    best_model:      str
    best_score:      float
    metric:          str
    all_results:     List[ModelResult]
    model_saved_to:  str
    train_rows:      int
    test_rows:       int
    n_features:      int


# ─────────────────────────────────────────────
#  HEALTH / INFO
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_name:   Optional[str] = None
    task_type:    Optional[str] = None
    api_version:  str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model_name:    str
    task_type:     str
    n_features:    int
    feature_names: Optional[List[str]] = None
    model_path:    str
    label_classes: Optional[List[str]] = None


# ─────────────────────────────────────────────
#  ERROR
# ─────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None
    status:  int = 500
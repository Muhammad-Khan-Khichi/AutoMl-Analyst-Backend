import io
import json
import pickle
import traceback
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingest import load_from_path, profile_dataframe, validate_target
from app.preprocess import build_pipeline, summarize_preprocessing
from app.automl import run_automl, save_model, load_model, get_classification_report
from app.utils import results_to_dataframe, export_predictions, get_feature_names

UPLOAD_DIR = Path("uploads")
MODEL_DIR  = Path("models/saved")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


SESSIONS: dict[str, dict] = {}



app = FastAPI(
    title="AutoML Platform API",
    description="Upload a dataset, run AutoML, get predictions and explainability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Upload a file first.")
    return SESSIONS[session_id]


def _df_from_session(session_id: str) -> pd.DataFrame:
    session = _get_session(session_id)
    path = session.get("file_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Dataset file not found for this session.")
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "AutoML Platform API is running 🚀"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

@app.post("/upload", tags=["Dataset"], summary="Upload a CSV or Excel file")
async def upload_dataset(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use .csv, .xlsx or .xls."
        )

    session_id = str(uuid.uuid4())
    save_path  = UPLOAD_DIR / f"{session_id}{suffix}"

    contents = await file.read()
    save_path.write_bytes(contents)

    # Parse & profile
    try:
        if suffix == ".csv":
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    profile = profile_dataframe(df)

    SESSIONS[session_id] = {
        "file_path":  str(save_path),
        "filename":   file.filename,
        "profile":    profile,
        "columns":    df.columns.tolist(),
    }

    return {
        "session_id": session_id,
        "filename":   file.filename,
        "profile": {
            "rows":          profile["rows"],
            "cols":          profile["cols"],
            "numeric_cols":  profile["numeric_cols"],
            "cat_cols":      profile["cat_cols"],
            "null_pct_avg":  profile["null_pct_avg"],
            "duplicates":    profile["duplicates"],
            "memory_mb":     profile["memory_mb"],
            "col_profiles":  profile["col_profiles"],
        },
        "columns": df.columns.tolist(),
    }



@app.get("/dataset/{session_id}/profile", tags=["Dataset"], summary="Get dataset profile")
def get_profile(session_id: str):
    session = _get_session(session_id)
    return session["profile"]


@app.get("/dataset/{session_id}/validate-target",tags=["Dataset"],summary="Validate a target column",)
def validate_target_col(session_id: str, target: str):
    df = _df_from_session(session_id)
    result = validate_target(df, target)
    if not result["valid"]:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/train/{session_id}", tags=["Training"], summary="Train all AutoML models")
def train(
    session_id: str,
    target:     str  = Form(...,  description="Target column name"),
    task:       str  = Form("auto", description="'classification', 'regression', or 'auto'"),
    test_size:  float = Form(0.2,  description="Fraction held out for test (0.1 – 0.4)"),
    cv_folds:   int  = Form(5,    description="Cross-validation folds (2 – 10)"),
    scaler:     str  = Form("standard", description="'standard', 'minmax', or 'robust'"),
):
    df = _df_from_session(session_id)

    if task not in {"classification", "regression"}:
        raise HTTPException(status_code=400, detail="task must be 'classification', 'regression', or 'auto'")

    # Validate
    val = validate_target(df, target)
    if not val["valid"]:
        raise HTTPException(status_code=422, detail=val["error"])

    # Preprocess
    try:
        preprocessor, X_train, X_test, y_train, y_test, le = build_pipeline(
            df, target, test_size=test_size, scaler=scaler
        )
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t  = preprocessor.transform(X_test)
        feature_names = get_feature_names(preprocessor, X_train)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    prep_summary = summarize_preprocessing(preprocessor, X_train, X_test, X_train_t, feature_names)

    # Train
    try:
        results = run_automl(
            X_train_t, X_test_t, y_train, y_test,
            task=task, cv=cv_folds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {traceback.format_exc()}")

    if not results:
        raise HTTPException(status_code=500, detail="All models failed. Check your data.")

    best = results[0]

    # Persist model
    model_path = str(MODEL_DIR / f"best_model_{session_id}.pkl")
    save_model(best["_model"], model_path)

    # Persist training state for later endpoints
    SESSIONS[session_id].update({
        "target":        target,
        "task":          task,
        "feature_names": feature_names,
        "le_classes":    le.classes_.tolist() if le else None,
        "model_path":    model_path,
        "results":       results,
        "X_test_t":      X_test_t,
        "y_test":        y_test,
        "y_pred":        best["_pred"],
        "prep_summary":  prep_summary,
    })

    # Build leaderboard (no private keys)
    score_key = "accuracy" if task == "classification" else "r2"
    leaderboard = results_to_dataframe(results, task).to_dict(orient="records")

    best_public = {k: v for k, v in best.items() if not k.startswith("_")}

    return {
        "session_id":      session_id,
        "task":            task,
        "target":          target,
        "preprocessing":   prep_summary,
        "best_model":      best_public,
        "leaderboard":     leaderboard,
        "models_tested":   len(results),
        "model_saved_at":  model_path,
    }



@app.get("/results/{session_id}", tags=["Results"], summary="Get training leaderboard")
def get_results(session_id: str):
    """Return the ranked leaderboard from the last training run."""
    session = _get_session(session_id)
    if "results" not in session:
        raise HTTPException(status_code=404, detail="No training results yet. Call POST /train first.")
    task       = session["task"]
    leaderboard = results_to_dataframe(session["results"], task).to_dict(orient="records")
    return {"task": task, "leaderboard": leaderboard}

@app.get(
    "/results/{session_id}/feature-importance",
    tags=["Results"],
    summary="Feature importances for the best model",
)
def feature_importance(session_id: str, top_n: int = 20):
    """Return the top-N feature importances (tree-based models only)."""
    session    = _get_session(session_id)
    if "results" not in session:
        raise HTTPException(status_code=404, detail="No training results yet.")

    best_model = session["results"][0]["_model"]
    names      = session["feature_names"]

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        # Linear models — use absolute coefficient values
        coef = best_model.coef_
        if coef.ndim > 1:
            importances = abs(coef).mean(axis=0)  # multi-class: average across classes
        else:
            importances = abs(coef)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{session['results'][0]['model']}' does not support feature importances."
        )
    pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    return {
        "model":    session["results"][0]["model"],
        "top_n":    top_n,
        "features": [{"feature": f, "importance": round(float(i), 6)} for f, i in pairs],
    }


@app.get(
    "/results/{session_id}/classification-report",
    tags=["Results"],
    summary="Detailed per-class metrics (classification only)",
)
def classification_report_endpoint(session_id: str):
    session = _get_session(session_id)
    if "results" not in session:
        raise HTTPException(status_code=404, detail="No training results yet.")
    if session["task"] != "classification":
        raise HTTPException(status_code=400, detail="This endpoint is for classification tasks only.")

    le_classes = session.get("le_classes")
    from sklearn.preprocessing import LabelEncoder
    le = None
    if le_classes:
        le = LabelEncoder()
        le.classes_ = np.array(le_classes)

    report = get_classification_report(session["y_test"], session["y_pred"], label_encoder=le)
    return report


@app.post(
    "/predict/{session_id}",
    tags=["Prediction"],
    summary="Run predictions on new rows (JSON)",
)
def predict(session_id: str, data: list[dict]):
    session = _get_session(session_id)
    if "model_path" not in session:
        raise HTTPException(status_code=404, detail="No trained model found. Call POST /train first.")

    model = load_model(session["model_path"])
    df_new = pd.DataFrame(data)

    try:
        preds = model.predict(df_new.values.astype(float))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    le_classes = session.get("le_classes")
    if le_classes:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(le_classes)
        try:
            preds = le.inverse_transform(preds.astype(int)).tolist()
        except Exception:
            preds = preds.tolist()
    else:
        preds = preds.tolist()

    return {"predictions": preds, "count": len(preds)}



@app.get(
    "/export/{session_id}/predictions",
    tags=["Export"],
    summary="Download predictions as CSV",
)
def export_predictions_csv(session_id: str):
    session = _get_session(session_id)
    if "y_test" not in session:
        raise HTTPException(status_code=404, detail="No predictions yet. Call POST /train first.")

    le_classes = session.get("le_classes")
    le = None
    if le_classes:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(le_classes)

    csv_str = export_predictions(session["y_test"], session["y_pred"], label_encoder=le)
    return StreamingResponse(
        io.StringIO(csv_str),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.get(
    "/export/{session_id}/leaderboard",
    tags=["Export"],
    summary="Download model leaderboard as CSV",
)
def export_leaderboard_csv(session_id: str):
    session = _get_session(session_id)
    if "results" not in session:
        raise HTTPException(status_code=404, detail="No training results yet.")

    df_lb  = results_to_dataframe(session["results"], session["task"])
    csv_io = io.StringIO()
    df_lb.to_csv(csv_io, index=False)
    csv_io.seek(0)
    return StreamingResponse(
        csv_io,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=leaderboard.csv"},
    )


@app.get(
    "/export/{session_id}/model",
    tags=["Export"],
    summary="Download the best trained model as a .pkl file",
)
def download_model(session_id: str):
    """Download the best model as a pickle file."""
    session = _get_session(session_id)
    model_path = session.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="No trained model found. Call POST /train first.")
    model_name = session.get("results", [{}])[0].get("model", "best_model").replace(" ", "_")
    return StreamingResponse(
        open(model_path, "rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={model_name}.pkl"},
    )


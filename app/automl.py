import numpy as np
import pickle
import time
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report,
)

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  MODEL REGISTRIES
# ─────────────────────────────────────────────

CLASSIFICATION_MODELS = {
    "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Extra Trees":            ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Decision Tree":          DecisionTreeClassifier(random_state=42),
    "AdaBoost":               AdaBoostClassifier(n_estimators=100, random_state=42),
    "KNN":                    KNeighborsClassifier(n_neighbors=5),
    "SVM":                    SVC(probability=True, random_state=42),
    "Naive Bayes":            GaussianNB(),
}

REGRESSION_MODELS = {
    "Linear Regression":      LinearRegression(),
    "Ridge":                  Ridge(random_state=42),
    "Lasso":                  Lasso(random_state=42),
    "ElasticNet":             ElasticNet(random_state=42),
    "Random Forest":          RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Extra Trees":            ExtraTreesRegressor(n_estimators=100, random_state=42),
    "Decision Tree":          DecisionTreeRegressor(random_state=42),
    "AdaBoost":               AdaBoostRegressor(n_estimators=100, random_state=42),
    "KNN":                    KNeighborsRegressor(n_neighbors=5),
    "SVR":                    SVR(),
}


# ─────────────────────────────────────────────
#  MAIN AutoML FUNCTION
# ─────────────────────────────────────────────

def run_automl(X_train, X_test, y_train, y_test,task: str = "classification",cv: int = 5,progress_callback=None) -> list:

    models = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    results = []
    total = len(models)

    for i, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(i, total, name)
        try:
            t0 = time.time()
            model.fit(X_train, y_train)
            elapsed = round(time.time() - t0, 2)
            y_pred = model.predict(X_test)

            if task == "classification":
                n_classes = len(np.unique(y_train))
                acc  = accuracy_score(y_test, y_pred)
                f1   = f1_score(y_test, y_pred,
                                average="binary" if n_classes == 2 else "weighted",
                                zero_division=0)
                # AUC only when predict_proba is available
                auc = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X_test)
                        if n_classes == 2:
                            auc = round(roc_auc_score(y_test, proba[:, 1]), 4)
                        else:
                            auc = round(roc_auc_score(
                                y_test, proba, multi_class="ovr", average="weighted"), 4)
                    except Exception:
                        auc = None

                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring="accuracy")

                results.append({
                    "model":       name,
                    "accuracy":    round(acc, 4),
                    "f1_score":    round(f1, 4),
                    "auc_roc":     auc,
                    "cv_mean":     round(cv_scores.mean(), 4),
                    "cv_std":      round(cv_scores.std(), 4),
                    "train_time":  elapsed,
                    "_model":      model,
                    "_pred":       y_pred,
                })
            else:
                r2   = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae  = mean_absolute_error(y_test, y_pred)
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring="r2")

                results.append({
                    "model":      name,
                    "r2":         round(r2, 4),
                    "rmse":       round(rmse, 4),
                    "mae":        round(mae, 4),
                    "cv_mean":    round(cv_scores.mean(), 4),
                    "cv_std":     round(cv_scores.std(), 4),
                    "train_time": elapsed,
                    "_model":     model,
                    "_pred":      y_pred,
                })
        except Exception as e:
            print(f"[AutoML] Skipping {name}: {e}")

    if progress_callback:
        progress_callback(total, total, "Done")

    # Sort by primary metric
    key = "accuracy" if task == "classification" else "r2"
    results.sort(key=lambda r: r.get(key, -999), reverse=True)
    return results


# ─────────────────────────────────────────────
#  SAVE / LOAD BEST MODEL
# ─────────────────────────────────────────────

def save_model(model, path: str = "models/saved/best_model.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: str = "models/saved/best_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
#  DETAILED CLASSIFICATION REPORT
# ─────────────────────────────────────────────

def get_classification_report(y_test, y_pred, label_encoder=None) -> dict:
    target_names = None
    if label_encoder is not None:
        target_names = [str(c) for c in label_encoder.classes_]
    return classification_report(
        y_test, y_pred, target_names=target_names,
        output_dict=True, zero_division=0
    )
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_curve, auc,
)
import warnings
warnings.filterwarnings("ignore")

# ─── Dark theme consistent with the UI ───────
DARK = {
    "figure.facecolor":  "#0b0d12",
    "axes.facecolor":    "#0f1117",
    "axes.edgecolor":    "#1e2234",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#e2e8f0",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "text.color":        "#94a3b8",
    "grid.color":        "#1e2234",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#0f1117",
    "legend.edgecolor":  "#1e2234",
}
ACCENT   = "#6366f1"
ACCENT2  = "#38bdf8"
ACCENT3  = "#f472b6"
PALETTE  = [ACCENT, ACCENT2, "#34d399", "#fb923c", ACCENT3, "#a78bfa", "#fbbf24", "#4ade80"]


def _fig(w=9, h=5):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0b0d12")
    return fig, ax


def _fig_multi(nrows, ncols, w, h):
    with plt.rc_context(DARK):
        fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor("#0b0d12")
    return fig, axes


# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def feature_importance_chart(model, feature_names: list, top_n: int = 20):
    if not hasattr(model, "feature_importances_"):
        return None

    fi  = model.feature_importances_
    idx = np.argsort(fi)[::-1][:top_n]
    features = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    values   = fi[idx]

    fig, ax = _fig(9, max(4, len(features) * 0.42))
    cmap = plt.colormaps["plasma"]
    colors  = [cmap(0.2 + 0.6 * (1 - v / values.max())) for v in values]

    bars = ax.barh(features[::-1], values[::-1], color=colors[::-1],
                   edgecolor="#0b0d12", height=0.65)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#94a3b8")

    ax.set_title(f"Top {len(features)} Feature Importances", pad=12)
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, values.max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  CLASSIFICATION CHARTS
# ─────────────────────────────────────────────

def confusion_matrix_chart(y_true, y_pred, labels=None):
    """Styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = _fig(5.5, 4.5)

    cmap = sns.light_palette(ACCENT, as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
        linewidths=0.5, linecolor="#0b0d12",
        annot_kws={"size": 13, "weight": "bold", "color": "#e2e8f0"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label",      labelpad=10)
    ax.set_title("Confusion Matrix", pad=12)
    fig.tight_layout()
    return fig


def per_class_metrics_chart(y_true, y_pred, labels=None):
    """Heatmap of precision / recall / f1 per class."""
    from sklearn.metrics import classification_report as cr
    report = cr(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    df_rep = pd.DataFrame(report).T
    df_rep = df_rep[~df_rep.index.isin(["accuracy", "macro avg", "weighted avg"])]
    df_rep = df_rep[["precision", "recall", "f1-score"]].astype(float).round(3)

    fig, ax = _fig(max(5, len(df_rep) * 0.8 + 2), max(3, len(df_rep) * 0.55 + 1))
    cmap = sns.light_palette("#38bdf8", as_cmap=True)
    sns.heatmap(
        df_rep, annot=True, fmt=".3f", cmap=cmap,
        vmin=0, vmax=1, linewidths=0.5, linecolor="#0b0d12",
        annot_kws={"size": 11, "color": "#e2e8f0"},
        ax=ax,
    )
    ax.set_title("Per-Class Metrics", pad=12)
    fig.tight_layout()
    return fig


def roc_curve_chart(y_true, model, X_test, label_encoder=None):
    """ROC curve for binary classification."""
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X_test)
        n_classes = proba.shape[1]
        fig, ax = _fig(6, 5)
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=ACCENT, lw=2.5,
                    label=f"ROC (AUC = {roc_auc:.3f})")
            ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
        else:
            from sklearn.preprocessing import label_binarize
            classes = np.unique(y_true)
            y_bin   = label_binarize(y_true, classes=classes)
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                roc_auc = auc(fpr, tpr)
                lbl = label_encoder.classes_[cls] if label_encoder else str(cls)
                ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                        label=f"{lbl} (AUC={roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1.5, label="Random")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend(fontsize=9)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"ROC chart error: {e}")
        return None


# ─────────────────────────────────────────────
#  REGRESSION CHARTS
# ─────────────────────────────────────────────

def actual_vs_predicted_chart(y_true, y_pred):
    """Scatter: actual vs predicted with identity line."""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    fig, ax = _fig(5.5, 5)

    ax.scatter(y_true, y_pred, alpha=0.5, color=ACCENT,
               edgecolors="#0b0d12", linewidth=0.3, s=35)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "--", color=ACCENT3, lw=2, label="Perfect fit")

    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted  (R² = {r2:.4f})", pad=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def residuals_chart(y_true, y_pred):
    """Residual plot + residual histogram."""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    with plt.rc_context(DARK):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#0b0d12")

    # Scatter residuals
    axes[0].scatter(y_pred, residuals, alpha=0.5, color=ACCENT2,
                    edgecolors="#0b0d12", linewidth=0.3, s=30)
    axes[0].axhline(0, color=ACCENT3, linewidth=2, linestyle="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Histogram of residuals
    axes[1].hist(residuals, bins=40, color=ACCENT, edgecolor="#0b0d12", alpha=0.85)
    axes[1].axvline(0, color=ACCENT3, linewidth=2, linestyle="--")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=2)
    return fig


# ─────────────────────────────────────────────
#  MODEL COMPARISON CHART
# ─────────────────────────────────────────────

def model_comparison_chart(results: list, metric: str):
    """Horizontal bar chart comparing all models on a metric."""
    names  = [r["model"] for r in results]
    scores = [r.get(metric, 0) for r in results]
    colors = [ACCENT if i == 0 else "#1e2234" for i in range(len(results))]

    fig, ax = _fig(9, max(4, len(results) * 0.52))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1],
                   edgecolor="#0b0d12", height=0.6)

    for bar, val in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + max(scores) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, color="#94a3b8")

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_',' ').title()}", pad=12)
    ax.set_xlim(0, max(scores) * 1.18)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  EDA CHARTS
# ─────────────────────────────────────────────

def distribution_grid(df: pd.DataFrame, cols: list, max_cols: int = 4):
    """Grid of histograms for numeric columns."""
    cols = [c for c in cols if c in df.columns][:12]
    n = len(cols)
    if n == 0:
        return None
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    with plt.rc_context(DARK):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.2))
    fig.patch.set_facecolor("#0b0d12")
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=35, color=PALETTE[i % len(PALETTE)],
                     edgecolor="#0b0d12", linewidth=0.4, alpha=0.9)
        axes[i].set_title(col, fontsize=11, pad=6)
        axes[i].grid(axis="y", alpha=0.3)
        axes[i].spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", y=1.01, fontsize=13, color="#e2e8f0")
    fig.tight_layout(pad=1.8)
    return fig


def correlation_heatmap(df: pd.DataFrame):
    """Lower-triangle correlation heatmap."""
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    n = corr.shape[0]
    size = min(14, n + 2)

    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(size, size - 1))
    fig.patch.set_facecolor("#0b0d12")
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": max(7, 11 - n // 3)},
        linewidths=0.5, linecolor="#0b0d12",
        ax=ax, square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Feature Correlation Matrix", pad=14)
    fig.tight_layout()
    return fig


def missing_values_chart(df: pd.DataFrame):
    """Horizontal bar chart of null percentages."""
    null_pct = (df.isnull().mean() * 100).round(2)
    null_pct = null_pct[null_pct > 0].sort_values(ascending=False)
    if null_pct.empty:
        return None

    fig, ax = _fig(8, max(3, len(null_pct) * 0.48))
    bars = ax.barh(null_pct.index[::-1], null_pct.values[::-1],
                   color=ACCENT3, edgecolor="#0b0d12", height=0.6)
    for bar, val in zip(bars, null_pct.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color="#94a3b8")

    ax.set_xlabel("Missing %"); ax.set_title("Missing Values by Column", pad=12)
    ax.set_xlim(0, null_pct.max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def target_distribution_chart(series: pd.Series, task: str):
    """Bar chart (classification) or histogram (regression) for target."""
    fig, ax = _fig(6, 3.5)
    if task == "classification":
        vc = series.value_counts()
        ax.bar(vc.index.astype(str), vc.values, color=ACCENT,
               edgecolor="#0b0d12", width=0.6)
        ax.set_xlabel("Class"); ax.set_ylabel("Count")
        ax.set_title("Target Class Distribution")
        for i, v in enumerate(vc.values):
            ax.text(i, v + vc.max() * 0.01, str(v),
                    ha="center", fontsize=10, color="#94a3b8")
    else:
        ax.hist(series.dropna(), bins=40, color=ACCENT,
                edgecolor="#0b0d12", alpha=0.9)
        ax.set_xlabel(series.name); ax.set_ylabel("Count")
        ax.set_title("Target Distribution")

    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
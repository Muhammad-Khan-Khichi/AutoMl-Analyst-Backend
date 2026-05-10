"""Reusable chart helpers (matplotlib + seaborn)."""
import matplotlib.pyplot as plt, seaborn as sns, numpy as np

MPL_DARK = {
    "figure.facecolor":"#0b0d12","axes.facecolor":"#0f1117",
    "axes.edgecolor":"#1e2130","axes.labelcolor":"#94a3b8",
    "axes.titlecolor":"#e2e8f0","xtick.color":"#475569",
    "ytick.color":"#475569","text.color":"#94a3b8",
    "grid.color":"#1e2130","grid.linestyle":"--","grid.alpha":0.6,
}

def styled_fig(w=10, h=5):
    with plt.rc_context(MPL_DARK): return plt.subplots(figsize=(w, h))

def distribution_grid(df, cols, max_cols=4):
    n = len(cols); ncols = min(max_cols, n); nrows = (n+ncols-1)//ncols
    with plt.rc_context(MPL_DARK):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3))
    axes = np.array(axes).flatten()
    pal = ["#6366f1","#818cf8","#38bdf8","#34d399","#fb923c","#f472b6","#a78bfa","#fbbf24"]
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=30, color=pal[i%len(pal)], edgecolor="#0b0d12")
        axes[i].set_title(col, fontsize=11); axes[i].grid(axis="y", alpha=0.3)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.tight_layout(pad=1.5); return fig

def correlation_heatmap(df):
    corr = df.select_dtypes("number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    with plt.rc_context(MPL_DARK):
        fig, ax = styled_fig(min(12,len(corr)+2), min(10,len(corr)+1))
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(240,10,as_cmap=True),
                center=0, annot=True, fmt=".2f", annot_kws={"size":9},
                linewidths=0.5, linecolor="#0b0d12", ax=ax, square=True)
    ax.set_title("Correlation matrix", fontsize=13, pad=14); fig.tight_layout(); return fig

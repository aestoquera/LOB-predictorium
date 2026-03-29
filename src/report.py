import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    n = a.size
    if n == 0:
        return a.astype(float)

    sorter = np.argsort(a, kind="mergesort")
    a_sorted = a[sorter]
    ranks = np.empty(n, dtype=float)

    # Detectamos bloques de empates
    starts = np.r_[0, np.flatnonzero(a_sorted[1:] != a_sorted[:-1]) + 1]
    ends = np.r_[starts[1:], n]

    for s, e in zip(starts, ends):
        # rank medio en escala 1..n
        r = 0.5 * (s + e - 1) + 1.0
        ranks[sorter[s:e]] = r

    return ranks


def _corr_1d(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> tuple[float, int]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = a.size

    if n < 3:
        return np.nan, n

    if method == "spearman":
        a = _rankdata_average(a)
        b = _rankdata_average(b)

    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return np.nan, n

    return float(np.sum(a * b) / denom), n


def _get_segment_df(panel, segment: str) -> pl.DataFrame:
    if segment == "all":
        return panel.df
    if segment == "warmup":
        return panel.warmup_df()
    if segment == "prediction":
        return panel.prediction_df()
    raise ValueError("segment debe ser 'all', 'warmup' o 'prediction'")


def _compute_feature_target_report(panel, segment: str = "warmup") -> pl.DataFrame:
    df = _get_segment_df(panel, segment)
    rows = []

    for feat in panel.feature_cols:
        x = df.get_column(feat).to_numpy()
        for target in panel.target_cols:
            y = df.get_column(target).to_numpy()

            pearson, n1 = _corr_1d(x, y, method="pearson")
            spearman, n2 = _corr_1d(x, y, method="spearman")

            rows.append(
                {
                    "feature": feat,
                    "target": target,
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson": np.abs(pearson) if np.isfinite(pearson) else np.nan,
                    "abs_spearman": np.abs(spearman) if np.isfinite(spearman) else np.nan,
                    "n": min(n1, n2),
                }
            )

    report = pl.DataFrame(rows).sort(["target", "abs_spearman"], descending=[False, True])
    return report


def _lagged_correlation(panel, feature: str, target: str, segment: str = "warmup", max_lag: int = 20):
    lags = np.arange(-max_lag, max_lag + 1)
    vals = []

    for lag in lags:
        xs = []
        ys = []

        for seq_ix in panel.sequence_ids:
            seq = _get_segment_df(panel, segment).filter(pl.col(panel.seq_col) == seq_ix).sort(panel.step_col)
            if seq.height == 0:
                continue

            x = seq.get_column(feature).to_numpy()
            y = seq.get_column(target).to_numpy()

            if lag > 0:
                if len(x) <= lag:
                    continue
                xs.append(x[:-lag])
                ys.append(y[lag:])
            elif lag < 0:
                k = -lag
                if len(x) <= k:
                    continue
                xs.append(x[k:])
                ys.append(y[:-k])
            else:
                xs.append(x)
                ys.append(y)

        if len(xs) == 0:
            vals.append(np.nan)
            continue

        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        corr, _ = _corr_1d(x_all, y_all, method="spearman")
        vals.append(corr)

    return lags, np.asarray(vals, dtype=float)


def analyze_panel_feature_target_relations(
    panel,
    segment: str = "warmup", # Puede ser warmup, all o prediction
    top_k: int = 8,
    max_lag: int = 20,
):
    """
    Devuelve un report cuantitativo y dibuja visualizaciones para estudiar
    la relación entre features y targets.
    """
    report = _compute_feature_target_report(panel, segment=segment)

    # --- 1) Heatmap de correlación (Spearman) ---
    heat = report.select(["feature", "target", "spearman"]).to_pandas().pivot(
        index="feature", columns="target", values="spearman"
    )

    plt.figure(figsize=(8, max(4, 0.35 * len(heat.index))))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Spearman"},
    )
    plt.title(f"Correlación Spearman feature-target ({segment})")
    plt.tight_layout()
    plt.show()

    # --- 2) Barras de las top-k variables por target ---
    for target in panel.target_cols:
        sub = (
            report
            .filter(pl.col("target") == target)
            .sort("abs_spearman", descending=True)
            .head(top_k)
            .to_pandas()
        )

        plt.figure(figsize=(10, max(4, 0.4 * len(sub))))
        sns.barplot(data=sub, x="spearman", y="feature")
        plt.axvline(0, color="black", linewidth=1)
        plt.title(f"Top {top_k} features asociadas a {target} ({segment})")
        plt.xlabel("Spearman")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    # --- 3) Correlación con desfase temporal (para ver si anticipan o se retrasan) ---
    for target in panel.target_cols:
        top_features = (
            report
            .filter(pl.col("target") == target)
            .sort("abs_spearman", descending=True)
            .head(min(5, top_k))
            .get_column("feature")
            .to_list()
        )

        plt.figure(figsize=(12, 5))
        for feat in top_features:
            lags, corr_vals = _lagged_correlation(
                panel, feature=feat, target=target, segment=segment, max_lag=max_lag
            )
            plt.plot(lags, corr_vals, marker="o", linewidth=1.5, label=feat)

        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.axhline(0, color="gray", linewidth=1)
        plt.title(f"Correlación con desfase temporal vs {target} ({segment})")
        plt.xlabel("Lag (positivo = feature antes que target)")
        plt.ylabel("Spearman")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return report
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


def _panel_matrix(panel, col: str) -> np.ndarray:
    """Devuelve una matriz [n_seq, T] para una columna concreta."""
    rows = []
    for seq_ix in panel.sequence_ids:
        seq = panel.sequence_df(seq_ix, cols=[col])
        rows.append(seq.get_column(col).to_numpy().astype(float))
    return np.vstack(rows)


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    """Z-score por secuencia para comparar forma, no escala."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def _acf_one(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelación normalizada para una serie 1D."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.full(max_lag + 1, np.nan)

    c = np.correlate(x, x, mode="full")
    mid = len(x) - 1
    acf = c[mid:mid + max_lag + 1] / denom
    return acf


def _mean_acf(X: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelación media entre secuencias."""
    acfs = np.vstack([_acf_one(row, max_lag) for row in X])
    return np.nanmean(acfs, axis=0)


def _mean_spectrum(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Espectro medio de potencia.
    Devuelve:
      freqs: frecuencias en ciclos/step
      power: potencia media
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(Xc, axis=1)) ** 2
    power = np.nanmean(spec, axis=0)
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0)
    return freqs, power


def _top_peaks(values: np.ndarray, x: np.ndarray, k: int = 5, min_sep: int = 3):
    """
    Selecciona picos altos evitando vecinos demasiado cercanos.
    """
    values = np.asarray(values)
    order = np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1]
    chosen = []

    for idx in order:
        if not np.isfinite(values[idx]):
            continue
        if any(abs(idx - c) < min_sep for c in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= k:
            break

    return [(float(x[i]), float(values[i])) for i in chosen]


def _fold_by_period(X: np.ndarray, period: int):
    """
    Pliega [n_seq, T] en:
      mean_cycle: [n_cycles, period]
      phase_mean: [period]
      phase_std: [period]
    """
    T = X.shape[1]
    n_cycles = T // period
    if n_cycles < 2:
        return None

    Xt = X[:, :n_cycles * period]
    folded = Xt.reshape(X.shape[0], n_cycles, period)

    mean_cycle = np.nanmean(folded, axis=0)
    phase_mean = np.nanmean(folded, axis=(0, 1))
    phase_std = np.nanstd(folded, axis=(0, 1))
    return mean_cycle, phase_mean, phase_std


def analyze_panel_seasonality(
    panel,
    columns=None,
    max_lag: int = 250,
    top_k: int = 5,
    use_zscore: bool = True,
):
    """
    Analiza periodicidad interna del panel.

    columns:
      - None -> panel.target_cols
      - o lista de columnas, por ejemplo panel.feature_cols
    """
    if columns is None:
        columns = list(panel.target_cols)

    records = []

    for col in columns:
        X = _panel_matrix(panel, col)
        if use_zscore:
            X = _zscore_rows(X)

        T = X.shape[1]

        # 1) Autocorrelación media
        acf = _mean_acf(X, max_lag=max_lag)
        lags = np.arange(max_lag + 1)

        # Suavizado ligero para que los picos sean más legibles
        kernel = np.ones(5) / 5
        acf_smooth = np.convolve(acf, kernel, mode="same")

        acf_candidates = _top_peaks(acf_smooth[1:], lags[1:], k=top_k, min_sep=4)
        for period, score in acf_candidates:
            records.append({
                "variable": col,
                "method": "acf",
                "period_steps": int(round(period)),
                "score": score,
                "cycles_in_1000": T / period,
            })

        # 2) Espectro medio
        freqs, power = _mean_spectrum(X)
        if len(power) > 0:
            power_plot = power.copy()
            power_plot[0] = np.nan  # quitamos DC

            # Convertimos frecuencia a periodo en steps
            spec_candidates = _top_peaks(power_plot[1:], freqs[1:], k=top_k, min_sep=4)
            for freq, score in spec_candidates:
                if freq > 0:
                    period = 1.0 / freq
                    records.append({
                        "variable": col,
                        "method": "spectrum",
                        "period_steps": int(round(period)),
                        "score": score,
                        "cycles_in_1000": T / period,
                    })

        # 3) Visualizaciones
        best_period = None
        if len(acf_candidates) > 0:
            best_period = int(round(acf_candidates[0][0]))

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(lags, acf, label="ACF media")
        ax1.plot(lags, acf_smooth, label="ACF suavizada", linewidth=2)
        ax1.axhline(0, color="black", linewidth=1)
        for p, s in acf_candidates[:top_k]:
            ax1.axvline(p, linestyle="--", alpha=0.4)
        ax1.set_title(f"{col} — autocorrelación media")
        ax1.set_xlabel("Lag (steps)")
        ax1.set_ylabel("Autocorr.")
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        valid = freqs > 0
        periods = np.where(valid, 1.0 / freqs, np.nan)
        ax2.plot(periods[valid], power[valid])
        ax2.set_xlim(2, min(T // 2, 500))
        ax2.invert_xaxis()
        ax2.set_title(f"{col} — espectro medio de potencia")
        ax2.set_xlabel("Periodo aproximado (steps)")
        ax2.set_ylabel("Potencia")

        ax3 = fig.add_subplot(gs[1, :])
        if best_period is not None and best_period >= 2:
            folded = _fold_by_period(X, best_period)
            if folded is not None:
                mean_cycle, phase_mean, phase_std = folded
                sns.heatmap(
                    mean_cycle,
                    ax=ax3,
                    cmap="viridis",
                    cbar_kws={"label": "valor normalizado"},
                )
                ax3.set_title(
                    f"{col} — patrón plegado por periodo candidato = {best_period} steps"
                )
                ax3.set_xlabel("Fase dentro del periodo")
                ax3.set_ylabel("Ciclo")

                # Línea de media de fase en un eje secundario para leer forma global
                ax4 = ax3.twinx()
                ax4.plot(np.arange(best_period) + 0.5, phase_mean, color="white", linewidth=2)
                ax4.fill_between(
                    np.arange(best_period) + 0.5,
                    phase_mean - phase_std,
                    phase_mean + phase_std,
                    color="white",
                    alpha=0.15,
                )
                ax4.set_ylabel("Media por fase")
                ax4.set_xlim(0, best_period)
            else:
                ax3.text(
                    0.5, 0.5,
                    "No hay suficientes ciclos para plegar con este periodo.",
                    ha="center", va="center"
                )
                ax3.axis("off")
        else:
            ax3.text(
                0.5, 0.5,
                "No se detectó un periodo candidato claro.",
                ha="center", va="center"
            )
            ax3.axis("off")

        plt.tight_layout()
        plt.show()

    report = pl.DataFrame(records).sort(
        ["variable", "method", "score"],
        descending=[False, False, True]
    )
    return report
import numpy as np
import polars as pl
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def _clean_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    return y


def _infer_d(series: np.ndarray, max_d: int = 2) -> int:
    """
    Estima d usando ADF sobre la serie y sus diferencias.
    """
    x = _clean_series(series)

    if len(x) < 10:
        return 0

    for d in range(max_d + 1):
        try:
            pval = adfuller(x, autolag="AIC")[1]
            if pval < 0.05:
                return d
        except Exception:
            # Si ADF falla, nos quedamos con el d actual
            return d

        x = np.diff(x)

        if len(x) < 10:
            return d + 1

    return max_d


def _infer_seasonality(
    series: np.ndarray, max_lag: int = 100, min_strength: float = 0.2
) -> int:
    """
    Estima un periodo estacional candidato con autocorrelación.
    Devuelve 0 si no hay evidencia clara.
    """
    x = _clean_series(series)
    if len(x) < 20:
        return 0

    x = x - np.mean(x)
    acf = np.correlate(x, x, mode="full")[len(x) - 1 :]
    acf[0] = 0.0

    limit = min(max_lag, len(acf) - 1)
    if limit < 2:
        return 0

    peak = np.argmax(acf[1 : limit + 1]) + 1
    if acf[peak] >= min_strength:
        return int(peak)

    return 0


def _grid_search_sarimax(
    y: np.ndarray,
    d: int,
    s: int,
    p_max: int = 2,
    q_max: int = 2,
    P_max: int = 1,
    Q_max: int = 1,
):
    """
    Grid search pequeño y robusto para SARIMAX.
    Devuelve el mejor modelo o fallback.
    """
    y = _clean_series(y)

    # Si la serie es demasiado corta, no forzamos modelos complejos
    if len(y) < 20:
        return (0, d, 0), (0, 0, 0, 0), np.inf, 0, "serie_demasiado_corta"

    best_aic = np.inf
    best_order = None
    best_seasonal = None
    n_fits = 0
    n_fail = 0

    seasonal_enabled = s is not None and s > 1

    # Rangos inclusivos
    for p in range(0, p_max + 1):
        for q in range(0, q_max + 1):
            for P in range(0, P_max + 1):
                for Q in range(0, Q_max + 1):
                    # Si no hay seasonality, solo probamos la parte no estacional
                    seasonal_order = (P, 0, Q, s) if seasonal_enabled else (0, 0, 0, 0)

                    # Evita modelos absurdamente grandes para series cortas
                    if p + q + P + Q == 0:
                        # igual probamos ARIMA puro (0,d,0)
                        pass

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            model = SARIMAX(
                                y,
                                order=(p, d, q),
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                                simple_differencing=False,
                            )
                            res = model.fit(
                                disp=False,
                                method="lbfgs",
                                maxiter=200,
                            )

                        n_fits += 1

                        if np.isfinite(res.aic) and res.aic < best_aic:
                            best_aic = float(res.aic)
                            best_order = (p, d, q)
                            best_seasonal = seasonal_order

                    except Exception:
                        n_fail += 1
                        continue

    if best_order is None:
        # Fallback seguro: modelo mínimo
        return (
            (0, d, 0),
            (0, 0, 0, 0),
            np.inf,
            n_fits,
            f"sin_fit_valido_{n_fail}_fallos",
        )

    return best_order, best_seasonal, best_aic, n_fits, f"ok_{n_fail}_fallos"


def sarimax_panel_analysis(
    panel,
    target: str = "t0",
    max_d: int = 2,
    max_season_lag: int = 100,
    p_max: int = 2,
    q_max: int = 2,
    P_max: int = 1,
    Q_max: int = 1,
):
    """
    Devuelve un Polars DataFrame con parámetros SARIMAX por secuencia.
    Nunca devuelve vacío salvo que el panel no tenga secuencias.
    """
    results = []

    for seq_ix in panel.sequence_ids:
        seq = panel.sequence_df(seq_ix, cols=[target])
        y = seq.get_column(target).to_numpy()
        y = _clean_series(y)

        if len(y) == 0:
            results.append(
                {
                    "seq_ix": seq_ix,
                    "target": target,
                    "p": None,
                    "d": None,
                    "q": None,
                    "P": None,
                    "D": None,
                    "Q": None,
                    "s": None,
                    "aic": None,
                    "n_points": 0,
                    "n_fits": 0,
                    "status": "serie_vacia",
                }
            )
            continue

        d = _infer_d(y, max_d=max_d)
        s = _infer_seasonality(y, max_lag=max_season_lag)
        D = 1 if s > 1 else 0

        order, seasonal, aic, n_fits, status = _grid_search_sarimax(
            y,
            d=d,
            s=s,
            p_max=p_max,
            q_max=q_max,
            P_max=P_max,
            Q_max=Q_max,
        )

        results.append(
            {
                "seq_ix": seq_ix,
                "target": target,
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "P": seasonal[0],
                "D": seasonal[1],
                "Q": seasonal[2],
                "s": seasonal[3],
                "aic": aic,
                "n_points": int(len(y)),
                "n_fits": int(n_fits),
                "status": status,
            }
        )

    return pl.DataFrame(results)


import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List


class SarimaxAnalyzer:
    def __init__(
        self,
        df1: pl.DataFrame,
        df2: pl.DataFrame,
        labels: List[str] = ["Target 0", "Target 1"],
    ):
        self.labels = labels
        # Añadimos etiqueta de origen y combinamos
        self.df1 = df1.with_columns(pl.lit(labels[0]).alias("source"))
        self.df2 = df2.with_columns(pl.lit(labels[1]).alias("source"))
        self.full_df = pl.concat([self.df1, self.df2])

        # Parámetros a analizar
        self.params = ["p", "d", "q", "P", "D", "Q", "s"]

    def quantitative_analysis(self):
        """Calcula estadísticas descriptivas y similitud de parámetros."""
        print("--- ANÁLISIS CUANTITATIVO ---")

        # 1. Similitud de parámetros (Exact Match)
        # Unimos por seq_ix para ver si coinciden en la misma secuencia
        overlap = self.df1.join(self.df2, on="seq_ix", suffix="_alt")

        # Calculamos coincidencia exacta de la tupla (p,d,q,P,D,Q,s)
        same_model = overlap.filter(
            (pl.col("p") == pl.col("p_alt"))
            & (pl.col("d") == pl.col("d_alt"))
            & (pl.col("q") == pl.col("q_alt"))
            & (pl.col("P") == pl.col("P_alt"))
            & (pl.col("Q") == pl.col("Q_alt"))
            & (pl.col("s") == pl.col("s_alt"))
        ).height

        similarity_pct = (same_model / self.df1.height) * 100

        # 2. Resumen de AIC por fuente
        aic_summary = self.full_df.group_by("source").agg(
            [
                pl.col("aic").mean().alias("mean_aic"),
                pl.col("aic").std().alias("std_aic"),
                pl.col("aic").median().alias("median_aic"),
            ]
        )

        print(
            f"Coincidencia exacta de parámetros entre DataFrames: {similarity_pct:.2f}%"
        )
        print("\nResumen de AIC:")
        print(aic_summary)

        return {"similarity_score": similarity_pct, "aic_summary": aic_summary}

    def plot_boxplots(self):
        """Genera boxplots comparativos para los órdenes y el AIC."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        cols_to_plot = self.params + ["aic"]

        for i, col in enumerate(cols_to_plot):
            # Convertimos a pandas solo para el plotting (compatibilidad de Seaborn)
            sns.boxplot(
                data=self.full_df.select(["source", col]).to_pandas(),
                x="source",
                y=col,
                ax=axes[i],
                palette="Set2",
                hue="source",
            )
            axes[i].set_title(f"Distribución de {col.upper()}")
            axes[i].set_xlabel("")

        plt.tight_layout()
        plt.show()

    def qualitative_summary(self):
        """Genera una interpretación basada en las modas de los parámetros."""
        print("\n--- ANÁLISIS CUALITATIVO ---")
        for label in self.labels:
            subset = self.full_df.filter(pl.col("source") == label)

            # Calculamos la moda de los parámetros principales
            p_mode = subset["p"].mode()[0]
            d_mode = subset["d"].mode()[0]
            q_mode = subset["q"].mode()[0]
            s_mode = subset["s"].mode()[0]

            print(f"\nInterpretación para [{label}]:")
            print(f"- La estructura predominante es ({p_mode}, {d_mode}, {q_mode}).")
            if s_mode > 1:
                print(f"- Se detecta una estacionalidad consistente en s={s_mode}.")
            else:
                print(
                    "- La mayoría de las series no presentan componentes estacionales claros."
                )

            # Evaluación de estabilidad
            status_counts = subset["status"].value_counts()
            success_rate = (
                subset.filter(pl.col("status").str.contains("ok")).height
                / subset.height
            ) * 100
            print(f"- Tasa de éxito en el ajuste: {success_rate:.2f}%")

            self.plot_boxplots()
            self.plot_parameter_distributions()
            self.plot_triplet_frequencies()

    def plot_parameter_distributions(self):
        """
        Genera histogramas de frecuencias para cada parámetro individual (p, d, q, P, D, Q, s).
        Permite ver qué órdenes son los más 'populares' en el panel.
        """
        fig, axes = plt.subplots(
            len(self.labels),
            len(self.params),
            figsize=(20, 5 * len(self.labels)),
            squeeze=False,
        )

        for i, label in enumerate(self.labels):
            subset = self.full_df.filter(pl.col("source") == label)
            for j, param in enumerate(self.params):
                data = subset[param].to_pandas()
                sns.countplot(x=data, ax=axes[i, j], palette="viridis", hue=data)
                axes[i, j].set_title(f"{label}: Frecuencia {param}")
                axes[i, j].set_xlabel("Valor")
                axes[i, j].set_ylabel("Conteo")

        plt.tight_layout()
        plt.show()

    def plot_triplet_frequencies(self, top_n: int = 5):
        """
        Analiza las tripletas (p, d, q) y (P, D, Q) más frecuentes.
        Esto identifica la 'arquitectura' de modelo más común.
        """
        for label in self.labels:
            subset = self.full_df.filter(pl.col("source") == label)

            # Crear representación de string para las tripletas
            triplet_df = subset.with_columns(
                [
                    pl.format(
                        "({}, {}, {})", pl.col("p"), pl.col("d"), pl.col("q")
                    ).alias("pdq_triplet"),
                    pl.format(
                        "({}, {}, {})", pl.col("P"), pl.col("D"), pl.col("Q")
                    ).alias("PDQ_triplet"),
                ]
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(f"Top {top_n} Tripletas más frecuentes - {label}", fontsize=16)

            # Graficar PDQ regular
            pdq_counts = (
                triplet_df["pdq_triplet"]
                .value_counts()
                .sort("count", descending=True)
                .head(top_n)
            )
            sns.barplot(
                data=pdq_counts.to_pandas(),
                x="pdq_triplet",
                y="count",
                ax=ax1,
                palette="magma",
                hue="pdq_triplet",
            )
            ax1.set_title("Frecuencia (p, d, q)")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

            # Graficar PDQ estacional
            PDQ_counts = (
                triplet_df["PDQ_triplet"]
                .value_counts()
                .sort("count", descending=True)
                .head(top_n)
            )
            sns.barplot(
                data=PDQ_counts.to_pandas(),
                x="PDQ_triplet",
                y="count",
                ax=ax2,
                palette="rocket",
                hue="PDQ_triplet",
            )
            ax2.set_title("Frecuencia (P, D, Q)")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

            plt.tight_layout()
            plt.show()

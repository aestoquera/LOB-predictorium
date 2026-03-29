from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import PredictionModel, DataPoint, weighted_pearson_correlation
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

import copy
from dataclasses import dataclass
from typing import Sequence


class AbstractPanelModel(PredictionModel, ABC):
    """
    Base abstracta para modelos que trabajan con un panel de entrenamiento
    y un panel de test/eval.

    Responsabilidades:
      - guardar train_panel y test_panel
      - ofrecer iteración en formato DataPoint
      - evaluar predicciones con weighted_pearson_correlation

    Subclases deben implementar:
      - fit()
      - predict(data_point)
    """

    def __init__(self, train_panel, test_panel):
        self.train_panel = train_panel
        self.test_panel = test_panel

        self.seq_col = train_panel.seq_col
        self.step_col = train_panel.step_col
        self.warmup_col = train_panel.warmup_col
        self.feature_cols = tuple(train_panel.feature_cols)
        self.target_cols = tuple(train_panel.target_cols)

        self._is_fitted = False

    @abstractmethod
    def fit(self):
        """
        Entrena el modelo usando self.train_panel.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        """
        Devuelve:
          - None si data_point.need_prediction es False
          - np.ndarray de shape (2,) si data_point.need_prediction es True
        """
        raise NotImplementedError

    def panel_to_datapoints(self, panel) -> Iterator[tuple[DataPoint, np.ndarray]]:
        """
        Convierte un panel en una secuencia de (DataPoint, labels_true).
        El orden queda fijado por (seq_ix, step_in_seq).
        """
        df = panel.df.sort([panel.seq_col, panel.step_col])

        cols = [
            panel.seq_col,
            panel.step_col,
            panel.warmup_col,
            *panel.feature_cols,
            *panel.target_cols,
        ]

        for row in df.select(cols).iter_rows(named=False):
            seq_ix = int(row[0])
            step_in_seq = int(row[1])
            need_prediction = bool(row[2])

            state = np.asarray(row[3 : 3 + len(panel.feature_cols)], dtype=float)
            labels = np.asarray(row[3 + len(panel.feature_cols) :], dtype=float)

            data_point = DataPoint(
                seq_ix=seq_ix,
                step_in_seq=step_in_seq,
                need_prediction=need_prediction,
                state=state,
            )

            yield data_point, labels

    def evaluate(self, panel=None) -> dict:
        """
        Evalúa el modelo sobre el panel de test por defecto.

        Devuelve un dict con:
          - score por target
          - weighted_pearson promedio
        """
        if panel is None:
            panel = self.test_panel

        predictions = []
        targets = []

        for data_point, labels in self.panel_to_datapoints(panel):
            pred = self.predict(data_point)

            if not data_point.need_prediction:
                if pred is not None:
                    raise ValueError(
                        f"Prediction is not needed for {data_point}, but predict() returned a value."
                    )
                continue

            if pred is None:
                raise ValueError(
                    f"Prediction is required for {data_point}, but predict() returned None."
                )

            pred = np.asarray(pred, dtype=float)

            if pred.shape != (len(self.target_cols),):
                raise ValueError(
                    f"Prediction has wrong shape: {pred.shape}, expected {(len(self.target_cols),)}"
                )

            predictions.append(pred)
            targets.append(labels)

        predictions = np.asarray(predictions, dtype=float)
        targets = np.asarray(targets, dtype=float)

        scores = {}
        for ix, target_name in enumerate(self.target_cols):
            scores[target_name] = weighted_pearson_correlation(
                targets[:, ix],
                predictions[:, ix],
            )

        scores["weighted_pearson"] = float(np.mean(list(scores.values())))
        return scores

    def qualitative_evaluation(self, seq_ix: int = None, panel=None):
        """
        Evaluación cualitativa de una secuencia del panel de test.

        Muestra:
        - Pred vs real (t0, t1)
        - Error temporal
        - Distribución de errores
        - Scatter real vs pred
        """

        if panel is None:
            panel = self.test_panel

        if seq_ix is None:
            seq_ix = panel.sequence_ids[0]

        # =========================
        # GENERAR PREDICCIONES
        # =========================
        t_vals = []
        real_t0, real_t1 = [], []
        pred_t0, pred_t1 = [], []

        for data_point, labels in self.panel_to_datapoints(panel):
            if data_point.seq_ix != seq_ix:
                continue

            pred = self.predict(data_point)

            if not data_point.need_prediction:
                continue

            t_vals.append(data_point.step_in_seq)

            real_t0.append(labels[0])
            real_t1.append(labels[1])

            pred_t0.append(pred[0])
            pred_t1.append(pred[1])

        t_vals = np.array(t_vals)

        real_t0 = np.array(real_t0)
        real_t1 = np.array(real_t1)

        pred_t0 = np.array(pred_t0)
        pred_t1 = np.array(pred_t1)

        # =========================
        # 1. PRED VS REAL
        # =========================
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(t_vals, real_t0, label="real t0")
        axes[0].plot(t_vals, pred_t0, label="pred t0")
        axes[0].set_title(f"t0 — Predicción vs Real (seq {seq_ix})")
        axes[0].legend()

        axes[1].plot(t_vals, real_t1, label="real t1")
        axes[1].plot(t_vals, pred_t1, label="pred t1")
        axes[1].set_title(f"t1 — Predicción vs Real (seq {seq_ix})")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        # =========================
        # 2. ERROR EN EL TIEMPO
        # =========================
        err_t0 = real_t0 - pred_t0
        err_t1 = real_t1 - pred_t1

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        axes[0].plot(t_vals, err_t0)
        axes[0].set_title("Error t0 en el tiempo")

        axes[1].plot(t_vals, err_t1)
        axes[1].set_title("Error t1 en el tiempo")

        plt.tight_layout()
        plt.show()

        # =========================
        # 3. DISTRIBUCIÓN DE ERRORES
        # =========================
        plt.figure(figsize=(10, 4))

        sns.histplot(err_t0, kde=True, label="t0", bins=50)
        sns.histplot(err_t1, kde=True, label="t1", bins=50)

        plt.title("Distribución de errores")
        plt.legend()
        plt.show()

        # =========================
        # 4. SCATTER REAL VS PRED
        # =========================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(real_t0, pred_t0, alpha=0.5)
        axes[0].set_title("t0: real vs pred")
        axes[0].set_xlabel("real")
        axes[0].set_ylabel("pred")

        axes[1].scatter(real_t1, pred_t1, alpha=0.5)
        axes[1].set_title("t1: real vs pred")
        axes[1].set_xlabel("real")
        axes[1].set_ylabel("pred")

        plt.tight_layout()
        plt.show()


class MySarimaModel(AbstractPanelModel):
    """
    ARIMAX global con dos modelos independientes:
      - uno para t0
      - otro para t1

    Usa las columnas no target como variables exógenas.
    Entrena SOLO con train_panel y predice step-by-step sobre test_panel.
    """

    def __init__(
        self,
        train_panel,
        test_panel,
        order_t0=(1, 0, 1),
        order_t1=(1, 0, 1),
        add_const: bool = True,
    ):
        super().__init__(train_panel, test_panel)

        self.order_t0 = tuple(order_t0)
        self.order_t1 = tuple(order_t1)
        self.add_const = add_const

        self.model_t0 = None
        self.model_t1 = None

        self.result_t0 = None
        self.result_t1 = None

        self._n_features = len(self.feature_cols)

    def _panel_xy(self, panel):
        """
        Extrae X e y de un panel, ordenado por secuencia y paso.
        """
        df = panel.df.sort([panel.seq_col, panel.step_col])

        cols = [*self.feature_cols, *self.target_cols]
        mat = df.select(cols).to_numpy().astype(float)

        X = mat[:, : self._n_features]
        y0 = mat[:, self._n_features + 0]
        y1 = mat[:, self._n_features + 1]

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y0) & np.isfinite(y1)
        return X[mask], y0[mask], y1[mask]

    def _prepare_exog(self, X: np.ndarray) -> np.ndarray:
        """
        Prepara exógenas para SARIMAX.
        """
        X = np.asarray(X, dtype=float).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if self.add_const:
            X = self._add_constant(X)

        return X

    def fit(self):
        """
        Entrena un SARIMAX global para cada target usando train_panel.
        """
        X_train, y0_train, y1_train = self._panel_xy(self.train_panel)

        if len(X_train) < 20:
            raise ValueError(
                "No hay suficientes observaciones limpias para entrenar SARIMAX."
            )

        exog_train = X_train
        if self.add_const:
            exog_train = self._add_constant(exog_train)
        # Modelo para t0
        self.model_t0 = SARIMAX(
            y0_train,
            exog=exog_train,
            order=self.order_t0,
            seasonal_order=(0, 0, 0, 0),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result_t0 = self.model_t0.fit(disp=False)

        # Modelo para t1
        self.model_t1 = SARIMAX(
            y1_train,
            exog=exog_train,
            order=self.order_t1,
            seasonal_order=(0, 0, 0, 0),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result_t1 = self.model_t1.fit(disp=False)

        self._is_fitted = True
        return self

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        """
        Devuelve None en warmup.
        Devuelve np.array([pred_t0, pred_t1]) cuando need_prediction=True.
        """
        if not data_point.need_prediction:
            return None

        if self.result_t0 is None or self.result_t1 is None:
            raise RuntimeError("El modelo no está entrenado. Llama primero a fit().")

        exog = self._prepare_exog(np.asarray(data_point.state, dtype=float))

        try:
            pred_t0 = float(self.result_t0.forecast(steps=1, exog=exog)[0])
        except Exception:
            pred_t0 = 0.0

        try:
            pred_t1 = float(self.result_t1.forecast(steps=1, exog=exog)[0])
        except Exception:
            pred_t1 = 0.0

        pred = np.array([pred_t0, pred_t1], dtype=float)
        pred = np.clip(pred, -6.0, 6.0)
        return pred

    def _add_constant(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])


class ARIMAEnsemble(AbstractPanelModel):
    """
    Ensemble global de ARIMAX/SARIMAX para paneles de series independientes.

    Idea:
      - Selecciona covariables relevantes de forma global sobre train_panel.
      - Construye un "prototype series" por target agregando las series de train.
      - Entrena múltiples SARIMAX con:
          * órdenes distintos
          * subconjuntos distintos de covariables
          * bootstrap sobre series del panel
      - Combina predicciones por media ponderada por AIC.

    Regularización práctica:
      - selección de features por correlación
      - filtrado de colinealidad
      - bagging / bootstrap
      - estandarización global
    """

    def __init__(
        self,
        train_panel,
        test_panel,
        orders_t0: Sequence[tuple[int, int, int]] = ((1, 0, 1), (2, 0, 1), (2, 0, 2)),
        orders_t1: Sequence[tuple[int, int, int]] = ((1, 0, 1), (2, 0, 1), (2, 0, 2)),
        top_k_features: int = 12,
        min_abs_corr: float = 0.03,
        max_pairwise_corr: float = 0.90,
        max_features_per_member: int = 8,
        n_bootstrap: int = 5,
        random_state: int = 42,
    ):
        super().__init__(train_panel, test_panel)

        self.orders_t0 = list(orders_t0)
        self.orders_t1 = list(orders_t1)

        self.top_k_features = int(top_k_features)
        self.min_abs_corr = float(min_abs_corr)
        self.max_pairwise_corr = float(max_pairwise_corr)
        self.max_features_per_member = int(max_features_per_member)
        self.n_bootstrap = int(n_bootstrap)
        self.rng = np.random.default_rng(random_state)

        self.members = {
            "t0": [],
            "t1": [],
        }

        self.feature_stats = {}  # feat -> (mean, std)
        self.target_stats = {}  # target -> (mean, std)
        self.selected_features = {}  # target -> list[str]

        self._seq_cache = {}
        self._active_seq_ix = None

        self._seq_prediction_cache = {}
        self._feature_idx_cache = {}  # target -> list of indices por miembro

    # -----------------------------
    # Utilidades numéricas
    # -----------------------------
    @staticmethod
    def _safe_std(x: np.ndarray) -> float:
        s = float(np.std(x))
        return s if s > 1e-12 else 1.0

    @staticmethod
    def _rankdata_average(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        n = a.size
        if n == 0:
            return a.astype(float)

        sorter = np.argsort(a, kind="mergesort")
        a_sorted = a[sorter]
        ranks = np.empty(n, dtype=float)

        starts = np.r_[0, np.flatnonzero(a_sorted[1:] != a_sorted[:-1]) + 1]
        ends = np.r_[starts[1:], n]

        for s, e in zip(starts, ends):
            r = 0.5 * (s + e - 1) + 1.0
            ranks[sorter[s:e]] = r

        return ranks

    def _spearman_corr(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        if len(a) < 3:
            return 0.0

        ar = self._rankdata_average(a)
        br = self._rankdata_average(b)

        ar = ar - ar.mean()
        br = br - br.mean()
        denom = np.sqrt(np.sum(ar * ar) * np.sum(br * br))
        if denom <= 0:
            return 0.0
        return float(np.sum(ar * br) / denom)

    def _pairwise_feature_corr(self, feat_a: str, feat_b: str) -> float:
        df = self.train_panel.df.select([feat_a, feat_b]).to_numpy().astype(float)
        return abs(self._spearman_corr(df[:, 0], df[:, 1]))

    def _scale_feature(self, feat: str, x: np.ndarray) -> np.ndarray:
        mu, sd = self.feature_stats[feat]
        return (np.asarray(x, dtype=float) - mu) / sd

    def _scale_target(self, target: str, y: np.ndarray) -> np.ndarray:
        mu, sd = self.target_stats[target]
        return (np.asarray(y, dtype=float) - mu) / sd

    def _unscale_target(self, target: str, y_scaled: np.ndarray) -> np.ndarray:
        mu, sd = self.target_stats[target]
        return np.asarray(y_scaled, dtype=float) * sd + mu

    def _prepare_exog_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Añade constante al inicio. SARIMAX usa exog explícitas.
        """
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        const = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([const, X])

    def _prepare_exog_row(
        self, data_point: DataPoint, feature_subset: Sequence[str]
    ) -> np.ndarray:
        """
        Convierte el estado del DataPoint en fila exógena estandarizada + constante.
        """
        feat_to_idx = {f: i for i, f in enumerate(self.feature_cols)}
        row = np.array(
            [data_point.state[feat_to_idx[f]] for f in feature_subset], dtype=float
        )

        scaled = np.array(
            [self._scale_feature(f, row[i]) for i, f in enumerate(feature_subset)],
            dtype=float,
        )
        return self._prepare_exog_matrix(scaled.reshape(1, -1))

    # -----------------------------
    # Feature selection
    # -----------------------------
    def _select_features_for_target(self, target: str) -> list[str]:
        """
        Ranking por correlación target-feature + filtro de colinealidad.
        """
        df = (
            self.train_panel.df.select([*self.feature_cols, target])
            .to_numpy()
            .astype(float)
        )
        X = df[:, : len(self.feature_cols)]
        y = df[:, len(self.feature_cols)]

        scores = []
        for j, feat in enumerate(self.feature_cols):
            corr = self._spearman_corr(X[:, j], y)
            scores.append((feat, abs(corr), corr))

        scores.sort(key=lambda t: t[1], reverse=True)

        selected = []
        for feat, abs_corr, corr in scores:
            if abs_corr < self.min_abs_corr and len(selected) >= self.top_k_features:
                break

            if selected:
                too_collinear = any(
                    self._pairwise_feature_corr(feat, kept) > self.max_pairwise_corr
                    for kept in selected
                )
                if too_collinear:
                    continue

            selected.append(feat)
            if len(selected) >= self.top_k_features:
                break

        if not selected:
            selected = [
                feat
                for feat, _, _ in scores[
                    : max(1, min(self.top_k_features, len(scores)))
                ]
            ]

        return selected

    # -----------------------------
    # Prototype series
    # -----------------------------
    def _compute_global_stats(self):
        """
        Estadísticos globales para estandarizar features y targets.
        """
        for feat in self.feature_cols:
            x = self.train_panel.df.get_column(feat).to_numpy().astype(float)
            x = x[np.isfinite(x)]
            self.feature_stats[feat] = (float(np.mean(x)), self._safe_std(x))

        for target in self.target_cols:
            y = self.train_panel.df.get_column(target).to_numpy().astype(float)
            y = y[np.isfinite(y)]
            self.target_stats[target] = (float(np.mean(y)), self._safe_std(y))

    def _build_prototype(
        self, target: str, feature_subset: Sequence[str], sampled_seq_ids: Sequence[int]
    ):
        """
        Construye una serie prototipo de dominio por mediana por timestep.
        """
        T = self.train_panel.seq_len

        ys = []
        Xs = {feat: [] for feat in feature_subset}

        for seq_ix in sampled_seq_ids:
            seq = self.train_panel.sequence_df(seq_ix)
            y = seq.get_column(target).to_numpy().astype(float)
            y = self._scale_target(target, y)
            ys.append(y)

            for feat in feature_subset:
                x = seq.get_column(feat).to_numpy().astype(float)
                x = self._scale_feature(feat, x)
                Xs[feat].append(x)

        y_stack = np.vstack(ys)
        y_proto = np.nanmedian(y_stack, axis=0)

        X_proto_cols = []
        for feat in feature_subset:
            x_stack = np.vstack(Xs[feat])
            x_proto = np.nanmedian(x_stack, axis=0)
            X_proto_cols.append(x_proto)

        X_proto = (
            np.column_stack(X_proto_cols)
            if X_proto_cols
            else np.empty((T, 0), dtype=float)
        )
        X_proto = np.nan_to_num(X_proto, nan=0.0, posinf=0.0, neginf=0.0)

        return y_proto, X_proto

    def _fit_member(
        self, target: str, order: tuple[int, int, int], feature_subset: Sequence[str]
    ):
        """
        Ajusta un miembro SARIMAX sobre una serie prototipo.
        """
        sampled_seq_ids = self.rng.choice(
            self.train_panel.sequence_ids,
            size=len(self.train_panel.sequence_ids),
            replace=True,
        )

        y_proto, X_proto = self._build_prototype(
            target, feature_subset, sampled_seq_ids
        )
        exog = self._prepare_exog_matrix(X_proto)

        model = SARIMAX(
            y_proto,
            exog=exog,
            order=order,
            seasonal_order=(0, 0, 0, 0),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        res = model.fit(disp=False)
        return {
            "target": target,
            "order": order,
            "features": list(feature_subset),
            "feature_idx": [self.feature_cols.index(f) for f in feature_subset],
            "results": res,
            "aic": float(res.aic) if np.isfinite(res.aic) else np.inf,
        }

    # -----------------------------
    # Fit
    # -----------------------------
    def fit(self):
        self._compute_global_stats()

        # Selección de covariables por target
        for target in self.target_cols:
            self.selected_features[target] = self._select_features_for_target(target)

        # Entrenamos ensemble por target
        for target, orders in zip(self.target_cols, [self.orders_t0, self.orders_t1]):
            candidate_members = []

            for order in orders:
                for _ in range(self.n_bootstrap):
                    base_feats = self.selected_features[target]
                    if len(base_feats) > self.max_features_per_member:
                        chosen_feats = list(
                            self.rng.choice(
                                base_feats,
                                size=self.max_features_per_member,
                                replace=False,
                            )
                        )
                    else:
                        chosen_feats = list(base_feats)

                    # Garantizamos que haya al menos una feature
                    if not chosen_feats:
                        chosen_feats = list(self.feature_cols[:1])

                    try:
                        member = self._fit_member(target, order, chosen_feats)
                        candidate_members.append(member)
                    except Exception:
                        continue

            if not candidate_members:
                raise RuntimeError(
                    f"No se pudo entrenar ningún miembro del ensemble para {target}."
                )

            # Pesos por AIC (cuanto menor, mejor)
            aics = np.array([m["aic"] for m in candidate_members], dtype=float)
            finite_mask = np.isfinite(aics)
            if not finite_mask.any():
                weights = np.ones(len(candidate_members), dtype=float) / len(
                    candidate_members
                )
            else:
                aics_f = aics[finite_mask]
                aic_min = np.min(aics_f)
                scores = np.zeros(len(candidate_members), dtype=float)
                scores[finite_mask] = np.exp(-0.5 * (aics_f - aic_min))
                if scores.sum() <= 0:
                    weights = np.ones(len(candidate_members), dtype=float) / len(
                        candidate_members
                    )
                else:
                    weights = scores / scores.sum()

            for m, w in zip(candidate_members, weights):
                m["weight"] = float(w)

            self.members[target] = candidate_members

        self._is_fitted = True
        return self

    # -----------------------------
    # State cache
    # -----------------------------
    def _init_seq_cache(self, seq_ix: int):
        """
        Crea estado independiente por secuencia para todos los miembros.
        """
        self._seq_cache[seq_ix] = {
            "t0": [copy.deepcopy(m["results"]) for m in self.members["t0"]],
            "t1": [copy.deepcopy(m["results"]) for m in self.members["t1"]],
        }

    def _forecast_target(
        self, seq_ix: int, target: str, data_point: DataPoint
    ) -> float:
        """
        Forecast ensemble para un target.
        """
        if seq_ix not in self._seq_cache:
            self._init_seq_cache(seq_ix)

        preds = []
        weights = []

        member_list = self.members[target]
        cache_list = self._seq_cache[seq_ix][target]

        for idx, member in enumerate(member_list):
            res = cache_list[idx]
            exog = self._prepare_exog_row(data_point, member["features"])

            try:
                pred_scaled = float(res.forecast(steps=1, exog=exog)[0])
            except Exception:
                pred_scaled = 0.0

            pred = float(self._unscale_target(target, pred_scaled))
            preds.append(pred)
            weights.append(member["weight"])

            # Avanza el estado del miembro con su propia predicción
            try:
                updated = res.append(
                    endog=np.array([pred_scaled], dtype=float), exog=exog, refit=False
                )
                cache_list[idx] = updated
            except Exception:
                cache_list[idx] = res

        preds = np.asarray(preds, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if weights.sum() <= 0:
            return float(np.mean(preds))
        return float(np.sum(weights * preds))

    # -----------------------------
    # Predict
    # -----------------------------

    def _contiguous_blocks(self, idx: np.ndarray):
        if len(idx) == 0:
            return []
        splits = np.where(np.diff(idx) != 1)[0] + 1
        return np.split(idx, splits)

    def _forecast_member_block(
        self, member, X_block: np.ndarray, target: str
    ) -> np.ndarray:
        """
        Forecast multi-step de un miembro sobre un bloque contiguo.
        Devuelve predicción en escala original.
        """
        if len(X_block) == 0:
            return np.array([], dtype=float)

        feature_idx = member.get("feature_idx")
        if feature_idx is None:
            feature_idx = [self.feature_cols.index(f) for f in member["features"]]

        X_sub = X_block[:, feature_idx]
        exog = self._prepare_exog_matrix(X_sub)

        pred_scaled = (
            member["results"].get_forecast(steps=len(X_sub), exog=exog).predicted_mean
        )

        pred_scaled = np.asarray(pred_scaled, dtype=float)
        return self._unscale_target(target, pred_scaled)

    def _forecast_sequence(self, panel, seq_ix: int):
        """
        Calcula de una vez todas las predicciones de una secuencia.
        """
        seq = panel.sequence_df(seq_ix)

        X_all = seq.select(list(self.feature_cols)).to_numpy().astype(float)
        y_all = seq.select(list(self.target_cols)).to_numpy().astype(float)
        mask = seq.get_column(self.warmup_col).to_numpy().astype(bool)

        preds = np.full((seq.height, len(self.target_cols)), np.nan, dtype=float)

        pred_positions = np.where(mask)[0]
        blocks = self._contiguous_blocks(pred_positions)

        for block in blocks:
            X_block = X_all[block]

            for target_idx, target in enumerate(self.target_cols):
                member_list = self.members[target]

                member_preds = []
                weights = []

                for member in member_list:
                    p = self._forecast_member_block(member, X_block, target)
                    member_preds.append(p)
                    weights.append(member["weight"])

                member_preds = np.vstack(member_preds)
                weights = np.asarray(weights, dtype=float)

                if weights.sum() <= 0:
                    block_pred = np.mean(member_preds, axis=0)
                else:
                    block_pred = np.average(member_preds, axis=0, weights=weights)

                preds[block, target_idx] = block_pred

        return preds, y_all, mask

    def evaluate(self, panel=None) -> dict:
        """
        Evaluación rápida por secuencia completa.
        Mucho más rápida que llamar predict fila a fila.
        """
        if panel is None:
            panel = self.test_panel

        predictions = []
        targets = []

        for seq_ix in panel.sequence_ids:
            pred_seq, y_seq, mask = self._forecast_sequence(panel, seq_ix)

            if not mask.any():
                continue

            predictions.append(pred_seq[mask])
            targets.append(y_seq[mask])

            # cache opcional para predict() online
            self._seq_prediction_cache[seq_ix] = pred_seq

        if len(predictions) == 0:
            return {
                self.target_cols[0]: 0.0,
                self.target_cols[1]: 0.0,
                "weighted_pearson": 0.0,
            }

        predictions = np.vstack(predictions)
        targets = np.vstack(targets)

        scores = {}
        for ix, target_name in enumerate(self.target_cols):
            scores[target_name] = weighted_pearson_correlation(
                targets[:, ix],
                predictions[:, ix],
            )

        scores["weighted_pearson"] = float(np.mean(list(scores.values())))
        return scores

    def predict(self, data_point):
        """
        Devuelve la predicción ya cacheada por secuencia.
        """
        if not data_point.need_prediction:
            return None

        seq_ix = data_point.seq_ix

        if seq_ix not in self._seq_prediction_cache:
            pred_seq, _, _ = self._forecast_sequence(self.test_panel, seq_ix)
            self._seq_prediction_cache[seq_ix] = pred_seq

        return np.asarray(
            self._seq_prediction_cache[seq_ix][data_point.step_in_seq], dtype=float
        )

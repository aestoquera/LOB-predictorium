from __future__ import annotations
from collections import deque
from typing import Optional, Sequence

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tqdm.auto import tqdm

from .utils import PredictionModel, DataPoint

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Utilidad para evaluar weighted Pearson de forma streaming
# ============================================================
class _WeightedPearsonAccumulator:
    def __init__(self):
        self.sum_w = 0.0
        self.sum_wy = 0.0
        self.sum_wp = 0.0
        self.sum_wyy = 0.0
        self.sum_wpp = 0.0
        self.sum_wyp = 0.0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), -6.0, 6.0)

        weights = np.maximum(np.abs(y_true), 1e-8)
        self.sum_w += float(np.sum(weights))
        self.sum_wy += float(np.sum(weights * y_true))
        self.sum_wp += float(np.sum(weights * y_pred))
        self.sum_wyy += float(np.sum(weights * y_true * y_true))
        self.sum_wpp += float(np.sum(weights * y_pred * y_pred))
        self.sum_wyp += float(np.sum(weights * y_true * y_pred))

    def value(self) -> float:
        if self.sum_w <= 0:
            return 0.0

        mean_y = self.sum_wy / self.sum_w
        mean_p = self.sum_wp / self.sum_w

        cov = (self.sum_wyp / self.sum_w) - (mean_y * mean_p)
        var_y = (self.sum_wyy / self.sum_w) - (mean_y * mean_y)
        var_p = (self.sum_wpp / self.sum_w) - (mean_p * mean_p)

        if var_y <= 0 or var_p <= 0:
            return 0.0

        return float(cov / np.sqrt(var_y * var_p))


# ============================================================
# Dataset streaming por ventanas desde parquet
# ============================================================
class StreamingWindowDataset(IterableDataset):
    """
    Lee el parquet por row-groups y genera ventanas on-the-fly.
    Supone que el parquet está ordenado por:
      - seq_ix
      - step_in_seq
    """

    def __init__(
        self,
        parquet_path: str,
        feature_cols: Sequence[str],
        target_cols: Sequence[str],
        window_size: int,
        seq_col: str = "seq_ix",
        step_col: str = "step_in_seq",
        warmup_col: str = "need_prediction",
        prediction_only: bool = False,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_cols = tuple(feature_cols)
        self.target_cols = tuple(target_cols)
        self.window_size = int(window_size)
        self.seq_col = seq_col
        self.step_col = step_col
        self.warmup_col = warmup_col
        self.prediction_only = prediction_only

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        worker = get_worker_info()

        row_groups = list(range(pf.num_row_groups))
        if worker is not None:
            row_groups = row_groups[worker.id :: worker.num_workers]

        cols = [
            self.seq_col,
            self.step_col,
            self.warmup_col,
            *self.feature_cols,
            *self.target_cols,
        ]

        seq_ix_current = None
        x_buffer = deque(maxlen=self.window_size)

        for rg in row_groups:
            table = pf.read_row_group(rg, columns=cols)
            df = pl.from_arrow(table)

            # Orden importante por seguridad
            df = df.sort([self.seq_col, self.step_col])

            seq_arr = df.get_column(self.seq_col).to_numpy()
            warmup_arr = df.get_column(self.warmup_col).to_numpy().astype(bool)

            X = (
                df.select(list(self.feature_cols))
                .to_numpy()
                .astype(np.float32, copy=False)
            )
            Y = (
                df.select(list(self.target_cols))
                .to_numpy()
                .astype(np.float32, copy=False)
            )

            for i in range(df.height):
                seq_ix = int(seq_arr[i])

                if seq_ix_current is None or seq_ix != seq_ix_current:
                    seq_ix_current = seq_ix
                    x_buffer.clear()

                x_buffer.append(X[i])

                if len(x_buffer) < self.window_size:
                    continue

                if self.prediction_only and not warmup_arr[i]:
                    continue

                window = np.asarray(x_buffer, dtype=np.float32)
                target = Y[i]

                yield (
                    torch.from_numpy(np.ascontiguousarray(window)),
                    torch.from_numpy(np.ascontiguousarray(target)),
                )


# ============================================================
# Base abstracta: recibe rutas, no paneles en memoria
# ============================================================
class DeepLearningForecastingModel(PredictionModel, nn.Module):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        seq_col: str = "seq_ix",
        step_col: str = "step_in_seq",
        warmup_col: str = "need_prediction",
        target_cols: tuple[str, ...] = ("t0", "t1"),
        ignore_cols: tuple[str, ...] = (),
    ):
        nn.Module.__init__(self)
        self.train_path = train_path
        self.test_path = test_path

        self.seq_col = seq_col
        self.step_col = step_col
        self.warmup_col = warmup_col
        self.target_cols = target_cols
        self.ignore_cols = ignore_cols

        schema = pl.read_parquet(train_path, n_rows=0)
        self._all_columns = tuple(schema.columns)
        self._feature_cols = self._infer_feature_cols()

        self._is_fitted = False

    def _infer_feature_cols(self) -> tuple[str, ...]:
        excluded = {
            self.seq_col,
            self.step_col,
            self.warmup_col,
            *self.target_cols,
            *self.ignore_cols,
        }
        return tuple(c for c in self._all_columns if c not in excluded)

    @property
    def feature_cols(self) -> tuple[str, ...]:
        return self._feature_cols

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        raise NotImplementedError

    def evaluate(self, path: Optional[str] = None) -> dict:
        """
        Evaluación streaming sobre el parquet de test (o sobre otro path si se pasa).
        No carga todo el dataset en memoria.
        """
        if path is None:
            path = self.test_path

        self.reset_state()

        accs = {t: _WeightedPearsonAccumulator() for t in self.target_cols}

        seq_ids = (
            self.test_sequence_ids()
            if str(path) == self.test_path
            else self._sequence_ids(path)
        )

        for seq_ix in seq_ids:
            self.on_sequence_start(seq_ix)
            seq = self._load_sequence(path, seq_ix)

            for row in seq.iter_rows(named=False):
                data_point, labels = self._row_to_datapoint(row)
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

                for i, target_name in enumerate(self.target_cols):
                    accs[target_name].update(labels[i : i + 1], pred[i : i + 1])

        scores = {t: accs[t].value() for t in self.target_cols}
        scores["weighted_pearson"] = float(np.mean(list(scores.values())))
        return scores

    def plot_sequence_predictions(self, seq_ix: int, max_steps: Optional[int] = None):
        """
        Genera predicciones sobre una secuencia y dibuja:
          1) pred vs real
          2) error en el tiempo
          3) histograma de errores
          4) scatter real vs pred
        """
        seq = self._load_sequence(self.test_path, seq_ix)
        if max_steps is not None:
            seq = seq[:max_steps]

        t_vals = seq.get_column(self.step_col).to_numpy()
        real = seq.select(list(self.target_cols)).to_numpy(dtype=np.float32)
        need_pred = seq.get_column(self.warmup_col).to_numpy().astype(bool)

        pred = np.zeros_like(real, dtype=np.float32)

        # reset de buffers para esta secuencia
        self.on_sequence_start(seq_ix)

        for i, row in enumerate(seq.iter_rows(named=False)):
            data_point, _ = self._row_to_datapoint(row)
            p = self.predict(data_point)
            if p is not None:
                pred[i] = p
            else:
                pred[i] = np.nan  # pasos warm-up o no predicción

        real_t0 = real[:, 0]
        real_t1 = real[:, 1]
        pred_t0 = pred[:, 0]
        pred_t1 = pred[:, 1]

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
        sns.histplot(err_t0, kde=True, label="t0", bins=50, color="C0")
        sns.histplot(err_t1, kde=True, label="t1", bins=50, color="C1")
        plt.title("Distribución de errores")
        plt.legend()
        plt.show()

        # =========================
        # 4. SCATTER REAL VS PRED
        # =========================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(real_t0, pred_t0, alpha=0.5, color="C0")
        axes[0].set_title("t0: real vs pred")
        axes[0].set_xlabel("real")
        axes[0].set_ylabel("pred")

        axes[1].scatter(real_t1, pred_t1, alpha=0.5, color="C1")
        axes[1].set_title("t1: real vs pred")
        axes[1].set_xlabel("real")
        axes[1].set_ylabel("pred")
        plt.tight_layout()
        plt.show()

    def _init_loss_tracking(self):
        """Inicializa listas para trackear los losses durante fit."""
        self.train_losses: list[float] = []
        self.eval_losses: list[float] = []

    def track_loss(self, train_loss: float, eval_loss: float = None):
        """Agrega un valor de loss a las listas internas."""
        self.train_losses.append(train_loss)
        if eval_loss is not None:
            self.eval_losses.append(eval_loss)

    def plot_loss_curve(self):
        """Plotea la evolución de los losses de entrenamiento y evaluación."""
        if not hasattr(self, "train_losses") or len(self.train_losses) == 0:
            print("No hay losses registrados. Ejecuta fit() con tracking habilitado.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss", marker="o")
        if hasattr(self, "eval_losses") and len(self.eval_losses) > 0:
            plt.plot(self.eval_losses, label="Eval Loss", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Evolución del Loss durante el entrenamiento")
        plt.legend()
        plt.grid(True)
        plt.show()


# ============================================================
# Backbone: GRU + Attention
# ============================================================
class GRUAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim debe ser divisible por num_heads")

        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_norm(x)
        h, _ = self.gru(x)
        attn_out, _ = self.attn(h, h, h)
        out = attn_out[:, -1, :]
        out = self.post_norm(out)
        return self.head(out).squeeze(-1)


# ============================================================
# Modelo concreto Torch
# ============================================================
class TorchSequenceModel(DeepLearningForecastingModel):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        window_size: int = 50,
        hidden_dim: int = 64,
        num_heads: int = 4,
        lr: float = 1e-3,
        epochs: int = 5,
        device: Optional[str] = None,
        seq_col: str = "seq_ix",
        step_col: str = "step_in_seq",
        warmup_col: str = "need_prediction",
        target_cols: tuple[str, ...] = ("t0", "t1"),
        ignore_cols: tuple[str, ...] = (),
    ):
        super().__init__(
            train_path=train_path,
            test_path=test_path,
            seq_col=seq_col,
            step_col=step_col,
            warmup_col=warmup_col,
            target_cols=target_cols,
            ignore_cols=ignore_cols,
        )

        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.lr = float(lr)
        self.epochs = int(epochs)

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        input_dim = len(self.feature_cols)

        self.model_t0 = GRUAttentionModel(
            input_dim, hidden_dim=hidden_dim, num_heads=num_heads
        ).to(self.device)
        self.model_t1 = GRUAttentionModel(
            input_dim, hidden_dim=hidden_dim, num_heads=num_heads
        ).to(self.device)

        self.opt_t0 = torch.optim.AdamW(self.model_t0.parameters(), lr=self.lr)
        self.opt_t1 = torch.optim.AdamW(self.model_t1.parameters(), lr=self.lr)

        self.loss_fn = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        self.buffers: dict[int, deque] = {}

    def reset_state(self) -> None:
        self.buffers = {}

    def on_sequence_start(self, seq_ix: int) -> None:
        self.buffers[seq_ix] = deque(maxlen=self.window_size)

    def fit(
        self, batch_size: int = 128, num_workers: int = 0, prediction_only: bool = False
    ):
        dataset = StreamingWindowDataset(
            parquet_path=self.train_path,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            window_size=self.window_size,
            seq_col=self.seq_col,
            step_col=self.step_col,
            warmup_col=self.warmup_col,
            prediction_only=prediction_only,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        self._init_loss_tracking()
        for epoch in range(self.epochs):
            self.model_t0.train()
            self.model_t1.train()

            total_loss0 = 0.0
            total_loss1 = 0.0
            n_batches = 0

            progress = tqdm(
                loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch"
            )

            for X, y in progress:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y0 = y[:, 0]
                y1 = y[:, 1]

                self.opt_t0.zero_grad(set_to_none=True)
                self.opt_t1.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type, enabled=(self.device.type == "cuda")
                ):
                    pred0 = self.model_t0(X)
                    loss0 = self.loss_fn(pred0, y0)

                    pred1 = self.model_t1(X)
                    loss1 = self.loss_fn(pred1, y1)

                    loss = loss0 + loss1

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.opt_t0)
                self.scaler.unscale_(self.opt_t1)
                torch.nn.utils.clip_grad_norm_(self.model_t0.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.model_t1.parameters(), 1.0)

                self.scaler.step(self.opt_t0)
                self.scaler.step(self.opt_t1)
                self.scaler.update()

                total_loss0 += float(loss0.item())
                total_loss1 += float(loss1.item())

                avg0 = total_loss0 / max(n_batches, 1)
                avg1 = total_loss1 / max(n_batches, 1)
                epoch_loss = (avg0 + avg1) / 2
                self.track_loss(train_loss=epoch_loss)

                n_batches += 1

                progress.set_postfix(
                    {
                        "loss_t0": avg0,
                        "loss_t1": avg1,
                    }
                )

        self._is_fitted = True
        return self

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        seq_ix = int(data_point.seq_ix)
        x = np.asarray(data_point.state, dtype=np.float32)

        if seq_ix not in self.buffers:
            self.on_sequence_start(seq_ix)

        self.buffers[seq_ix].append(x)

        if not data_point.need_prediction:
            return None

        if len(self.buffers[seq_ix]) < self.window_size:
            return np.zeros(2, dtype=np.float32)

        window = np.asarray(self.buffers[seq_ix], dtype=np.float32)[None, :, :]
        window = torch.from_numpy(np.ascontiguousarray(window)).to(self.device)

        self.model_t0.eval()
        self.model_t1.eval()

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type, enabled=(self.device.type == "cuda")
            ):
                pred0 = float(self.model_t0(window).cpu().item())
                pred1 = float(self.model_t1(window).cpu().item())

        pred = np.array([pred0, pred1], dtype=np.float32)
        return np.clip(pred, -6.0, 6.0)

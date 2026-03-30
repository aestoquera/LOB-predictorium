"""Clean, reviewed forecasting trainers for large parquet datasets.

This file contains two trainers:

- ForecastingTrainer: stable baseline
- SpikeAwareTrainer: spike-aware variant for rare-event / surprise modeling

The code is designed to be out-of-core, sequence-wise, and conservative.
It also avoids redundant end-of-epoch validation and keeps the training history
consistent for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import json
import math
import random
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DataConfig:
    train_path: str
    valid_path: str
    seq_len: int = 1000
    seq_col: str = "seq_ix"
    step_col: str = "step_in_seq"
    need_pred_col: str = "need_prediction"
    target_cols: Tuple[str, str] = ("t0", "t1")
    batch_rows: int = 64_000
    num_workers: int = 0
    pin_memory: bool = True
    target_region_only: bool = True
    use_lags: bool = True
    emphasize_spikes: bool = False
    spike_alpha: float = 1.5
    spike_floor: float = 1e-4


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    patience: int = 3
    min_delta: float = 1e-4
    log_every: int = 50
    val_every_steps: int = 500
    max_train_steps: Optional[int] = None
    amp: bool = True
    save_dir: str = "runs/exp_001"
    checkpoint_name: str = "best.pt"


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    dropout: float = 0.10
    kernel_size: int = 3
    num_layers: int = 4
    num_scales: int = 4


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _choose_group_count(channels: int, max_groups: int = 8) -> int:
    """Choose the largest group count <= max_groups that divides channels."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class OnlineStandardScaler:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)

        if self.count == 0:
            self.count = n
            self.mean = batch_mean
            self.m2 = batch_var * n
            return

        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        self.m2 = self.m2 + batch_var * n + (delta**2) * self.count * n / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        denom = max(self.count - 1, 1)
        var = self.m2 / denom
        return np.sqrt(np.maximum(var, 1e-12))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


# =============================================================================
# Streaming parquet reader
# =============================================================================


class SequenceStream:
    """Iterate over full sequences from a parquet file without loading everything."""

    def __init__(self, parquet_path: str, cfg: DataConfig):
        self.parquet_path = parquet_path
        self.cfg = cfg
        self.pf = pq.ParquetFile(parquet_path)
        self.schema = self.pf.schema_arrow
        self.columns = self.schema.names

        required = {cfg.seq_col, cfg.step_col, cfg.need_pred_col, *cfg.target_cols}
        missing = required - set(self.columns)
        if missing:
            raise ValueError(f"Missing columns in {parquet_path}: {sorted(missing)}")

        self.feature_cols = [
            c for c in self.columns
            if c not in required and pa.types.is_floating(self.schema.field(c).type)
        ]
        if not self.feature_cols:
            raise ValueError("No float covariate columns detected.")

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        yield from self.pf.iter_batches(batch_size=self.cfg.batch_rows, columns=self.columns)

    def iter_sequences(self) -> Iterator[Dict[str, np.ndarray]]:
        current_seq = None
        buf: Dict[str, List[np.ndarray]] = {c: [] for c in self.columns}

        def flush() -> Optional[Dict[str, np.ndarray]]:
            if current_seq is None:
                return None
            return {c: np.concatenate(parts, axis=0) for c, parts in buf.items()}

        for batch in self.iter_batches():
            tbl = pa.Table.from_batches([batch])
            cols = {c: tbl[c].to_numpy(zero_copy_only=False) for c in self.columns}
            seqs = cols[self.cfg.seq_col]

            cuts = [0]
            cuts.extend((np.flatnonzero(np.diff(seqs) != 0) + 1).tolist())
            cuts.append(len(seqs))

            for a, b in zip(cuts[:-1], cuts[1:]):
                seq_id = int(seqs[a])
                if current_seq is None:
                    current_seq = seq_id
                if seq_id != current_seq:
                    out = flush()
                    if out is not None:
                        yield out
                    current_seq = seq_id
                    buf = {c: [] for c in self.columns}
                for c in self.columns:
                    buf[c].append(np.asarray(cols[c][a:b]))

        out = flush()
        if out is not None:
            yield out


# =============================================================================
# Metrics and losses
# =============================================================================


def weighted_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_pred = np.clip(y_pred, -6.0, 6.0)

    weights = np.abs(y_true)
    weights = np.maximum(weights, 1e-8)
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0.0

    mean_true = np.sum(y_true * weights) / sum_w
    mean_pred = np.sum(y_pred * weights) / sum_w
    dev_true = y_true - mean_true
    dev_pred = y_pred - mean_pred

    cov = np.sum(weights * dev_true * dev_pred) / sum_w
    var_true = np.sum(weights * dev_true**2) / sum_w
    var_pred = np.sum(weights * dev_pred**2) / sum_w
    if var_true <= 0 or var_pred <= 0:
        return 0.0
    return float(cov / (np.sqrt(var_true) * np.sqrt(var_pred)))


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    err = pred - target
    abs_err = err.abs()
    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    lin = abs_err - quad
    loss = 0.5 * quad**2 + delta * lin
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
    return loss.sum() / denom


def weighted_pearson_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = torch.clamp(pred, -6.0, 6.0)
    mask_f = mask.unsqueeze(-1).float()
    weights = torch.abs(target) * mask_f
    weights = torch.clamp(weights, min=1e-8)

    wsum = weights.sum(dim=(0, 1)).clamp_min(1e-8)
    mean_true = (target * weights).sum(dim=(0, 1)) / wsum
    mean_pred = (pred * weights).sum(dim=(0, 1)) / wsum

    dev_true = target - mean_true
    dev_pred = pred - mean_pred

    cov = (weights * dev_true * dev_pred).sum(dim=(0, 1)) / wsum
    var_true = (weights * dev_true**2).sum(dim=(0, 1)) / wsum
    var_pred = (weights * dev_pred**2).sum(dim=(0, 1)) / wsum

    corr = cov / (torch.sqrt(var_true.clamp_min(1e-8)) * torch.sqrt(var_pred.clamp_min(1e-8)) + 1e-8)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return 1.0 - corr.mean()


# =============================================================================
# Models
# =============================================================================


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.norm = nn.GroupNorm(num_groups=_choose_group_count(out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.crop = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        if self.crop > 0:
            h = h[:, :, :-self.crop]
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        res = self.res_proj(x)
        if self.crop > 0:
            res = res[:, :, -h.shape[-1]:]
        return h + res


class BaselineForecaster(nn.Module):
    """Simple causal CNN that predicts both targets directly."""

    def __init__(self, input_dim: int, cfg: ModelConfig, output_dim: int = 2):
        super().__init__()
        layers = []
        d_in = input_dim
        for _ in range(cfg.num_layers):
            layers.append(CausalConvBlock(d_in, cfg.hidden_dim, cfg.kernel_size, cfg.dropout))
            d_in = cfg.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.backbone(x)
        y = self.head(h)
        return y.transpose(1, 2)


class SpikeAwareForecaster(nn.Module):
    """Multiscale causal CNN with a residual t0 head and a t1 head."""

    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        blocks = []
        d_in = input_dim
        for i in range(cfg.num_scales):
            dilation = 2**i
            block = CausalConvBlock(d_in, cfg.hidden_dim, cfg.kernel_size, cfg.dropout)
            blocks.append(block)
            d_in = cfg.hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.head_t0 = nn.Sequential(
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.hidden_dim, 1, kernel_size=1),
        )
        self.head_t1 = nn.Sequential(
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        h = self.backbone(x)
        delta_t0 = self.head_t0(h).transpose(1, 2)
        pred_t1 = self.head_t1(h).transpose(1, 2)
        return delta_t0, pred_t1


# =============================================================================
# Dataset
# =============================================================================


class ForecastSequenceDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        data_cfg: DataConfig,
        feature_scaler: OnlineStandardScaler,
        target_scaler: OnlineStandardScaler,
        training: bool,
    ):
        super().__init__()
        self.stream = SequenceStream(parquet_path, data_cfg)
        self.cfg = data_cfg
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.training = training

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for seq in self.stream.iter_sequences():
            x_cov = np.stack([seq[c] for c in self.stream.feature_cols], axis=1).astype(np.float32)
            y = np.stack([seq[c] for c in self.cfg.target_cols], axis=1).astype(np.float32)
            need = seq[self.cfg.need_pred_col].astype(np.float32)
            step = seq[self.cfg.step_col].astype(np.float32)[:, None] / max(self.cfg.seq_len - 1, 1)

            x_cov = self.feature_scaler.transform(x_cov).astype(np.float32)
            y_scaled = self.target_scaler.transform(y).astype(np.float32)

            lag = np.zeros_like(y_scaled, dtype=np.float32)
            if self.cfg.use_lags:
                lag[1:] = y_scaled[:-1]
                lag[0] = y_scaled[0]
            x = np.concatenate([x_cov, step, lag], axis=1)

            mask = need > 0.5 if self.cfg.target_region_only else np.ones_like(need, dtype=bool)

            sample_weight = np.ones_like(need, dtype=np.float32)
            if self.cfg.emphasize_spikes:
                amp = np.abs(y[:, 0])
                amp = np.power(np.maximum(amp, self.cfg.spike_floor), self.cfg.spike_alpha)
                amp = amp / (amp.mean() + 1e-8)
                diff = np.zeros_like(y[:, 0], dtype=np.float32)
                diff[1:] = np.abs(y[1:, 0] - y[:-1, 0])
                diff = np.power(np.maximum(diff, self.cfg.spike_floor), self.cfg.spike_alpha)
                diff = diff / (diff.mean() + 1e-8)
                sample_weight = np.maximum(amp, diff).astype(np.float32)

            yield {
                "x": torch.from_numpy(x),
                "y": torch.from_numpy(y_scaled),
                "lag": torch.from_numpy(lag),
                "mask": torch.from_numpy(mask),
                "sample_weight": torch.from_numpy(sample_weight),
            }


def collate_full_sequence(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "lag": torch.stack([b["lag"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "sample_weight": torch.stack([b["sample_weight"] for b in batch], dim=0),
    }


def make_loader(dataset: IterableDataset, cfg: DataConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=False,
        collate_fn=collate_full_sequence,
    )


# =============================================================================
# Eval outputs and early stopping
# =============================================================================


@dataclass
class EvalOutputs:
    loss: float
    rmse: float
    weighted_pearson_mean: float
    weighted_pearson_t0: float
    weighted_pearson_t1: float
    y_true: np.ndarray
    y_pred: np.ndarray
    mask: np.ndarray


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return False


# =============================================================================
# Base trainer
# =============================================================================


class _BaseTrainer:
    def __init__(self, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg: ModelConfig):
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.save_dir = Path(train_cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        set_seed(train_cfg.seed)
        self.feature_scaler, self.target_scaler = self.fit_scalers()

        self.train_ds = ForecastSequenceDataset(
            data_cfg.train_path, data_cfg, self.feature_scaler, self.target_scaler, training=True
        )
        self.valid_ds = ForecastSequenceDataset(
            data_cfg.valid_path, data_cfg, self.feature_scaler, self.target_scaler, training=False
        )
        self.train_loader = make_loader(self.train_ds, data_cfg)
        self.valid_loader = make_loader(self.valid_ds, data_cfg)
        self.feature_cols = SequenceStream(data_cfg.train_path, data_cfg).feature_cols

        (self.save_dir / "scalers.json").write_text(
            json.dumps(
                {
                    "feature_mean": self.feature_scaler.mean.tolist(),
                    "feature_std": self.feature_scaler.std.tolist(),
                    "target_mean": self.target_scaler.mean.tolist(),
                    "target_std": self.target_scaler.std.tolist(),
                },
                indent=2,
            )
        )

        self.best_val_loss = math.inf
        self.global_step = 0
        self.history = {
            "train_steps": [],
            "train_loss": [],
            "eval_steps": [],
            "eval_loss": [],
            "eval_wpc_mean": [],
            "eval_wpc_t0": [],
            "eval_wpc_t1": [],
        }
        self.early = EarlyStopping(train_cfg.patience, train_cfg.min_delta)

    def fit_scalers(self) -> Tuple[OnlineStandardScaler, OnlineStandardScaler]:
        stream = SequenceStream(self.data_cfg.train_path, self.data_cfg)
        feature_scaler = OnlineStandardScaler(len(stream.feature_cols))
        target_scaler = OnlineStandardScaler(len(self.data_cfg.target_cols))
        for seq in stream.iter_sequences():
            x = np.stack([seq[c] for c in stream.feature_cols], axis=1).astype(np.float64)
            y = np.stack([seq[c] for c in self.data_cfg.target_cols], axis=1).astype(np.float64)
            feature_scaler.update(x)
            target_scaler.update(y)
        return feature_scaler, target_scaler

    @staticmethod
    def _denorm_target(y_scaled: np.ndarray, target_scaler: OnlineStandardScaler) -> np.ndarray:
        return y_scaled * target_scaler.std + target_scaler.mean

    def _record_eval(self, ev: EvalOutputs) -> None:
        self.history["eval_steps"].append(self.global_step)
        self.history["eval_loss"].append(ev.loss)
        self.history["eval_wpc_mean"].append(ev.weighted_pearson_mean)
        self.history["eval_wpc_t0"].append(ev.weighted_pearson_t0)
        self.history["eval_wpc_t1"].append(ev.weighted_pearson_t1)

    def final_metrics(self) -> Dict[str, float]:
        return {
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "last_eval_loss": self.history["eval_loss"][-1] if self.history["eval_loss"] else float("nan"),
            "last_eval_wpc_mean": self.history["eval_wpc_mean"][-1] if self.history["eval_wpc_mean"] else float("nan"),
            "last_eval_wpc_t0": self.history["eval_wpc_t0"][-1] if self.history["eval_wpc_t0"] else float("nan"),
            "last_eval_wpc_t1": self.history["eval_wpc_t1"][-1] if self.history["eval_wpc_t1"] else float("nan"),
        }


# =============================================================================
# Baseline trainer
# =============================================================================


class ForecastingTrainer(_BaseTrainer):
    """Stable baseline: direct prediction of t0 and t1."""

    def __init__(self, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg: ModelConfig):
        super().__init__(data_cfg, train_cfg, model_cfg)
        input_dim = len(self.feature_cols) + 1 + 2  # covariates + step + lag_t0 + lag_t1
        self.model = BaselineForecaster(input_dim=input_dim, cfg=model_cfg).to(train_cfg.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        self.scaler = torch.amp.GradScaler(enabled=train_cfg.amp and train_cfg.device.startswith("cuda"))

    def _save_checkpoint(self) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_cfg": asdict(self.model_cfg),
                "data_cfg": asdict(self.data_cfg),
                "train_cfg": asdict(self.train_cfg),
                "feature_cols": self.feature_cols,
            },
            self.save_dir / self.train_cfg.checkpoint_name,
        )

    @torch.no_grad()
    def evaluate(self) -> EvalOutputs:
        self.model.eval()
        losses, rmses = [], []
        y_true_all, y_pred_all, mask_all = [], [], []

        for batch in self.valid_loader:
            x = batch["x"].to(self.train_cfg.device)
            y = batch["y"].to(self.train_cfg.device)
            mask = batch["mask"].to(self.train_cfg.device)

            pred = self.model(x)
            loss = 0.3 * masked_huber_loss(pred, y, mask) + 0.7 * weighted_pearson_loss(pred, y, mask)
            losses.append(loss.item())

            y_np = y.cpu().numpy()[0]
            p_np = pred.cpu().numpy()[0]
            y_np = self._denorm_target(y_np, self.target_scaler)
            p_np = self._denorm_target(p_np, self.target_scaler)
            m_np = mask.cpu().numpy()[0].astype(bool)

            rmses.append(float(np.sqrt(np.mean((y_np[m_np] - p_np[m_np]) ** 2))))
            y_true_all.append(y_np)
            y_pred_all.append(p_np)
            mask_all.append(m_np)

        y_true = np.concatenate(y_true_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)
        mask = np.concatenate(mask_all, axis=0)
        per_target = [weighted_pearson_correlation(y_true[:, i][mask], y_pred[:, i][mask]) for i in range(2)]

        return EvalOutputs(
            loss=float(np.mean(losses)),
            rmse=float(np.mean(rmses)),
            weighted_pearson_mean=float(np.mean(per_target)),
            weighted_pearson_t0=float(per_target[0]),
            weighted_pearson_t1=float(per_target[1]),
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
        )

    def fit(self) -> Dict[str, float]:
        for epoch in range(self.train_cfg.epochs):
            self.model.train()
            epoch_losses: List[float] = []
            t0 = time.time()

            for batch in self.train_loader:
                x = batch["x"].to(self.train_cfg.device)
                y = batch["y"].to(self.train_cfg.device)
                mask = batch["mask"].to(self.train_cfg.device)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda" if self.train_cfg.device.startswith("cuda") else "cpu",
                    enabled=self.scaler.is_enabled(),
                ):
                    pred = self.model(x)
                    loss = 0.3 * masked_huber_loss(pred, y, mask) + 0.7 * weighted_pearson_loss(pred, y, mask)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_losses.append(loss.item())
                self.global_step += 1
                self.history["train_steps"].append(self.global_step)
                self.history["train_loss"].append(loss.item())

                if self.global_step % self.train_cfg.log_every == 0:
                    recent = np.mean(epoch_losses[-self.train_cfg.log_every :])
                    print(f"epoch={epoch} step={self.global_step} train_loss={recent:.6f}")

                if self.train_cfg.max_train_steps is not None and self.global_step >= self.train_cfg.max_train_steps:
                    break

                if self.global_step % self.train_cfg.val_every_steps == 0:
                    ev = self.evaluate()
                    self._record_eval(ev)
                    print(
                        f"[val] step={self.global_step} loss={ev.loss:.6f} rmse={ev.rmse:.6f} "
                        f"wpc_mean={ev.weighted_pearson_mean:.6f} t0={ev.weighted_pearson_t0:.6f} t1={ev.weighted_pearson_t1:.6f}"
                    )
                    if ev.loss < self.best_val_loss - self.train_cfg.min_delta:
                        self.best_val_loss = ev.loss
                        self._save_checkpoint()
                    self.early.step(ev.loss)
                    if self.early.should_stop:
                        print("Early stopping triggered during epoch.")
                        return self.final_metrics()

            # Only run an end-of-epoch validation if the last batch did not already trigger one.
            if not self.history["eval_steps"] or self.history["eval_steps"][-1] != self.global_step:
                ev = self.evaluate()
                self._record_eval(ev)
                print(
                    f"epoch={epoch} train_loss={np.mean(epoch_losses):.6f} val_loss={ev.loss:.6f} "
                    f"val_rmse={ev.rmse:.6f} wpc_mean={ev.weighted_pearson_mean:.6f} time={time.time() - t0:.1f}s"
                )
                if ev.loss < self.best_val_loss - self.train_cfg.min_delta:
                    self.best_val_loss = ev.loss
                    self._save_checkpoint()
                self.early.step(ev.loss)
                if self.early.should_stop:
                    print("Early stopping triggered.")
                    return self.final_metrics()

            if self.train_cfg.max_train_steps is not None and self.global_step >= self.train_cfg.max_train_steps:
                break

        return self.final_metrics()


# =============================================================================
# Spike-aware trainer
# =============================================================================


class SpikeAwareTrainer(_BaseTrainer):
    """Spike-aware version focused on rare, abrupt changes in t0."""

    def __init__(self, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg: ModelConfig):
        super().__init__(data_cfg, train_cfg, model_cfg)
        input_dim = len(self.feature_cols) + 1 + 2  # covariates + step + lag_t0 + lag_t1
        self.model = SpikeAwareForecaster(input_dim=input_dim, cfg=model_cfg).to(train_cfg.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        self.scaler = torch.amp.GradScaler(enabled=train_cfg.amp and train_cfg.device.startswith("cuda"))

    def _save_checkpoint(self) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_cfg": asdict(self.model_cfg),
                "data_cfg": asdict(self.data_cfg),
                "train_cfg": asdict(self.train_cfg),
                "feature_cols": self.feature_cols,
            },
            self.save_dir / self.train_cfg.checkpoint_name,
        )

    def _build_prediction(self, delta_t0: torch.Tensor, lag_t0: torch.Tensor, pred_t1: torch.Tensor) -> torch.Tensor:
        pred_t0 = delta_t0 + lag_t0
        return torch.cat([pred_t0, pred_t1], dim=-1)

    @torch.no_grad()
    def evaluate(self) -> EvalOutputs:
        self.model.eval()
        losses, rmses = [], []
        y_true_all, y_pred_all, mask_all = [], [], []

        for batch in self.valid_loader:
            x = batch["x"].to(self.train_cfg.device)
            y = batch["y"].to(self.train_cfg.device)
            lag = batch["lag"].to(self.train_cfg.device)
            mask = batch["mask"].to(self.train_cfg.device)
            sw = batch["sample_weight"].to(self.train_cfg.device)

            pred_delta_t0, pred_t1 = self.model(x)
            pred = self._build_prediction(pred_delta_t0, lag[..., 0:1], pred_t1)

            mask3 = mask.unsqueeze(-1).float()
            w = sw.unsqueeze(-1).float() * mask3

            delta_t0_true = y[..., 0:1] - lag[..., 0:1]
            loss_t0 = torch.nn.functional.smooth_l1_loss(pred_delta_t0, delta_t0_true, reduction="none")
            loss_t0 = (loss_t0 * w).sum() / w.sum().clamp_min(1.0)

            loss_t1 = torch.nn.functional.smooth_l1_loss(pred_t1, y[..., 1:2], reduction="none")
            loss_t1 = (loss_t1 * w).sum() / w.sum().clamp_min(1.0)

            loss_corr = weighted_pearson_loss(pred, y, mask)
            loss = 0.35 * (loss_t0 + loss_t1) + 0.65 * loss_corr
            losses.append(loss.item())

            y_np = y.cpu().numpy()[0]
            p_np = pred.cpu().numpy()[0]
            y_np = self._denorm_target(y_np, self.target_scaler)
            p_np = self._denorm_target(p_np, self.target_scaler)
            m_np = mask.cpu().numpy()[0].astype(bool)

            rmses.append(float(np.sqrt(np.mean((y_np[m_np] - p_np[m_np]) ** 2))))
            y_true_all.append(y_np)
            y_pred_all.append(p_np)
            mask_all.append(m_np)

        y_true = np.concatenate(y_true_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)
        mask = np.concatenate(mask_all, axis=0)
        per_target = [weighted_pearson_correlation(y_true[:, i][mask], y_pred[:, i][mask]) for i in range(2)]

        return EvalOutputs(
            loss=float(np.mean(losses)),
            rmse=float(np.mean(rmses)),
            weighted_pearson_mean=float(np.mean(per_target)),
            weighted_pearson_t0=float(per_target[0]),
            weighted_pearson_t1=float(per_target[1]),
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
        )

    def fit(self) -> Dict[str, float]:
        for epoch in range(self.train_cfg.epochs):
            self.model.train()
            epoch_losses: List[float] = []
            t0 = time.time()

            for batch in self.train_loader:
                x = batch["x"].to(self.train_cfg.device)
                y = batch["y"].to(self.train_cfg.device)
                lag = batch["lag"].to(self.train_cfg.device)
                mask = batch["mask"].to(self.train_cfg.device)
                sw = batch["sample_weight"].to(self.train_cfg.device)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda" if self.train_cfg.device.startswith("cuda") else "cpu",
                    enabled=self.scaler.is_enabled(),
                ):
                    pred_delta_t0, pred_t1 = self.model(x)
                    pred = self._build_prediction(pred_delta_t0, lag[..., 0:1], pred_t1)

                    mask3 = mask.unsqueeze(-1).float()
                    w = sw.unsqueeze(-1).float() * mask3

                    delta_t0_true = y[..., 0:1] - lag[..., 0:1]
                    loss_t0 = torch.nn.functional.smooth_l1_loss(pred_delta_t0, delta_t0_true, reduction="none")
                    loss_t0 = (loss_t0 * w).sum() / w.sum().clamp_min(1.0)

                    loss_t1 = torch.nn.functional.smooth_l1_loss(pred_t1, y[..., 1:2], reduction="none")
                    loss_t1 = (loss_t1 * w).sum() / w.sum().clamp_min(1.0)

                    loss_corr = weighted_pearson_loss(pred, y, mask)
                    loss = 0.35 * (loss_t0 + loss_t1) + 0.65 * loss_corr

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_losses.append(loss.item())
                self.global_step += 1
                self.history["train_steps"].append(self.global_step)
                self.history["train_loss"].append(loss.item())

                if self.global_step % self.train_cfg.log_every == 0:
                    recent = np.mean(epoch_losses[-self.train_cfg.log_every :])
                    print(f"epoch={epoch} step={self.global_step} train_loss={recent:.6f}")

                if self.train_cfg.max_train_steps is not None and self.global_step >= self.train_cfg.max_train_steps:
                    break

                if self.global_step % self.train_cfg.val_every_steps == 0:
                    ev = self.evaluate()
                    self._record_eval(ev)
                    print(
                        f"[val] step={self.global_step} loss={ev.loss:.6f} rmse={ev.rmse:.6f} "
                        f"wpc_mean={ev.weighted_pearson_mean:.6f} t0={ev.weighted_pearson_t0:.6f} t1={ev.weighted_pearson_t1:.6f}"
                    )
                    if ev.loss < self.best_val_loss - self.train_cfg.min_delta:
                        self.best_val_loss = ev.loss
                        self._save_checkpoint()
                    self.early.step(ev.loss)
                    if self.early.should_stop:
                        print("Early stopping triggered during epoch.")
                        return self.final_metrics()

            # Avoid duplicate validation if the last step already validated.
            if not self.history["eval_steps"] or self.history["eval_steps"][-1] != self.global_step:
                ev = self.evaluate()
                self._record_eval(ev)
                print(
                    f"epoch={epoch} train_loss={np.mean(epoch_losses):.6f} val_loss={ev.loss:.6f} "
                    f"val_rmse={ev.rmse:.6f} wpc_mean={ev.weighted_pearson_mean:.6f} time={time.time() - t0:.1f}s"
                )
                if ev.loss < self.best_val_loss - self.train_cfg.min_delta:
                    self.best_val_loss = ev.loss
                    self._save_checkpoint()
                self.early.step(ev.loss)
                if self.early.should_stop:
                    print("Early stopping triggered.")
                    return self.final_metrics()

            if self.train_cfg.max_train_steps is not None and self.global_step >= self.train_cfg.max_train_steps:
                break

        return self.final_metrics()

    # -----------------------------------------------------------------
    # Diagnostics / plots
    # -----------------------------------------------------------------

    def plot_training_curves(self) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(self.history["train_steps"], self.history["train_loss"], label="train")
        plt.plot(self.history["eval_steps"], self.history["eval_loss"], label="eval")
        plt.legend()
        plt.title("Train vs Eval Loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.show()

    def _load_sequence(self, parquet_path: str, seq_ix: int) -> pa.Table:
        stream = SequenceStream(parquet_path, self.data_cfg)
        for seq in stream.iter_sequences():
            if int(seq[self.data_cfg.seq_col][0]) == seq_ix:
                arrays = [pa.array(seq[c]) for c in stream.columns]
                return pa.table({c: arr for c, arr in zip(stream.columns, arrays)})
        raise IndexError(f"Sequence {seq_ix} not found in {parquet_path}")

    @torch.no_grad()
    def plot_sequence_predictions(self, seq_ix: int, max_steps: Optional[int] = None) -> None:
        import matplotlib.pyplot as plt

        seq = self._load_sequence(self.data_cfg.valid_path, seq_ix)
        if max_steps is not None:
            seq = seq[:max_steps]

        t_vals = seq.get_column(self.data_cfg.step_col).to_numpy()
        real = seq.select(list(self.data_cfg.target_cols)).to_numpy()
        need_pred = seq.get_column(self.data_cfg.need_pred_col).to_numpy().astype(bool)

        x_cov = np.stack([seq[c].to_numpy() for c in self.feature_cols], axis=1).astype(np.float32)
        y = real.astype(np.float32)
        step = t_vals.astype(np.float32)[:, None] / max(self.data_cfg.seq_len - 1, 1)
        x_cov = self.feature_scaler.transform(x_cov).astype(np.float32)
        y_scaled = self.target_scaler.transform(y).astype(np.float32)
        lag = np.zeros_like(y_scaled, dtype=np.float32)
        if self.data_cfg.use_lags:
            lag[1:] = y_scaled[:-1]
            lag[0] = y_scaled[0]
        x = np.concatenate([x_cov, step, lag], axis=1)

        x_t = torch.from_numpy(x).unsqueeze(0).to(self.train_cfg.device)
        lag_t = torch.from_numpy(lag).unsqueeze(0).to(self.train_cfg.device)
        self.model.eval()
        pred_delta_t0, pred_t1 = self.model(x_t)
        pred_scaled = torch.cat([pred_delta_t0 + lag_t[..., 0:1], pred_t1], dim=-1).cpu().numpy()[0]
        pred = self._denorm_target(pred_scaled, self.target_scaler)

        real_t0 = real[:, 0]
        real_t1 = real[:, 1]
        pred_t0 = pred[:, 0]
        pred_t1 = pred[:, 1]

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

        err_t0 = real_t0 - pred_t0
        err_t1 = real_t1 - pred_t1

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        axes[0].plot(t_vals, err_t0)
        axes[0].set_title("Error t0 en el tiempo")
        axes[1].plot(t_vals, err_t1)
        axes[1].set_title("Error t1 en el tiempo")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.hist(err_t0[need_pred], bins=50, alpha=0.6, label="t0")
        plt.hist(err_t1[need_pred], bins=50, alpha=0.6, label="t1")
        plt.title("Distribución de errores")
        plt.legend()
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(real_t0[need_pred], pred_t0[need_pred], alpha=0.5)
        axes[0].set_title("t0: real vs pred")
        axes[0].set_xlabel("real")
        axes[0].set_ylabel("pred")

        axes[1].scatter(real_t1[need_pred], pred_t1[need_pred], alpha=0.5)
        axes[1].set_title("t1: real vs pred")
        axes[1].set_xlabel("real")
        axes[1].set_ylabel("pred")
        plt.tight_layout()
        plt.show()

    def plot_eval_diagnostics(self, max_points: int = 2000) -> None:
        """
        Visualización completa de evaluación:
        1) Pred vs Real (serie temporal)
        2) Error en el tiempo
        3) Distribución de errores
        4) Scatter real vs pred

        max_points: limita puntos para no saturar plots
        """
        ev = self.evaluate()

        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Filtrar solo donde hay predicción
        y_true = ev.y_true[ev.mask]
        y_pred = ev.y_pred[ev.mask]

        # Subsample si es muy grande (clave para 1M puntos)
        if len(y_true) > max_points:
            idx = np.random.choice(len(y_true), max_points, replace=False)
            y_true = y_true[idx]
            y_pred = y_pred[idx]

        real_t0, real_t1 = y_true[:, 0], y_true[:, 1]
        pred_t0, pred_t1 = y_pred[:, 0], y_pred[:, 1]

        # =========================
        # 1. PRED VS REAL (TIEMPO)
        # =========================
        t_vals = np.arange(len(real_t0))

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(t_vals, real_t0, label="real t0")
        axes[0].plot(t_vals, pred_t0, label="pred t0")
        axes[0].set_title("t0 — Predicción vs Real (eval)")
        axes[0].legend()

        axes[1].plot(t_vals, real_t1, label="real t1")
        axes[1].plot(t_vals, pred_t1, label="pred t1")
        axes[1].set_title("t1 — Predicción vs Real (eval)")
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
        # 3. HISTOGRAMA ERRORES
        # =========================
        plt.figure(figsize=(10, 4))
        sns.histplot(err_t0, kde=True, label="t0", bins=50, color="C0")
        sns.histplot(err_t1, kde=True, label="t1", bins=50, color="C1")
        plt.legend()
        plt.title("Distribución de errores")
        plt.tight_layout()
        plt.show()

        # =========================
        # 4. SCATTER REAL VS PRED
        # =========================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(real_t0, pred_t0, alpha=0.4, color="C0")
        axes[0].set_title("t0: real vs pred")
        axes[0].set_xlabel("real")
        axes[0].set_ylabel("pred")

        axes[1].scatter(real_t1, pred_t1, alpha=0.4, color="C1")
        axes[1].set_title("t1: real vs pred")
        axes[1].set_xlabel("real")
        axes[1].set_ylabel("pred")

        plt.tight_layout()
        plt.show()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence, Any

import polars as pl
import numpy as np


@dataclass
class SequencePanelData:
    """
    Wrapper para datos secuenciales en formato panel.

    Columnas esperadas:
      - seq_col: identificador de secuencia
      - step_col: timestamp/índice temporal dentro de la secuencia
      - warmup_col: 0/1 indicando warm-up vs predicción
      - target_cols: variables a predecir
      - el resto: features exógenas / correlacionadas
    """

    df: pl.DataFrame
    seq_col: str = "seq_ix"
    step_col: str = "step_in_seq"
    warmup_col: str = "need_prediction"
    target_cols: tuple[str, ...] = ("t0", "t1")
    strict_length: bool = True
    expected_seq_len: Optional[int] = None
    ignore_cols: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._validate_schema()
        self._feature_cols = self._infer_feature_cols()

    def _validate_schema(self) -> None:
        required = {self.seq_col, self.step_col, self.warmup_col, *self.target_cols}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

        if self.df.height == 0:
            raise ValueError("El DataFrame está vacío.")

        if self.strict_length:
            lengths = self.df.group_by(self.seq_col).len().get_column("len").to_list()
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"Las secuencias no tienen la misma longitud: {sorted(set(lengths))}"
                )
            if (
                self.expected_seq_len is not None
                and lengths[0] != self.expected_seq_len
            ):
                raise ValueError(
                    f"Longitud esperada={self.expected_seq_len}, pero encontrada={lengths[0]}"
                )

        # Orden consistente para evitar sorpresas en slicing / plotting
        self.df = self.df.sort([self.seq_col, self.step_col])

    def _infer_feature_cols(self) -> tuple[str, ...]:
        excluded = {
            self.seq_col,
            self.step_col,
            self.warmup_col,
            *self.target_cols,
            *self.ignore_cols,
        }
        return tuple(c for c in self.df.columns if c not in excluded)

    @property
    def feature_cols(self) -> tuple[str, ...]:
        return self._feature_cols

    @property
    def sequence_ids(self) -> list[int]:
        return (
            self.df.select(self.seq_col)
            .unique()
            .sort(self.seq_col)
            .get_column(self.seq_col)
            .to_list()
        )

    @property
    def n_sequences(self) -> int:
        return len(self.sequence_ids)

    @property
    def seq_len(self) -> int:
        return self.df.group_by(self.seq_col).len().get_column("len")[0]

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def n_targets(self) -> int:
        return len(self.target_cols)

    def overview(self) -> pl.DataFrame:
        """
        Resumen por secuencia: longitud, nº de warm-up y nº de pasos de predicción.
        """
        return (
            self.df.group_by(self.seq_col)
            .agg(
                [
                    pl.len().alias("n_rows"),
                    pl.col(self.warmup_col).sum().alias("n_pred_steps"),
                    (pl.len() - pl.col(self.warmup_col).sum()).alias("n_warmup_steps"),
                    pl.col(self.step_col).min().alias("step_min"),
                    pl.col(self.step_col).max().alias("step_max"),
                ]
            )
            .sort(self.seq_col)
        )

    def sequence_df(
        self, seq_ix: int, cols: Optional[Sequence[str]] = None
    ) -> pl.DataFrame:
        """
        Devuelve una secuencia concreta ordenada por tiempo.
        """
        out = self.df.filter(pl.col(self.seq_col) == seq_ix).sort(self.step_col)
        if cols is not None:
            out = out.select(list(cols))
        return out

    def warmup_df(self, seq_ix: Optional[int] = None) -> pl.DataFrame:
        """
        Devuelve solo los pasos de warm-up.
        """
        base = self.df if seq_ix is None else self.sequence_df(seq_ix)
        return base.filter(pl.col(self.warmup_col) == 0)

    def prediction_df(self, seq_ix: Optional[int] = None) -> pl.DataFrame:
        """
        Devuelve solo los pasos de predicción.
        """
        base = self.df if seq_ix is None else self.sequence_df(seq_ix)
        return base.filter(pl.col(self.warmup_col) == 1)

    def to_long(
        self,
        seq_ix: Optional[int] = None,
        include_targets: bool = True,
        include_features: bool = True,
    ) -> pl.DataFrame:
        """
        Formato largo para EDA / plotting:
        [seq_ix, step_in_seq, need_prediction, variable, value]
        """
        base = self.df if seq_ix is None else self.sequence_df(seq_ix)

        value_vars: list[str] = []
        if include_features:
            value_vars.extend(self.feature_cols)
        if include_targets:
            value_vars.extend(self.target_cols)

        return base.melt(
            id_vars=[self.seq_col, self.step_col, self.warmup_col],
            value_vars=value_vars,
            variable_name="variable",
            value_name="value",
        )

    def to_numpy(
        self,
        seq_ix: int,
        include_targets_in_x: bool = False,
        return_mask: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Devuelve arrays numpy para una secuencia:
          X: [T, F]
          y: [T, n_targets]
          mask: [T] con 1 en predicción y 0 en warm-up
        """
        seq = self.sequence_df(seq_ix)

        x_cols = list(self.feature_cols)
        if include_targets_in_x:
            x_cols = x_cols + list(self.target_cols)

        X = seq.select(x_cols).to_numpy()
        y = seq.select(list(self.target_cols)).to_numpy()
        mask = seq.get_column(self.warmup_col).to_numpy().astype(bool)

        out = {"X": X, "y": y}
        if return_mask:
            out["mask"] = mask
            out["X_warmup"] = X[~mask]
            out["y_warmup"] = y[~mask]
            out["X_pred"] = X[mask]
            out["y_pred"] = y[mask]
        return out

    def stack_all_sequences(
        self,
        include_targets_in_x: bool = False,
        return_mask: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Devuelve tensores densos:
          X: [N, T, F]
          y: [N, T, n_targets]
          mask: [N, T]
        """
        xs, ys, masks = [], [], []
        for seq_ix in self.sequence_ids:
            arr = self.to_numpy(
                seq_ix=seq_ix,
                include_targets_in_x=include_targets_in_x,
                return_mask=return_mask,
            )
            xs.append(arr["X"])
            ys.append(arr["y"])
            if return_mask:
                masks.append(arr["mask"])

        out = {
            "X": np.stack(xs, axis=0),
            "y": np.stack(ys, axis=0),
        }
        if return_mask:
            out["mask"] = np.stack(masks, axis=0)
        return out

    def to_sarimax_frame(
        self,
        seq_ix: int,
        target: str = "t0",
        exog_cols: Optional[Sequence[str]] = None,
    ) -> pl.DataFrame:
        """
        Frame listo para modelado clásico de una sola serie (p.ej. SARIMAX).
        Devuelve columnas:
          step_in_seq, target, exog...
        """
        if target not in self.target_cols:
            raise ValueError(
                f"target='{target}' no está en target_cols={self.target_cols}"
            )

        if exog_cols is None:
            exog_cols = self.feature_cols

        cols = [self.step_col, target, *exog_cols]
        return self.sequence_df(seq_ix, cols=cols)

    def iter_sequences(self) -> Iterator[tuple[int, pl.DataFrame]]:
        """
        Iterador de (seq_ix, dataframe_de_secuencia).
        """
        for seq_ix in self.sequence_ids:
            yield seq_ix, self.sequence_df(seq_ix)

    def get_sequence_bundle(self, seq_ix: int) -> dict[str, Any]:
        """
        Devuelve un bundle práctico para EDA / modelado.
        """
        seq = self.sequence_df(seq_ix)
        arr = self.to_numpy(seq_ix)

        return {
            "seq_ix": seq_ix,
            "df": seq,
            "X": arr["X"],
            "y": arr["y"],
            "mask": arr["mask"],
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols,
        }

    def plot_sequence(
        self,
        seq_ix: int,
        target: str = "t0",
        feature_cols: Optional[Sequence[str]] = None,
        figsize: tuple[int, int] = (14, 6),
    ) -> None:
        """
        Plot simple para inspección rápida.
        """
        import matplotlib.pyplot as plt

        if target not in self.target_cols:
            raise ValueError(
                f"target='{target}' no está en target_cols={self.target_cols}"
            )

        seq = self.sequence_df(seq_ix)
        feature_cols = (
            list(feature_cols)
            if feature_cols is not None
            else list(self.feature_cols[::6])
        )

        fig, ax1 = plt.subplots(figsize=figsize)

        x = seq.get_column(self.step_col).to_numpy()
        y = seq.get_column(target).to_numpy()
        ax1.plot(x, y, label=target)
        ax1.set_xlabel(self.step_col)
        ax1.set_ylabel(target)

        for c in feature_cols:
            ax1.plot(x, seq.get_column(c).to_numpy(), alpha=0.5, label=c)

        warmup_mask = seq.get_column(self.warmup_col).to_numpy().astype(bool)
        if warmup_mask.any():
            first_pred = np.argmax(warmup_mask)
            ax1.axvline(
                x[first_pred], linestyle="--", alpha=0.7, label="inicio predicción"
            )

        ax1.legend(loc="best")
        plt.tight_layout()
        plt.show()


try:
    import torch
    from torch.utils.data import Dataset

    class SequenceTorchDataset(Dataset):
        """
        Dataset para entrenamiento deep learning.
        Devuelve secuencias completas.
        """

        def __init__(
            self,
            panel: SequencePanelData,
            seq_ids: Optional[Sequence[int]] = None,
            include_targets_in_x: bool = False,
        ):
            self.panel = panel
            self.seq_ids = list(seq_ids) if seq_ids is not None else panel.sequence_ids
            self.include_targets_in_x = include_targets_in_x

        def __len__(self) -> int:
            return len(self.seq_ids)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            seq_ix = self.seq_ids[idx]
            arr = self.panel.to_numpy(
                seq_ix=seq_ix,
                include_targets_in_x=self.include_targets_in_x,
                return_mask=True,
            )
            return {
                "seq_ix": torch.tensor(seq_ix, dtype=torch.long),
                "X": torch.tensor(arr["X"], dtype=torch.float32),
                "y": torch.tensor(arr["y"], dtype=torch.float32),
                "mask": torch.tensor(arr["mask"], dtype=torch.bool),
            }

except Exception:
    SequenceTorchDataset = None

#!/usr/bin/env python
"""OhioT1DM clinical-event evaluation.

This runner follows the Sequential Transformer comparison track:
  - PH=30 min main table (5-minute sampling)
  - observed window = 120 min (24 steps)
  - OhioT1DM: official per-subject `ws-training` / `ws-testing`, with the tail
    of the official training split held out for validation
  - metrics: RMSE, Time Gain, Hyper Sensitivity, Hypo Sensitivity

The implementation reuses the validation-selected anchor/correction path from
the GlucoBench runner. Official DiaTrend data are controlled-access workbooks;
the earlier research workspace accidentally used the unrelated DiaData
integration under that name, so this public runner intentionally does not
claim support for DiaTrend until a verified adapter is available.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def _configure_line_buffered_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


_configure_line_buffered_output()

import torch

from glucoflow.data.adapters import OhioT1DMAdapter
from evaluate_glucobench import (
    ArrayStandardizer,
    CorrectionMLP,
    DEVICE,
    correction_loss_value,
    fit_anchor,
    median_rmse_mae,
    metric_priority_tuple,
    persistence_forecast,
    select_anchor_mix,
    build_event_loss_weights,
    CorrectionDataset,
)
from torch.utils.data import DataLoader


class CorrectionMLPWithEventHead(torch.nn.Module):
    """Correction MLP with auxiliary binary event classification head."""

    def __init__(self, d_in: int, d_out: int, d_model: int, dropout: float):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.correction_head = torch.nn.Linear(d_model, d_out)
        self.event_head = torch.nn.Linear(d_model, 2)  # [hyper_logit, hypo_logit]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        return self.correction_head(features)

    def forward_with_events(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.correction_head(features), self.event_head(features)


class TemporalConvCorrectionBackbone(torch.nn.Module):
    """Sequence-aware correction backbone for Table C.

    It preserves the temporal structure of the history window instead of
    forcing the entire correction input through a flat MLP from the first
    layer onward.
    """

    def __init__(self, history_len: int, anchor_len: int, temporal_len: int, d_model: int, dropout: float):
        super().__init__()
        self.history_len = int(history_len)
        self.anchor_len = int(anchor_len)
        self.temporal_len = int(temporal_len)

        self.history_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.history_proj = torch.nn.Linear(d_model * self.history_len, d_model)
        self.anchor_proj = torch.nn.Sequential(
            torch.nn.Linear(self.anchor_len, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.temporal_proj = None
        fusion_in = d_model * 2
        if self.temporal_len > 0:
            self.temporal_proj = torch.nn.Sequential(
                torch.nn.Linear(self.temporal_len, d_model),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            fusion_in += d_model
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(fusion_in, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        history = x[:, : self.history_len]
        anchor = x[:, self.history_len : self.history_len + self.anchor_len]
        temporal = None
        if self.temporal_len > 0:
            temporal = x[:, self.history_len + self.anchor_len :]
        return history, anchor, temporal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history, anchor, temporal = self._split(x)
        hist_feat = self.history_encoder(history.unsqueeze(1)).flatten(1)
        hist_feat = self.history_proj(hist_feat)
        parts = [hist_feat, self.anchor_proj(anchor)]
        if self.temporal_proj is not None and temporal is not None:
            parts.append(self.temporal_proj(temporal))
        return self.fuse(torch.cat(parts, dim=1))


class TemporalConvCorrectionModel(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, d_model: int, dropout: float, history_len: int, anchor_len: int):
        super().__init__()
        temporal_len = max(int(d_in) - int(history_len) - int(anchor_len), 0)
        self.backbone = TemporalConvCorrectionBackbone(
            history_len=history_len,
            anchor_len=anchor_len,
            temporal_len=temporal_len,
            d_model=d_model,
            dropout=dropout,
        )
        self.correction_head = torch.nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.correction_head(self.backbone(x))


class TemporalConvCorrectionWithEventHead(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, d_model: int, dropout: float, history_len: int, anchor_len: int):
        super().__init__()
        temporal_len = max(int(d_in) - int(history_len) - int(anchor_len), 0)
        self.backbone = TemporalConvCorrectionBackbone(
            history_len=history_len,
            anchor_len=anchor_len,
            temporal_len=temporal_len,
            d_model=d_model,
            dropout=dropout,
        )
        self.correction_head = torch.nn.Linear(d_model, d_out)
        self.event_head = torch.nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.correction_head(self.backbone(x))

    def forward_with_events(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.correction_head(features), self.event_head(features)


class EventCorrectionDataset(torch.utils.data.Dataset):
    """Correction dataset that carries window-level hyper/hypo labels."""

    def __init__(
        self,
        cond: np.ndarray,
        target: np.ndarray,
        event_target: np.ndarray,
        weight: np.ndarray | None = None,
    ):
        self.cond = cond.astype(np.float32)
        self.target = target.astype(np.float32)
        self.event_target = event_target.astype(np.float32)
        self.weight = None if weight is None else weight.astype(np.float32)

    def __len__(self) -> int:
        return len(self.cond)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "cond": torch.from_numpy(self.cond[idx]),
            "target": torch.from_numpy(self.target[idx]),
            "event_target": torch.from_numpy(self.event_target[idx]),
        }
        if self.weight is not None:
            item["weight"] = torch.from_numpy(self.weight[idx])
        return item


def _window_slope(values: np.ndarray) -> np.ndarray:
    x = np.arange(values.shape[1], dtype=np.float32)
    x_centered = x - float(x.mean())
    denom = float(np.square(x_centered).sum())
    centered = values.astype(np.float32) - values.astype(np.float32).mean(axis=1, keepdims=True)
    return (centered * x_centered[None, :]).sum(axis=1) / max(denom, 1e-6)


def build_temporal_feature_block(history_raw: np.ndarray) -> np.ndarray:
    recent6 = history_raw[:, -6:].astype(np.float32)
    recent12 = history_raw[:, -12:].astype(np.float32)
    recent_diffs = np.diff(recent6, axis=1).astype(np.float32)
    std6 = recent6.std(axis=1, keepdims=True).astype(np.float32)
    std12 = recent12.std(axis=1, keepdims=True).astype(np.float32)
    slope6 = _window_slope(recent6).reshape(-1, 1).astype(np.float32)
    slope12 = _window_slope(recent12).reshape(-1, 1).astype(np.float32)
    accel3 = (history_raw[:, -1] - 2.0 * history_raw[:, -2] + history_raw[:, -3]).reshape(-1, 1).astype(np.float32)
    range12 = (recent12.max(axis=1) - recent12.min(axis=1)).reshape(-1, 1).astype(np.float32)
    return np.concatenate(
        [recent_diffs, std6, std12, slope6, slope12, accel3, range12],
        axis=1,
    ).astype(np.float32)


def build_tablec_condition_artifacts(
    train_hist: np.ndarray,
    train_anchor: np.ndarray,
    feature_set: str,
) -> dict:
    artifacts = {
        "feature_set": feature_set,
        "history_scaler": ArrayStandardizer.fit(train_hist),
        "anchor_scaler": ArrayStandardizer.fit(train_anchor),
        "temporal_scaler": None,
    }
    if feature_set == "temporal_v1":
        artifacts["temporal_scaler"] = ArrayStandardizer.fit(build_temporal_feature_block(train_hist))
    return artifacts


def build_tablec_correction_inputs(
    history_raw: np.ndarray,
    anchor_raw: np.ndarray,
    artifacts: dict,
) -> np.ndarray:
    parts = [
        artifacts["history_scaler"].transform(history_raw),
        artifacts["anchor_scaler"].transform(anchor_raw),
    ]
    if artifacts["feature_set"] == "temporal_v1":
        temporal_scaler = artifacts.get("temporal_scaler")
        if temporal_scaler is None:
            raise ValueError("temporal_v1 feature_set requires a temporal_scaler")
        parts.append(temporal_scaler.transform(build_temporal_feature_block(history_raw)))
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_tablec_event_labels(future_glucose: np.ndarray) -> np.ndarray:
    has_hyper = (np.max(future_glucose, axis=1) >= HYPER_THRESHOLD).astype(np.float32)
    has_hypo = (np.min(future_glucose, axis=1) <= HYPO_THRESHOLD).astype(np.float32)
    return np.stack([has_hyper, has_hypo], axis=1).astype(np.float32)


def build_event_head_pos_weight(event_labels: np.ndarray, max_weight: float = 20.0) -> np.ndarray:
    if len(event_labels) == 0:
        return np.ones((2,), dtype=np.float32)
    positives = event_labels.sum(axis=0).astype(np.float32)
    negatives = float(len(event_labels)) - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    return np.clip(pos_weight.astype(np.float32), 1.0, max(max_weight, 1.0))


def event_head_loss_value(
    event_logits: torch.Tensor,
    event_target: torch.Tensor,
    loss_name: str,
    pos_weight: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        event_logits,
        event_target,
        reduction="none",
        pos_weight=pos_weight,
    )
    if loss_name != "focal":
        return bce.mean()
    probs = torch.sigmoid(event_logits)
    p_t = probs * event_target + (1.0 - probs) * (1.0 - event_target)
    focal_factor = torch.pow(torch.clamp(1.0 - p_t, min=0.0), float(focal_gamma))
    return (bce * focal_factor).mean()


def quantile_event_aux_loss(
    correction_pred: torch.Tensor,
    target: torch.Tensor,
    event_target: torch.Tensor,
    hyper_quantile: float = 0.8,
    hypo_quantile: float = 0.2,
) -> torch.Tensor:
    error = target - correction_pred
    losses: list[torch.Tensor] = []

    hyper_mask = event_target[:, 0] > 0.5
    if bool(hyper_mask.any().item()):
        hyper_error = error[hyper_mask]
        losses.append(torch.maximum(hyper_quantile * hyper_error, (hyper_quantile - 1.0) * hyper_error).mean())

    hypo_mask = event_target[:, 1] > 0.5
    if bool(hypo_mask.any().item()):
        hypo_error = error[hypo_mask]
        losses.append(torch.maximum(hypo_quantile * hypo_error, (hypo_quantile - 1.0) * hypo_error).mean())

    if not losses:
        return correction_pred.new_zeros(())
    return torch.stack(losses).mean()


def build_tablec_model(
    d_in: int,
    d_out: int,
    d_model: int,
    dropout: float,
    correction_arch: str,
    use_event_head: bool,
) -> torch.nn.Module:
    if correction_arch == "temporal_conv":
        if use_event_head:
            return TemporalConvCorrectionWithEventHead(
                d_in=d_in,
                d_out=d_out,
                d_model=d_model,
                dropout=dropout,
                history_len=IN_LEN,
                anchor_len=OUT_LEN,
            )
        return TemporalConvCorrectionModel(
            d_in=d_in,
            d_out=d_out,
            d_model=d_model,
            dropout=dropout,
            history_len=IN_LEN,
            anchor_len=OUT_LEN,
        )
    if use_event_head:
        return CorrectionMLPWithEventHead(
            d_in=d_in,
            d_out=d_out,
            d_model=d_model,
            dropout=dropout,
        )
    return CorrectionMLP(
        d_in=d_in,
        d_out=d_out,
        d_model=d_model,
        dropout=dropout,
    )


def correction_checkpoint_tuple(val_fut: np.ndarray, point_pred: np.ndarray, args) -> tuple[float, ...]:
    baseline = getattr(args, "_dataset_baseline", None)
    priority = getattr(args, "table_selection_priority", "rmse_first")
    if priority == "adaptive":
        priority = DATASET_PRIORITY_OVERRIDES.get(getattr(args, "_dataset_name", ""), "baseline_gap")
    if baseline is not None and priority in {"baseline_gap", "acceptance_first"}:
        return tablec_priority_tuple(compute_tablec_metrics(val_fut, point_pred), priority, baseline=baseline)
    val_rmse, val_mae = median_rmse_mae(val_fut, point_pred)
    return metric_priority_tuple(val_rmse, val_mae, args.selection_priority)


def train_tablec_correction_model(
    train_hist,
    train_fut,
    train_anchor,
    val_hist,
    val_fut,
    val_anchor,
    args,
    use_event_head: bool = False,
):
    condition_artifacts = build_tablec_condition_artifacts(
        train_hist=train_hist,
        train_anchor=train_anchor,
        feature_set=getattr(args, "temporal_feature_set", "baseline"),
    )
    residual_scaler = ArrayStandardizer.fit(train_fut - train_anchor)

    train_cond = build_tablec_correction_inputs(train_hist, train_anchor, condition_artifacts)
    val_cond = build_tablec_correction_inputs(val_hist, val_anchor, condition_artifacts)
    train_target = residual_scaler.transform(train_fut - train_anchor)
    need_event_targets = bool(use_event_head or getattr(args, "quantile_event_loss", False))
    train_event_target = build_tablec_event_labels(train_fut) if need_event_targets else None
    train_weight = getattr(args, "_balanced_weights", None) if getattr(args, "_use_balanced", False) else build_event_loss_weights(
        train_fut,
        high_weight=getattr(args, "event_loss_high_weight", 1.0),
        low_weight=getattr(args, "event_loss_low_weight", 1.0),
    )

    if need_event_targets:
        train_loader = DataLoader(
            EventCorrectionDataset(
                train_cond,
                train_target,
                event_target=train_event_target,
                weight=train_weight,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    else:
        train_loader = DataLoader(
            CorrectionDataset(train_cond, train_target, weight=train_weight),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    if use_event_head:
        model = build_tablec_model(
            d_in=train_cond.shape[1],
            d_out=train_target.shape[1],
            d_model=args.d_model,
            dropout=args.dropout,
            correction_arch=getattr(args, "correction_arch", "mlp"),
            use_event_head=True,
        ).to(DEVICE)
        event_weight = float(getattr(args, "event_head_weight", 0.5))
        event_loss_name = getattr(args, "event_head_loss", "balanced_bce")
        event_pos_weight = None
        if event_loss_name in {"balanced_bce", "focal"}:
            event_pos_weight = torch.tensor(
                build_event_head_pos_weight(
                    train_event_target,
                    max_weight=float(getattr(args, "event_head_max_pos_weight", 20.0)),
                ),
                dtype=torch.float32,
                device=DEVICE,
            )
    else:
        model = build_tablec_model(
            d_in=train_cond.shape[1],
            d_out=train_target.shape[1],
            d_model=args.d_model,
            dropout=args.dropout,
            correction_arch=getattr(args, "correction_arch", "mlp"),
            use_event_head=False,
        ).to(DEVICE)
        event_weight = 0.0
        event_loss_name = "bce"
        event_pos_weight = None

    correction_lr = args.correction_lr if args.correction_lr > 0 else args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=correction_lr, weight_decay=args.weight_decay)
    scheduler = None
    if getattr(args, "cosine_schedule", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=correction_lr * 0.01,
        )

    best_state = None
    best_metric = None
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        corr_losses = []
        event_losses = []
        quantile_losses = []
        for batch in train_loader:
            cond = batch["cond"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            weight = batch.get("weight")
            if weight is not None:
                weight = weight.to(DEVICE)
            event_target = batch.get("event_target")
            if event_target is not None:
                event_target = event_target.to(DEVICE)
            if use_event_head:
                correction_pred, event_logits = model.forward_with_events(cond)
            else:
                correction_pred = model(cond)
                event_logits = None

            corr_loss = correction_loss_value(
                correction_pred,
                target,
                loss_name=args.correction_loss,
                huber_delta=float(args.correction_huber_delta),
                weight=weight,
            )
            loss = corr_loss
            corr_losses.append(float(corr_loss.item()))

            if use_event_head and event_logits is not None and event_target is not None:
                evt_loss = event_head_loss_value(
                    event_logits,
                    event_target,
                    loss_name=event_loss_name,
                    pos_weight=event_pos_weight,
                    focal_gamma=float(getattr(args, "event_head_focal_gamma", 2.0)),
                )
                loss = loss + event_weight * evt_loss
                event_losses.append(float(evt_loss.item()))

            if getattr(args, "quantile_event_loss", False) and event_target is not None:
                q_loss = quantile_event_aux_loss(
                    correction_pred,
                    target,
                    event_target,
                    hyper_quantile=float(getattr(args, "quantile_hyper", 0.8)),
                    hypo_quantile=float(getattr(args, "quantile_hypo", 0.2)),
                )
                loss = loss + float(getattr(args, "quantile_event_loss_weight", 0.15)) * q_loss
                quantile_losses.append(float(q_loss.item()))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(val_cond, dtype=torch.float32, device=DEVICE)
            if use_event_head:
                val_pred, _ = model.forward_with_events(val_tensor)
            else:
                val_pred = model(val_tensor)
        point_pred = val_anchor + residual_scaler.inverse(val_pred.cpu().numpy())
        val_rmse, val_mae = median_rmse_mae(val_fut, point_pred)
        current_metric = correction_checkpoint_tuple(val_fut, point_pred, args)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "corr_loss": float(np.mean(corr_losses)) if corr_losses else 0.0,
                "event_loss": float(np.mean(event_losses)) if event_losses else 0.0,
                "quantile_loss": float(np.mean(quantile_losses)) if quantile_losses else 0.0,
                "val_rmse": float(val_rmse),
                "val_mae": float(val_mae),
            }
        )
        print(
            f"[{getattr(args, '_dataset_name', 'tablec')}] epoch {epoch}/{args.epochs} "
            f"loss={history[-1]['train_loss']:.4f} corr={history[-1]['corr_loss']:.4f} "
            f"evt={history[-1]['event_loss']:.4f} q={history[-1]['quantile_loss']:.4f} "
            f"val_rmse={val_rmse:.4f} val_mae={val_mae:.4f}"
        )
        if best_metric is None or current_metric < best_metric:
            best_metric = current_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        raise RuntimeError("Table C correction model training failed")
    model.load_state_dict(best_state)
    return model, {
        **condition_artifacts,
        "residual_scaler": residual_scaler,
        "best_epoch": int(best_epoch),
        "val_rmse": float(history[best_epoch - 1]["val_rmse"]),
        "val_mae": float(history[best_epoch - 1]["val_mae"]),
        "train_history": history,
        "uses_event_head": bool(use_event_head),
        "correction_arch": getattr(args, "correction_arch", "mlp"),
    }


def train_correction_model_with_events(
    train_hist,
    train_fut,
    train_anchor,
    val_hist,
    val_fut,
    val_anchor,
    args,
):
    return train_tablec_correction_model(
        train_hist,
        train_fut,
        train_anchor,
        val_hist,
        val_fut,
        val_anchor,
        args,
        use_event_head=True,
    )


@torch.no_grad()
def predict_tablec_correction_point(
    model: torch.nn.Module,
    history_raw: np.ndarray,
    anchor_raw: np.ndarray,
    artifacts: dict,
    batch_size: int,
    return_event_probs: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    cond = build_tablec_correction_inputs(history_raw, anchor_raw, artifacts)
    residual_preds = []
    event_probs = [] if return_event_probs and artifacts.get("uses_event_head", False) else None
    model.eval()
    for start in range(0, len(cond), batch_size):
        stop = min(start + batch_size, len(cond))
        batch_cond = torch.tensor(cond[start:stop], dtype=torch.float32, device=DEVICE)
        if event_probs is not None and hasattr(model, "forward_with_events"):
            residual_pred, event_logits = model.forward_with_events(batch_cond)
            event_probs.append(torch.sigmoid(event_logits).cpu().numpy())
        else:
            residual_pred = model(batch_cond)
        residual_preds.append(residual_pred.cpu().numpy())
    residual_pred = np.concatenate(residual_preds, axis=0)
    point_pred = anchor_raw + artifacts["residual_scaler"].inverse(residual_pred)
    probs_out = None if event_probs is None else np.concatenate(event_probs, axis=0).astype(np.float32)
    return point_pred.astype(np.float32), probs_out

RUN_DIR = ROOT / "artifacts" / "metrics" / "phase5_tablec_runs"
TABLE_PATH = ROOT / "artifacts" / "tables" / "main" / "ohio_t1dm_events.csv"
SUMMARY_PATH = ROOT / "artifacts" / "run_summaries" / "ohio_t1dm_events.md"
FIG_PATH = ROOT / "artifacts" / "figures" / "main" / "fig_tablec_rmse.png"

PH_MINUTES = 30
STEP_MINUTES = 5
IN_LEN = 24
OUT_LEN = 6
TRAIN_FRAC = 0.64
VAL_FRAC = 0.16
OHIO_TRAIN_FRAC = 0.8
HYPER_THRESHOLD = 180.0
HYPO_THRESHOLD = 70.0

BASELINES = {
    "ohio_t1dm": {"display": "OhioT1DM", "rmse": 14.96, "tg": 17.56, "hyper": 96.26, "hypo": 75.82},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["run", "aggregate"], default="run")
    parser.add_argument("--dataset", choices=["ohio_t1dm"], default="ohio_t1dm")
    parser.add_argument("--config-name", type=str, default="anchor_mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--correction-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--correction-arch", choices=["mlp", "temporal_conv"], default="mlp")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--correction-loss", choices=["mse", "huber", "l1"], default="mse")
    parser.add_argument("--correction-huber-delta", type=float, default=1.0)
    parser.add_argument("--event-loss-high-weight", type=float, default=1.0)
    parser.add_argument("--event-loss-low-weight", type=float, default=1.0)
    parser.add_argument("--selection-priority", choices=["rmse_mae", "mae_rmse"], default="rmse_mae")
    parser.add_argument("--anchor-alpha", type=float, default=1.0)
    parser.add_argument("--anchor-mix-grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument(
        "--table-selection-priority",
        choices=[
            "rmse_first",
            "event_first",
            "hyper_first",
            "baseline_gap",
            "acceptance_first",
            "hyper_weighted",
            "adaptive",
        ],
        default="event_first",
    )
    parser.add_argument("--event-rmse-tolerance", type=float, default=0.15)
    parser.add_argument("--event-up-scale-grid", type=str, default="0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6")
    parser.add_argument("--event-down-scale-grid", type=str, default="0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6")
    parser.add_argument("--event-prob-threshold-grid", type=str, default="0.45,0.55,0.65")
    parser.add_argument("--event-threshold-margin", type=float, default=20.0)
    parser.add_argument("--max-train-windows", type=int, default=120000)
    parser.add_argument("--max-val-windows", type=int, default=24000)
    parser.add_argument("--sample-rows-per-file", type=int, default=0)
    parser.add_argument("--temporal-feature-set", choices=["baseline", "temporal_v1"], default="baseline")
    parser.add_argument(
        "--event-gated-scaling",
        action="store_true",
        help="Gate threshold-aware scaling with event-head probabilities when available.",
    )
    parser.add_argument("--use-event-head", action="store_true",
                        help="Add auxiliary hyper/hypo classification head (multi-task learning)")
    parser.add_argument("--event-head-weight", type=float, default=0.5,
                        help="Weight of event classification loss relative to correction loss")
    parser.add_argument(
        "--event-head-loss",
        choices=["bce", "balanced_bce", "focal"],
        default="balanced_bce",
        help="Loss family for the auxiliary event head.",
    )
    parser.add_argument("--event-head-focal-gamma", type=float, default=2.0,
                        help="Focal gamma when --event-head-loss=focal.")
    parser.add_argument("--event-head-max-pos-weight", type=float, default=20.0,
                        help="Cap for inverse-frequency positive class weights in the event head.")
    parser.add_argument("--smote-factor-hyper", type=int, default=0,
                        help="SMOTE oversampling factor for hyper windows (0=off)")
    parser.add_argument("--smote-factor-hypo", type=int, default=0,
                        help="SMOTE oversampling factor for hypo windows (0=off)")
    parser.add_argument("--smote-noise-std", type=float, default=1.5,
                        help="Gaussian noise std for SMOTE augmentation (mg/dL)")
    parser.add_argument("--balanced-loss", action="store_true",
                        help="Use paper-style balanced loss: hypo×3, hyper×2, normal downweighted")
    parser.add_argument("--quantile-event-loss", action="store_true",
                        help="Use asymmetric pinball loss near event thresholds")
    parser.add_argument("--quantile-event-loss-weight", type=float, default=0.15,
                        help="Weight of the asymmetric event-aware residual loss.")
    parser.add_argument("--quantile-hyper", type=float, default=0.8,
                        help="Quantile used for hyper windows in the auxiliary pinball loss.")
    parser.add_argument("--quantile-hypo", type=float, default=0.2,
                        help="Quantile used for hypo windows in the auxiliary pinball loss.")
    parser.add_argument("--cosine-schedule", action="store_true",
                        help="Use cosine annealing LR schedule instead of flat LR.")
    parser.add_argument("--seed-ensemble", type=str, default="",
                        help="Comma-separated seeds for prediction ensemble (trains N models, averages predictions).")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def smote_oversample(
    history: np.ndarray,
    future: np.ndarray,
    hyper_factor: int = 3,
    hypo_factor: int = 5,
    noise_std: float = 1.5,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE-style oversampling of event windows (Sequential Transformer paper technique).

    For each hyper/hypo window, create `factor-1` synthetic copies with small
    Gaussian noise added to both history and future glucose values.
    """
    if hyper_factor <= 1 and hypo_factor <= 1:
        return history, future
    if rng is None:
        rng = np.random.RandomState(42)

    has_hyper = np.max(future, axis=1) >= HYPER_THRESHOLD
    has_hypo = np.min(future, axis=1) <= HYPO_THRESHOLD

    parts_h, parts_f = [history], [future]
    if hyper_factor > 1:
        idx = np.where(has_hyper & ~has_hypo)[0]  # hyper-only windows
        for _ in range(hyper_factor - 1):
            noise_h = rng.randn(len(idx), history.shape[1]).astype(np.float32) * noise_std
            noise_f = rng.randn(len(idx), future.shape[1]).astype(np.float32) * noise_std
            parts_h.append(np.clip(history[idx] + noise_h, 40, 400).astype(np.float32))
            parts_f.append(np.clip(future[idx] + noise_f, 40, 400).astype(np.float32))
    if hypo_factor > 1:
        idx = np.where(has_hypo)[0]  # hypo windows (may overlap with hyper)
        for _ in range(hypo_factor - 1):
            noise_h = rng.randn(len(idx), history.shape[1]).astype(np.float32) * noise_std
            noise_f = rng.randn(len(idx), future.shape[1]).astype(np.float32) * noise_std
            parts_h.append(np.clip(history[idx] + noise_h, 40, 400).astype(np.float32))
            parts_f.append(np.clip(future[idx] + noise_f, 40, 400).astype(np.float32))
    return np.concatenate(parts_h, axis=0), np.concatenate(parts_f, axis=0)


def build_balanced_loss_weights(
    future_glucose: np.ndarray,
    hypo_relevance: float = 3.0,
    hyper_relevance: float = 2.0,
    normal_relevance: float = 1.0,
) -> np.ndarray:
    """Paper-style balanced loss: w = relevance × (1 - class_count/total).

    From Sequential Transformer (Barbato et al. 2025): weights inversely
    proportional to class frequency, scaled by clinical relevance.
    """
    n = len(future_glucose)
    has_hyper = np.max(future_glucose, axis=1) >= HYPER_THRESHOLD
    has_hypo = np.min(future_glucose, axis=1) <= HYPO_THRESHOLD
    is_normal = ~(has_hyper | has_hypo)

    n_hyper = max(int(has_hyper.sum()), 1)
    n_hypo = max(int(has_hypo.sum()), 1)
    n_normal = max(int(is_normal.sum()), 1)

    w_hyper = hyper_relevance * (1.0 - n_hyper / n)
    w_hypo = hypo_relevance * (1.0 - n_hypo / n)
    w_normal = normal_relevance * (1.0 - n_normal / n)

    # Per-window weights, broadcast to (N, OUT_LEN)
    weights = np.ones((n, future_glucose.shape[1]), dtype=np.float32) * w_normal
    weights[has_hyper] = w_hyper
    weights[has_hypo] = np.maximum(weights[has_hypo], w_hypo)
    return weights


def chronological_slices(n_rows: int) -> tuple[slice, slice, slice]:
    train_end = max(int(n_rows * TRAIN_FRAC), IN_LEN + OUT_LEN)
    val_end = max(int(n_rows * (TRAIN_FRAC + VAL_FRAC)), train_end + IN_LEN + OUT_LEN)
    train_end = min(train_end, max(n_rows - 2 * (IN_LEN + OUT_LEN), IN_LEN + OUT_LEN))
    val_end = min(val_end, max(n_rows - (IN_LEN + OUT_LEN), train_end + IN_LEN + OUT_LEN))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n_rows)


def train_val_slices(n_rows: int, train_frac: float) -> tuple[slice, slice]:
    train_end = max(int(n_rows * train_frac), IN_LEN + OUT_LEN)
    train_end = min(train_end, max(n_rows - (IN_LEN + OUT_LEN), IN_LEN + OUT_LEN))
    return slice(0, train_end), slice(train_end, n_rows)


def windows_from_series(glucose: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = IN_LEN + OUT_LEN
    if len(glucose) < total:
        return (
            np.empty((0, IN_LEN), dtype=np.float32),
            np.empty((0, OUT_LEN), dtype=np.float32),
            np.empty((0, IN_LEN), dtype="datetime64[s]"),
        )
    values = glucose.astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(values, total)
    ts_windows = np.lib.stride_tricks.sliding_window_view(timestamps.astype("datetime64[s]"), IN_LEN)
    ts_windows = ts_windows[: windows.shape[0]]
    valid = ~np.isnan(windows).any(axis=1)
    windows = windows[valid]
    ts_windows = ts_windows[valid]
    return windows[:, :IN_LEN].copy(), windows[:, IN_LEN:].copy(), ts_windows.copy()


def regularize_5min_series(timestamps: np.ndarray, glucose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    series = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "glucose_mgdl": glucose.astype(np.float32),
        }
    ).dropna(subset=["timestamp"])
    if series.empty:
        return np.empty((0,), dtype="datetime64[s]"), np.empty((0,), dtype=np.float32)
    series = series.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    full_index = pd.date_range(series["timestamp"].iloc[0], series["timestamp"].iloc[-1], freq="5min")
    regular = series.set_index("timestamp").reindex(full_index)
    return regular.index.to_numpy(dtype="datetime64[s]"), regular["glucose_mgdl"].to_numpy(dtype=np.float32)


def metric_time_gain_minutes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    gains = []
    for truth, pred in zip(y_true, y_pred):
        best_shift = 0
        best_mse = float("inf")
        for shift in range(OUT_LEN):
            overlap = OUT_LEN - shift
            mse = float(np.mean((truth[shift:] - pred[:overlap]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_shift = shift
        gains.append(PH_MINUTES - best_shift * STEP_MINUTES)
    return float(np.mean(gains)) if gains else 0.0


def event_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, kind: str) -> float:
    if kind == "hyper":
        true_event = np.max(y_true, axis=1) >= threshold
        pred_event = np.max(y_pred, axis=1) >= threshold
    else:
        true_event = np.min(y_true, axis=1) <= threshold
        pred_event = np.min(y_pred, axis=1) <= threshold
    positives = int(true_event.sum())
    if positives == 0:
        return float("nan")
    tp = int(np.logical_and(true_event, pred_event).sum())
    return 100.0 * tp / positives


def compute_tablec_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean(np.square(y_pred - y_true)))),
        "tg": metric_time_gain_minutes(y_true, y_pred),
        "hyper": event_sensitivity(y_true, y_pred, HYPER_THRESHOLD, "hyper"),
        "hypo": event_sensitivity(y_true, y_pred, HYPO_THRESHOLD, "hypo"),
    }


def update_tablec_aggregates(stats: dict, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    stats["sq_error_sum"] += float(np.square(y_pred - y_true).sum())
    stats["value_count"] += int(y_true.size)

    gains = []
    for truth, pred in zip(y_true, y_pred):
        best_shift = 0
        best_mse = float("inf")
        for shift in range(OUT_LEN):
            overlap = OUT_LEN - shift
            mse = float(np.mean((truth[shift:] - pred[:overlap]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_shift = shift
        gains.append(PH_MINUTES - best_shift * STEP_MINUTES)
    stats["tg_sum"] += float(np.sum(gains))
    stats["window_count"] += int(len(y_true))

    hyper_true = np.max(y_true, axis=1) >= HYPER_THRESHOLD
    hyper_pred = np.max(y_pred, axis=1) >= HYPER_THRESHOLD
    stats["hyper_positives"] += int(hyper_true.sum())
    stats["hyper_tp"] += int(np.logical_and(hyper_true, hyper_pred).sum())

    hypo_true = np.min(y_true, axis=1) <= HYPO_THRESHOLD
    hypo_pred = np.min(y_pred, axis=1) <= HYPO_THRESHOLD
    stats["hypo_positives"] += int(hypo_true.sum())
    stats["hypo_tp"] += int(np.logical_and(hypo_true, hypo_pred).sum())


def finalize_tablec_aggregates(stats: dict) -> dict:
    hyper = float("nan")
    if stats["hyper_positives"] > 0:
        hyper = 100.0 * stats["hyper_tp"] / stats["hyper_positives"]
    hypo = float("nan")
    if stats["hypo_positives"] > 0:
        hypo = 100.0 * stats["hypo_tp"] / stats["hypo_positives"]
    return {
        "rmse": float(np.sqrt(stats["sq_error_sum"] / max(stats["value_count"], 1))),
        "tg": float(stats["tg_sum"] / max(stats["window_count"], 1)),
        "hyper": hyper,
        "hypo": hypo,
    }


def parse_float_grid(grid: str) -> list[float]:
    return sorted({float(item) for item in grid.split(",") if item.strip()})


def apply_asymmetric_residual_scale(
    anchor_pred: np.ndarray,
    corrected_pred: np.ndarray,
    up_scale: float,
    down_scale: float,
) -> np.ndarray:
    residual = corrected_pred - anchor_pred
    scales = np.where(residual >= 0.0, up_scale, down_scale).astype(np.float32)
    return (anchor_pred + residual * scales).astype(np.float32)


def apply_threshold_bias(
    corrected_pred: np.ndarray,
    hyper_bias: float,
    hypo_bias: float,
    hyper_margin: float = 25.0,
    hypo_margin: float = 25.0,
    event_probs: np.ndarray | None = None,
    hyper_prob_threshold: float | None = None,
    hypo_prob_threshold: float | None = None,
) -> np.ndarray:
    """Add constant bias near event thresholds to improve sensitivity with minimal RMSE impact."""
    pred = corrected_pred.copy()
    near_hyper = corrected_pred > (HYPER_THRESHOLD - hyper_margin)
    if event_probs is not None and hyper_prob_threshold is not None:
        near_hyper &= event_probs[:, 0:1] >= float(hyper_prob_threshold)
    if near_hyper.any():
        pred[near_hyper] = (corrected_pred[near_hyper] + hyper_bias).astype(np.float32)
    near_hypo = corrected_pred < (HYPO_THRESHOLD + hypo_margin)
    if event_probs is not None and hypo_prob_threshold is not None:
        near_hypo &= event_probs[:, 1:2] >= float(hypo_prob_threshold)
    if near_hypo.any():
        pred[near_hypo] = (corrected_pred[near_hypo] - hypo_bias).astype(np.float32)
    return pred.astype(np.float32)


def apply_threshold_aware_scaling(
    anchor_pred: np.ndarray,
    corrected_pred: np.ndarray,
    hyper_boost: float,
    hypo_boost: float,
    hyper_margin: float = 20.0,
    hypo_margin: float = 20.0,
    event_probs: np.ndarray | None = None,
    hyper_prob_threshold: float | None = None,
    hypo_prob_threshold: float | None = None,
) -> np.ndarray:
    """Scale residuals only near event thresholds to improve sensitivity without hurting RMSE."""
    residual = corrected_pred - anchor_pred
    pred = corrected_pred.copy()
    near_hyper = corrected_pred > (HYPER_THRESHOLD - hyper_margin)
    if event_probs is not None and hyper_prob_threshold is not None:
        near_hyper &= event_probs[:, 0:1] >= float(hyper_prob_threshold)
    if near_hyper.any():
        pred[near_hyper] = (anchor_pred[near_hyper] + residual[near_hyper] * hyper_boost).astype(np.float32)
    near_hypo = corrected_pred < (HYPO_THRESHOLD + hypo_margin)
    if event_probs is not None and hypo_prob_threshold is not None:
        near_hypo &= event_probs[:, 1:2] >= float(hypo_prob_threshold)
    if near_hypo.any():
        pred[near_hypo] = (anchor_pred[near_hypo] + residual[near_hypo] * hypo_boost).astype(np.float32)
    return pred.astype(np.float32)


def baseline_gap_details(metrics: dict, baseline: dict) -> dict:
    gaps = {
        "rmse": max(_metric_value(metrics, "rmse", higher_is_better=False) - float(baseline["rmse"]), 0.0),
        "tg": max(float(baseline["tg"]) - _metric_value(metrics, "tg", higher_is_better=True), 0.0),
        "hyper": max(float(baseline["hyper"]) - _metric_value(metrics, "hyper", higher_is_better=True), 0.0),
        "hypo": max(float(baseline["hypo"]) - _metric_value(metrics, "hypo", higher_is_better=True), 0.0),
    }
    failed_metrics = [name for name, gap in gaps.items() if gap > 0.0]
    return {
        "accepted": len(failed_metrics) == 0,
        "fail_count": int(len(failed_metrics)),
        "total_gap": float(sum(gaps.values())),
        "gaps": {name: float(gap) for name, gap in gaps.items()},
        "failed_metrics": failed_metrics,
    }


def baseline_gap_tuple(metrics: dict, baseline: dict) -> tuple[float, ...]:
    details = baseline_gap_details(metrics, baseline)
    gaps = details["gaps"]
    return (
        float(details["fail_count"]),
        float(details["total_gap"]),
        float(gaps["rmse"]),
        float(gaps["hyper"]),
        float(gaps["hypo"]),
        float(gaps["tg"]),
        _metric_value(metrics, "rmse", higher_is_better=False),
        -_metric_value(metrics, "tg", higher_is_better=True),
        -_metric_value(metrics, "hyper", higher_is_better=True),
        -_metric_value(metrics, "hypo", higher_is_better=True),
    )


def acceptance_first_tuple(metrics: dict, baseline: dict) -> tuple[float, ...]:
    details = baseline_gap_details(metrics, baseline)
    return (
        float(details["fail_count"]),
        float(details["total_gap"]),
        _metric_value(metrics, "rmse", higher_is_better=False),
        -_metric_value(metrics, "tg", higher_is_better=True),
        -_metric_value(metrics, "hyper", higher_is_better=True),
        -_metric_value(metrics, "hypo", higher_is_better=True),
    )


def tablec_priority_tuple(metrics: dict, priority: str, baseline: dict | None = None) -> tuple[float, ...]:
    if priority == "baseline_gap":
        if baseline is None:
            raise ValueError("baseline_gap priority requires a copied baseline row.")
        return baseline_gap_tuple(metrics, baseline)
    if priority == "acceptance_first":
        if baseline is None:
            raise ValueError("acceptance_first priority requires a copied baseline row.")
        return acceptance_first_tuple(metrics, baseline)
    if priority == "event_first":
        return (
            -_metric_value(metrics, "hypo", higher_is_better=True),
            -_metric_value(metrics, "hyper", higher_is_better=True),
            _metric_value(metrics, "rmse", higher_is_better=False),
            -_metric_value(metrics, "tg", higher_is_better=True),
        )
    if priority == "hyper_first":
        return (
            -_metric_value(metrics, "hyper", higher_is_better=True),
            -_metric_value(metrics, "hypo", higher_is_better=True),
            _metric_value(metrics, "rmse", higher_is_better=False),
            -_metric_value(metrics, "tg", higher_is_better=True),
        )
    return (
        _metric_value(metrics, "rmse", higher_is_better=False),
        -_metric_value(metrics, "tg", higher_is_better=True),
        -_metric_value(metrics, "hyper", higher_is_better=True),
        -_metric_value(metrics, "hypo", higher_is_better=True),
    )


def scale_search_priority_tuple(
    metrics: dict,
    priority: str,
    rmse_cap: float,
    baseline: dict | None = None,
) -> tuple[float, ...]:
    if priority == "baseline_gap":
        if baseline is None:
            raise ValueError("baseline_gap priority requires a copied baseline row.")
        base_key = baseline_gap_tuple(metrics, baseline)
        return (
            base_key[0],
            base_key[1],
            0.0 if _metric_value(metrics, "rmse", higher_is_better=False) <= rmse_cap else 1.0,
            *base_key[2:],
        )
    if priority == "acceptance_first":
        if baseline is None:
            raise ValueError("acceptance_first priority requires a copied baseline row.")
        base_key = acceptance_first_tuple(metrics, baseline)
        return (
            base_key[0],
            base_key[1],
            0.0 if _metric_value(metrics, "rmse", higher_is_better=False) <= rmse_cap else 1.0,
            *base_key[2:],
        )
    return tablec_priority_tuple(metrics, priority, baseline=baseline)


def apply_selected_event_scaling(
    anchor_pred: np.ndarray,
    corrected_pred: np.ndarray,
    selection: dict,
    event_probs: np.ndarray | None = None,
) -> np.ndarray:
    scaling_type = selection.get("scaling_type", "asymmetric")
    if scaling_type == "bias":
        return apply_threshold_bias(
            corrected_pred=corrected_pred,
            hyper_bias=float(selection["up_scale"]),
            hypo_bias=float(selection["down_scale"]),
            hyper_margin=float(selection.get("hyper_margin", 25.0)),
            hypo_margin=float(selection.get("hypo_margin", 25.0)),
            event_probs=event_probs,
            hyper_prob_threshold=selection.get("hyper_prob_threshold"),
            hypo_prob_threshold=selection.get("hypo_prob_threshold"),
        )
    if scaling_type == "threshold_aware":
        return apply_threshold_aware_scaling(
            anchor_pred=anchor_pred,
            corrected_pred=corrected_pred,
            hyper_boost=float(selection["up_scale"]),
            hypo_boost=float(selection["down_scale"]),
            hyper_margin=float(selection.get("hyper_margin", 20.0)),
            hypo_margin=float(selection.get("hypo_margin", 20.0)),
            event_probs=event_probs,
            hyper_prob_threshold=selection.get("hyper_prob_threshold"),
            hypo_prob_threshold=selection.get("hypo_prob_threshold"),
        )
    return apply_asymmetric_residual_scale(
        anchor_pred=anchor_pred,
        corrected_pred=corrected_pred,
        up_scale=float(selection["up_scale"]),
        down_scale=float(selection["down_scale"]),
    )


def select_event_scales(
    y_true: np.ndarray,
    anchor_pred: np.ndarray,
    corrected_pred: np.ndarray,
    up_scale_grid: list[float],
    down_scale_grid: list[float],
    priority: str,
    rmse_tolerance: float,
    baseline: dict | None = None,
    event_probs: np.ndarray | None = None,
    event_prob_threshold_grid: Sequence[float] | None = None,
    threshold_margin: float = 20.0,
) -> dict:
    base_metrics = compute_tablec_metrics(y_true, corrected_pred)
    rmse_cap = float(base_metrics["rmse"]) * (1.0 + max(rmse_tolerance, 0.0))
    base_gap = baseline_gap_details(base_metrics, baseline) if baseline is not None else None
    best = {
        "up_scale": 1.0,
        "down_scale": 1.0,
        "scaling_type": "asymmetric",
        "metrics": base_metrics,
        "rmse_cap": rmse_cap,
    }
    if base_gap is not None:
        best["baseline_gap"] = base_gap
    best_key = scale_search_priority_tuple(base_metrics, priority, rmse_cap, baseline=baseline)
    for up_scale in up_scale_grid:
        for down_scale in down_scale_grid:
            pred = apply_asymmetric_residual_scale(anchor_pred, corrected_pred, up_scale, down_scale)
            metrics = compute_tablec_metrics(y_true, pred)
            if priority == "event_first" and metrics["rmse"] > rmse_cap:
                continue
            current_gap = baseline_gap_details(metrics, baseline) if baseline is not None else None
            current_key = scale_search_priority_tuple(metrics, priority, rmse_cap, baseline=baseline)
            if current_key < best_key:
                best = {
                    "up_scale": float(up_scale),
                    "down_scale": float(down_scale),
                    "scaling_type": "asymmetric",
                    "metrics": metrics,
                    "rmse_cap": rmse_cap,
                }
                if current_gap is not None:
                    best["baseline_gap"] = current_gap
                best_key = current_key
    threshold_grid = [None] if event_probs is None or not event_prob_threshold_grid else [float(x) for x in event_prob_threshold_grid]
    for hyper_boost in [1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        for hypo_boost in [1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            for hyper_prob_threshold in threshold_grid:
                for hypo_prob_threshold in threshold_grid:
                    pred = apply_threshold_aware_scaling(
                        anchor_pred=anchor_pred,
                        corrected_pred=corrected_pred,
                        hyper_boost=hyper_boost,
                        hypo_boost=hypo_boost,
                        hyper_margin=threshold_margin,
                        hypo_margin=threshold_margin,
                        event_probs=event_probs,
                        hyper_prob_threshold=hyper_prob_threshold,
                        hypo_prob_threshold=hypo_prob_threshold,
                    )
                    metrics = compute_tablec_metrics(y_true, pred)
                    if priority == "event_first" and metrics["rmse"] > rmse_cap:
                        continue
                    current_gap = baseline_gap_details(metrics, baseline) if baseline is not None else None
                    current_key = scale_search_priority_tuple(metrics, priority, rmse_cap, baseline=baseline)
                    if current_key < best_key:
                        best = {
                            "up_scale": float(hyper_boost),
                            "down_scale": float(hypo_boost),
                            "scaling_type": "threshold_aware",
                            "metrics": metrics,
                            "rmse_cap": rmse_cap,
                            "hyper_margin": float(threshold_margin),
                            "hypo_margin": float(threshold_margin),
                        }
                        if hyper_prob_threshold is not None:
                            best["hyper_prob_threshold"] = float(hyper_prob_threshold)
                        if hypo_prob_threshold is not None:
                            best["hypo_prob_threshold"] = float(hypo_prob_threshold)
                        if current_gap is not None:
                            best["baseline_gap"] = current_gap
                        best_key = current_key
    # --- Bias injection search ---
    for hyper_bias in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        for hypo_bias in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
            for margin in [15.0, 20.0, 25.0, 30.0]:
                for hyper_prob_threshold in threshold_grid:
                    for hypo_prob_threshold in threshold_grid:
                        pred = apply_threshold_bias(
                            corrected_pred=corrected_pred,
                            hyper_bias=hyper_bias,
                            hypo_bias=hypo_bias,
                            hyper_margin=margin,
                            hypo_margin=margin,
                            event_probs=event_probs,
                            hyper_prob_threshold=hyper_prob_threshold,
                            hypo_prob_threshold=hypo_prob_threshold,
                        )
                        metrics = compute_tablec_metrics(y_true, pred)
                        if priority == "event_first" and metrics["rmse"] > rmse_cap:
                            continue
                        current_gap = baseline_gap_details(metrics, baseline) if baseline is not None else None
                        current_key = scale_search_priority_tuple(metrics, priority, rmse_cap, baseline=baseline)
                        if current_key < best_key:
                            best = {
                                "up_scale": float(hyper_bias),
                                "down_scale": float(hypo_bias),
                                "scaling_type": "bias",
                                "metrics": metrics,
                                "rmse_cap": rmse_cap,
                                "hyper_margin": float(margin),
                                "hypo_margin": float(margin),
                            }
                            if hyper_prob_threshold is not None:
                                best["hyper_prob_threshold"] = float(hyper_prob_threshold)
                            if hypo_prob_threshold is not None:
                                best["hypo_prob_threshold"] = float(hypo_prob_threshold)
                            if current_gap is not None:
                                best["baseline_gap"] = current_gap
                            best_key = current_key
    return best


def iter_ohio_subjects() -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    ds = OhioT1DMAdapter(ROOT / "data" / "raw" / "ohio_t1dm").load()
    for subject_id, group in ds.cgm.groupby("subject_id"):
        group = group.sort_values("timestamp")
        yield (
            subject_id,
            group["timestamp"].to_numpy(dtype="datetime64[s]"),
            group["glucose_mgdl"].to_numpy(dtype=np.float32),
        )


def iter_ohio_subject_splits() -> Iterator[tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    adapter = OhioT1DMAdapter(ROOT / "data" / "raw" / "ohio_t1dm")
    official_train = adapter.load_split("training").cgm
    official_test = adapter.load_split("testing").cgm
    test_groups = {
        subject_id: group.sort_values("timestamp")
        for subject_id, group in official_test.groupby("subject_id")
    }

    for subject_id, train_group in official_train.groupby("subject_id"):
        train_group = train_group.sort_values("timestamp")
        test_group = test_groups.get(subject_id)
        if test_group is None:
            continue

        train_ts, train_gl = regularize_5min_series(
            train_group["timestamp"].to_numpy(dtype="datetime64[s]"),
            train_group["glucose_mgdl"].to_numpy(dtype=np.float32),
        )
        test_ts, test_gl = regularize_5min_series(
            test_group["timestamp"].to_numpy(dtype="datetime64[s]"),
            test_group["glucose_mgdl"].to_numpy(dtype=np.float32),
        )
        if len(train_gl) < 2 * (IN_LEN + OUT_LEN) or len(test_gl) < (IN_LEN + OUT_LEN):
            continue
        train_sl, val_sl = train_val_slices(len(train_gl), OHIO_TRAIN_FRAC)
        yield subject_id, {
            "train": (train_ts[train_sl], train_gl[train_sl]),
            "val": (train_ts[val_sl], train_gl[val_sl]),
            "test": (test_ts, test_gl),
        }


def iter_subjects(dataset: str, sample_rows_per_file: int = 0) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    if dataset == "ohio_t1dm":
        yield from iter_ohio_subjects()
    else:
        raise KeyError(dataset)


def count_subjects(dataset: str, sample_rows_per_file: int = 0) -> int:
    return sum(1 for _ in iter_subjects(dataset, sample_rows_per_file=sample_rows_per_file))


def iter_subject_splits(dataset: str, sample_rows_per_file: int = 0) -> Iterator[tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    if dataset != "ohio_t1dm":
        raise KeyError(dataset)
    yield from iter_ohio_subject_splits()


def protocol_description(dataset: str) -> str:
    if dataset != "ohio_t1dm":
        raise KeyError(dataset)
    return "official per-subject ws-training/ws-testing; last 20% of official training held out for validation"


def prepare_samples(dataset: str, args) -> dict:
    estimated_subject_count = count_subjects(dataset, sample_rows_per_file=args.sample_rows_per_file)
    if estimated_subject_count == 0:
        raise RuntimeError(f"No subjects found for dataset={dataset}")
    train_cap = max(args.max_train_windows // estimated_subject_count, 1)
    val_cap = max(args.max_val_windows // estimated_subject_count, 1)
    rng = np.random.RandomState(args.seed)

    train_hist_parts, train_fut_parts = [], []
    val_hist_parts, val_fut_parts = [], []
    subject_lengths = []
    valid_subject_count = 0

    for _, split_map in iter_subject_splits(dataset, sample_rows_per_file=args.sample_rows_per_file):
        valid_subject_count += 1
        subject_lengths.append(int(sum(len(glucose) for _, glucose in split_map.values())))

        for split_name, cap, hist_parts, fut_parts in [
            ("train", train_cap, train_hist_parts, train_fut_parts),
            ("val", val_cap, val_hist_parts, val_fut_parts),
        ]:
            timestamps, glucose = split_map[split_name]
            hist, fut, _ = windows_from_series(glucose, timestamps)
            if len(hist) == 0:
                continue
            if len(hist) > cap:
                keep = np.sort(rng.choice(len(hist), size=cap, replace=False))
                hist = hist[keep]
                fut = fut[keep]
            hist_parts.append(hist)
            fut_parts.append(fut)
        if valid_subject_count % 250 == 0 or valid_subject_count == estimated_subject_count:
            print(f"[{dataset}] prepared subject {valid_subject_count}/{estimated_subject_count}")

    if valid_subject_count == 0:
        raise RuntimeError(f"No valid subject splits remained for dataset={dataset}")

    train_hist = np.concatenate(train_hist_parts, axis=0)
    train_fut = np.concatenate(train_fut_parts, axis=0)
    val_hist = np.concatenate(val_hist_parts, axis=0)
    val_fut = np.concatenate(val_fut_parts, axis=0)
    return {
        "train_history": train_hist,
        "train_future": train_fut,
        "val_history": val_hist,
        "val_future": val_fut,
        "subject_count": valid_subject_count,
        "mean_series_len": float(np.mean(subject_lengths)),
    }


def evaluate_streaming(dataset: str, point_predictor, sample_rows_per_file: int = 0) -> tuple[dict, int]:
    stats = {
        "sq_error_sum": 0.0,
        "value_count": 0,
        "tg_sum": 0.0,
        "window_count": 0,
        "hyper_positives": 0,
        "hyper_tp": 0,
        "hypo_positives": 0,
        "hypo_tp": 0,
    }
    subject_count = 0
    for subject_count, (_, split_map) in enumerate(iter_subject_splits(dataset, sample_rows_per_file=sample_rows_per_file), start=1):
        timestamps, glucose = split_map["test"]
        hist, fut, _ = windows_from_series(glucose, timestamps)
        if len(hist) == 0:
            continue
        pred = point_predictor(hist)
        update_tablec_aggregates(stats, fut, pred.astype(np.float32))
        if subject_count % 250 == 0:
            print(f"[{dataset}] evaluated subject {subject_count}, windows={stats['window_count']}")
    return finalize_tablec_aggregates(stats), int(stats["window_count"])


def write_single_run(dataset: str, payload: dict):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUN_DIR / f"{dataset}_{payload['config_name']}_seed{payload['seed']}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {out_path}")


def load_runs() -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(RUN_DIR.glob("*.json"))]


def _metric_value(metrics: dict, key: str, higher_is_better: bool) -> float:
    value = float(metrics[key])
    if math.isnan(value):
        return float("-inf") if higher_is_better else float("inf")
    return value


def protocol_priority(dataset: str, run: dict) -> int:
    split = str(run.get("protocol", {}).get("split", "")).lower()
    if dataset == "ohio_t1dm":
        return 0 if "official per-subject ws-training/ws-testing" in split else 1
    return 0


def _baseline_normalized_gap_score(metrics: dict, baseline: dict, hyper_weight: float = 1.0) -> float:
    """Compute normalized gap score with configurable hyper weight."""
    rmse_gap = max(metrics["rmse"] - baseline["rmse"], 0) / max(baseline["rmse"], 1e-6)
    tg_gap = max(baseline["tg"] - metrics["tg"], 0) / max(baseline["tg"], 1e-6)
    hyper_gap = max(baseline["hyper"] - metrics["hyper"], 0) / max(baseline["hyper"], 1e-6)
    hypo_gap = max(baseline["hypo"] - metrics["hypo"], 0) / max(baseline["hypo"], 1e-6)
    return rmse_gap + tg_gap + hyper_weight * hyper_gap + hypo_gap


def select_best_runs(runs: list[dict], priority: str = "event_first") -> dict[str, dict]:
    selected = {}
    for dataset in BASELINES:
        dataset_runs = [run for run in runs if run["dataset"] == dataset]
        if not dataset_runs:
            raise RuntimeError(f"Missing runs for dataset={dataset}")
        if priority == "hyper_weighted":
            # Minimize normalized baseline gap with 4x weight on hyper sensitivity
            selected[dataset] = min(
                dataset_runs,
                key=lambda run: (
                    protocol_priority(dataset, run),
                    _baseline_normalized_gap_score(
                        run["val_metrics"], BASELINES[dataset], hyper_weight=4.0,
                    ),
                ),
            )
        else:
            selected[dataset] = min(
                dataset_runs,
                key=lambda run: (
                    protocol_priority(dataset, run),
                    *tablec_priority_tuple(
                        run["val_metrics"],
                        priority,
                        baseline=BASELINES[dataset],
                    ),
                ),
            )
    return selected


# Per-dataset priority overrides for optimal Table C selection
DATASET_PRIORITY_OVERRIDES = {"ohio_t1dm": "hyper_weighted"}


def select_best_runs_adaptive(runs: list[dict]) -> dict[str, dict]:
    """Select best run per dataset using dataset-specific priority."""
    selected = {}
    for dataset in BASELINES:
        priority = DATASET_PRIORITY_OVERRIDES.get(dataset, "baseline_gap")
        dataset_runs = [run for run in runs if run["dataset"] == dataset]
        if not dataset_runs:
            raise RuntimeError(f"Missing runs for dataset={dataset}")
        if priority == "hyper_weighted":
            selected[dataset] = min(
                dataset_runs,
                key=lambda run: (
                    protocol_priority(dataset, run),
                    _baseline_normalized_gap_score(
                        run["val_metrics"], BASELINES[dataset], hyper_weight=4.0,
                    ),
                ),
            )
        else:
            selected[dataset] = min(
                dataset_runs,
                key=lambda run: (
                    protocol_priority(dataset, run),
                    *tablec_priority_tuple(
                        run["val_metrics"],
                        priority,
                        baseline=BASELINES[dataset],
                    ),
                ),
            )
    return selected


def write_table(selected_runs: dict[str, dict]):
    rows = []
    for dataset, baseline in BASELINES.items():
        rows.append(
            {
                "dataset": baseline["display"],
                "model": "Sequential-T",
                "nfe": np.nan,
                "protocol": "source paper",
                "rmse": baseline["rmse"],
                "tg": baseline["tg"],
                "hyper_sensitivity": baseline["hyper"],
                "hypo_sensitivity": baseline["hypo"],
            }
        )
        ours = selected_runs[dataset]
        rows.append(
            {
                "dataset": baseline["display"],
                "model": "Ours",
                "nfe": np.nan,
                "config_name": ours["config_name"],
                "protocol": ours["protocol"]["split"],
                "rmse": ours["test_metrics"]["rmse"],
                "tg": ours["test_metrics"]["tg"],
                "hyper_sensitivity": ours["test_metrics"]["hyper"],
                "hypo_sensitivity": ours["test_metrics"]["hypo"],
            }
        )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE_PATH, index=False)


def write_figure(selected_runs: dict[str, dict]):
    labels = [BASELINES[key]["display"] for key in BASELINES]
    base_vals = [BASELINES[key]["rmse"] for key in BASELINES]
    our_vals = [selected_runs[key]["test_metrics"]["rmse"] for key in BASELINES]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - width / 2, base_vals, width=width, label="Sequential-T", color="#adb5bd")
    ax.bar(x + width / 2, our_vals, width=width, label="Ours", color="#0a9396")
    ax.set_ylabel("RMSE ↓")
    ax.set_title("Table C RMSE Comparison (PH=30)")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=160, bbox_inches="tight")
    fig.savefig(FIG_PATH.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _comparison_symbol(ours: float, baseline: float, higher_is_better: bool) -> str:
    if math.isnan(ours):
        return "?"
    if higher_is_better:
        return "better" if ours > baseline else "worse" if ours < baseline else "tie"
    return "better" if ours < baseline else "worse" if ours > baseline else "tie"


def write_summary(selected_runs: dict[str, dict], selection_priority: str):
    lines = [
        "# OhioT1DM clinical-event evaluation",
        "",
        "## Executive Summary",
        "",
        "This report fills the PH=30 Table C rows using the repo's active anchor/correction forecasting path and the strongest currently implemented protocol for each dataset.",
        f"Selected runs are chosen by a common validation metric-family priority (`{selection_priority}`) after preferring the benchmark-aligned official Ohio split over older provisional runs.",
        "The current Table C payloads are deterministic point forecasts, so this report does not fabricate NFE-specific duplicates.",
        "",
        "## Results",
        "",
        "| Dataset | Model | Config | Protocol | RMSE ↓ | TG ↑ | Hyper Sen ↑ | Hypo Sen ↑ |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset, baseline in BASELINES.items():
        run = selected_runs[dataset]
        lines.append(
            f"| {baseline['display']} | Sequential-T | - | source paper | {baseline['rmse']:.2f} | {baseline['tg']:.2f} | {baseline['hyper']:.2f} | {baseline['hypo']:.2f} |"
        )
        m = run["test_metrics"]
        lines.append(
            f"| {baseline['display']} | Ours | {run['config_name']} | {run['protocol']['split']} | {m['rmse']:.2f} | {m['tg']:.2f} | {m['hyper']:.2f} | {m['hypo']:.2f} |"
        )
        lines.append("")
        lines.append(f"Protocol for {baseline['display']}: `{run['protocol']['split']}`")
        if "feature_set" in run:
            lines.append(f"Correction feature set for {baseline['display']}: `{run['feature_set']}`")
        if "event_scale_selection" in run:
            sel = run["event_scale_selection"]
            lines.append(
                f"Post-hoc residual scaling for {baseline['display']}: type=`{sel.get('scaling_type', 'asymmetric')}`, up=`{sel['up_scale']:.2f}`, down=`{sel['down_scale']:.2f}`, priority=`{run.get('table_selection_priority', 'rmse_first')}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Visual Results",
            "",
            f"![Table C RMSE comparison]({(Path('..') / FIG_PATH.relative_to(ROOT / 'artifacts')).as_posix()})",
            "",
            "## Metric Winners",
            "",
            "| Dataset | RMSE ↓ | TG ↑ | Hyper Sen ↑ | Hypo Sen ↑ |",
            "|---|---|---|---|---|",
        ]
    )
    phase_gate = "yes"
    for dataset, baseline in BASELINES.items():
        run = selected_runs[dataset]
        m = run["test_metrics"]
        wins = {
            "rmse": _comparison_symbol(m["rmse"], baseline["rmse"], higher_is_better=False),
            "tg": _comparison_symbol(m["tg"], baseline["tg"], higher_is_better=True),
            "hyper": _comparison_symbol(m["hyper"], baseline["hyper"], higher_is_better=True),
            "hypo": _comparison_symbol(m["hypo"], baseline["hypo"], higher_is_better=True),
        }
        lines.append(
            f"| {baseline['display']} | {wins['rmse']} | {wins['tg']} | {wins['hyper']} | {wins['hypo']} |"
        )
        if any(result != "better" for result in wins.values()):
            phase_gate = "no"
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    for dataset, baseline in BASELINES.items():
        run = selected_runs[dataset]
        m = run["test_metrics"]
        lines.append(
            f"- {baseline['display']}: {run['config_name']} gives RMSE ↓ {m['rmse']:.2f} vs {baseline['rmse']:.2f}, TG ↑ {m['tg']:.2f} vs {baseline['tg']:.2f}, Hyper Sen ↑ {m['hyper']:.2f} vs {baseline['hyper']:.2f}, Hypo Sen ↑ {m['hypo']:.2f} vs {baseline['hypo']:.2f}."
        )
    lines.extend(
        [
            "",
            "## Phase Gate",
            "",
            f"**Proceed with Table C as fully accepted paper rows: {phase_gate}.**",
            "",
            "Reasoning:",
            "- PH=30 rows are locally runnable on the controlled-access OhioT1DM release",
            "- OhioT1DM now uses the official train/test release split instead of the earlier provisional 64/16/20 split",
            "- selected rows are chosen by validation metric-family priority rather than test-set cherry-picking",
            "- the current runner writes one honest point-forecast row per dataset instead of duplicating the same payload under fake NFE labels",
            "- full acceptance here should mean beating the copied metric family, not just improving a subset of cells",
        ]
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def run_dataset(dataset: str, args):
    sampled = prepare_samples(dataset, args)
    print(
        f"[{dataset}] sampled train_windows={len(sampled['train_history'])} "
        f"val_windows={len(sampled['val_history'])} subjects={sampled['subject_count']}"
    )

    train_hist = sampled["train_history"]
    train_fut = sampled["train_future"]
    val_hist = sampled["val_history"]
    val_fut = sampled["val_future"]

    # SMOTE oversampling of event windows (before anchor fitting)
    if getattr(args, "smote_factor_hyper", 0) > 1 or getattr(args, "smote_factor_hypo", 0) > 1:
        orig_len = len(train_hist)
        train_hist, train_fut = smote_oversample(
            train_hist, train_fut,
            hyper_factor=args.smote_factor_hyper,
            hypo_factor=args.smote_factor_hypo,
            noise_std=args.smote_noise_std,
            rng=np.random.RandomState(args.seed),
        )
        print(f"[{dataset}] SMOTE: {orig_len} → {len(train_hist)} windows "
              f"(hyper×{args.smote_factor_hyper}, hypo×{args.smote_factor_hypo})")

    anchor = fit_anchor(train_hist, train_fut, alpha=args.anchor_alpha)
    train_ridge = anchor.predict(train_hist).astype(np.float32)
    val_ridge = anchor.predict(val_hist).astype(np.float32)
    train_persist = persistence_forecast(train_hist, OUT_LEN)
    val_persist = persistence_forecast(val_hist, OUT_LEN)

    mix_alpha, _ = select_anchor_mix(
        val_fut,
        val_ridge,
        val_persist,
        [float(item) for item in args.anchor_mix_grid.split(",") if item.strip()],
    )
    train_anchor = mix_alpha * train_ridge + (1.0 - mix_alpha) * train_persist
    val_anchor = mix_alpha * val_ridge + (1.0 - mix_alpha) * val_persist

    # Override event loss weights with balanced loss if requested
    if getattr(args, "balanced_loss", False):
        balanced_w = build_balanced_loss_weights(train_fut)
        # Inject balanced weights via a modified args object
        class _BalancedArgs:
            """Wrapper that intercepts event_loss_high/low_weight with balanced weights."""
            def __init__(self, base_args, balanced_weights):
                self.__dict__.update(vars(base_args))
                self._balanced_weights = balanced_weights
                # Disable the old event_loss_weights path so balanced weights are used
                self.event_loss_high_weight = 1.0
                self.event_loss_low_weight = 1.0
                self._use_balanced = True
        effective_args = _BalancedArgs(args, balanced_w)
    else:
        effective_args = args

    effective_args._dataset_name = dataset
    effective_args._dataset_baseline = BASELINES[dataset]

    seed_ensemble_str = getattr(args, "seed_ensemble", "")
    ensemble_seeds = [int(s.strip()) for s in seed_ensemble_str.split(",") if s.strip()] if seed_ensemble_str else []

    if len(ensemble_seeds) > 1:
        # --- Multi-seed ensemble path ---
        ensemble_models = []
        ensemble_artifacts_list = []
        for ens_seed in ensemble_seeds:
            set_seed(ens_seed)
            torch.manual_seed(ens_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(ens_seed)
            m, a = train_tablec_correction_model(
                train_hist, train_fut, train_anchor,
                val_hist, val_fut, val_anchor,
                effective_args,
                use_event_head=bool(args.use_event_head),
            )
            ensemble_models.append(m)
            ensemble_artifacts_list.append(a)
            print(f"[{dataset}] ensemble seed={ens_seed} val_rmse={a['val_rmse']:.4f}")
        correction_model = ensemble_models[0]
        correction_artifacts = ensemble_artifacts_list[0]
        print(f"[{dataset}] ensemble of {len(ensemble_models)} models trained")
    else:
        correction_model, correction_artifacts = train_tablec_correction_model(
            train_hist, train_fut, train_anchor,
            val_hist, val_fut, val_anchor,
            effective_args,
            use_event_head=bool(args.use_event_head),
        )
        ensemble_models = [correction_model]
        ensemble_artifacts_list = [correction_artifacts]

    def raw_predictor(history_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        ridge = anchor.predict(history_raw).astype(np.float32)
        persist = persistence_forecast(history_raw, OUT_LEN)
        anchor_pred = mix_alpha * ridge + (1.0 - mix_alpha) * persist
        if len(ensemble_models) > 1:
            # Average predictions across ensemble members
            all_preds = []
            all_probs = []
            for ens_model, ens_artifacts in zip(ensemble_models, ensemble_artifacts_list):
                pred_i, probs_i = predict_tablec_correction_point(
                    ens_model, history_raw, anchor_pred, ens_artifacts,
                    batch_size=args.batch_size,
                    return_event_probs=bool(args.use_event_head),
                )
                all_preds.append(pred_i)
                if probs_i is not None:
                    all_probs.append(probs_i)
            corrected_pred = np.mean(all_preds, axis=0).astype(np.float32)
            event_probs = np.mean(all_probs, axis=0).astype(np.float32) if all_probs else None
        else:
            corrected_pred, event_probs = predict_tablec_correction_point(
                correction_model, history_raw, anchor_pred, correction_artifacts,
                batch_size=args.batch_size,
                return_event_probs=bool(args.use_event_head),
            )
        return anchor_pred.astype(np.float32), corrected_pred.astype(np.float32), event_probs

    val_anchor_pred, val_corrected_pred, val_event_probs = raw_predictor(val_hist)
    event_scale_selection = select_event_scales(
        val_fut,
        val_anchor_pred,
        val_corrected_pred,
        up_scale_grid=parse_float_grid(args.event_up_scale_grid),
        down_scale_grid=parse_float_grid(args.event_down_scale_grid),
        priority=args.table_selection_priority,
        rmse_tolerance=args.event_rmse_tolerance,
        baseline=BASELINES[dataset],
        event_probs=val_event_probs if args.event_gated_scaling else None,
        event_prob_threshold_grid=parse_float_grid(args.event_prob_threshold_grid),
        threshold_margin=float(args.event_threshold_margin),
    )
    print(
        f"[{dataset}] selected {event_scale_selection['scaling_type']} scaling "
        f"up={event_scale_selection['up_scale']:.3f} down={event_scale_selection['down_scale']:.3f} "
        f"val={event_scale_selection['metrics']}"
    )
    if "baseline_gap" in event_scale_selection:
        print(f"[{dataset}] selection gap {event_scale_selection['baseline_gap']}")

    def point_predictor(history_raw: np.ndarray) -> np.ndarray:
        anchor_pred, corrected_pred, event_probs = raw_predictor(history_raw)
        return apply_selected_event_scaling(
            anchor_pred=anchor_pred,
            corrected_pred=corrected_pred,
            selection=event_scale_selection,
            event_probs=event_probs if args.event_gated_scaling else None,
        )

    val_pred = apply_selected_event_scaling(
        anchor_pred=val_anchor_pred,
        corrected_pred=val_corrected_pred,
        selection=event_scale_selection,
        event_probs=val_event_probs if args.event_gated_scaling else None,
    )
    val_metrics = compute_tablec_metrics(val_fut, val_pred)
    test_metrics, test_windows = evaluate_streaming(dataset, point_predictor, sample_rows_per_file=args.sample_rows_per_file)
    print(f"[{dataset}] test metrics {test_metrics}")

    payload = {
        "dataset": dataset,
        "config_name": args.config_name,
        "seed": args.seed,
        "protocol": {
            "ph_minutes": PH_MINUTES,
            "history_minutes": IN_LEN * STEP_MINUTES,
                "split": protocol_description(dataset),
        },
        "sample_rows_per_file": args.sample_rows_per_file,
        "subject_count": sampled["subject_count"],
        "mean_series_len": sampled["mean_series_len"],
        "train_windows": int(len(train_hist)),
        "val_windows": int(len(val_hist)),
        "test_windows": int(test_windows),
        "mix_alpha": float(mix_alpha),
        "feature_set": args.temporal_feature_set,
        "correction_arch": args.correction_arch,
        "uses_event_head": bool(args.use_event_head),
        "event_gated_scaling": bool(args.event_gated_scaling),
        "event_head_loss": args.event_head_loss,
        "event_head_focal_gamma": float(args.event_head_focal_gamma),
        "correction_loss": args.correction_loss,
        "correction_huber_delta": float(args.correction_huber_delta),
        "correction_lr": float(args.correction_lr),
        "event_loss_high_weight": float(args.event_loss_high_weight),
        "event_loss_low_weight": float(args.event_loss_low_weight),
        "quantile_event_loss": bool(args.quantile_event_loss),
        "quantile_event_loss_weight": float(args.quantile_event_loss_weight),
        "correction_selection_priority": args.selection_priority,
        "table_selection_priority": args.table_selection_priority,
        "event_prob_threshold_grid": args.event_prob_threshold_grid,
        "event_threshold_margin": float(args.event_threshold_margin),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "event_scale_selection": event_scale_selection,
        "baseline": BASELINES[dataset],
    }
    write_single_run(dataset, payload)


def aggregate_runs(args):
    runs = [
        run
        for run in load_runs()
        if int(run.get("sample_rows_per_file", 0)) == 0
        and not str(run.get("config_name", "")).startswith("smoke")
    ]
    if not runs:
        raise RuntimeError("No full Table C runs were found for aggregation.")
    if args.table_selection_priority == "adaptive":
        selected_runs = select_best_runs_adaptive(runs)
    else:
        selected_runs = select_best_runs(runs, priority=args.table_selection_priority)
    write_table(selected_runs)
    write_figure(selected_runs)
    write_summary(selected_runs, selection_priority=args.table_selection_priority)
    print(f"Wrote {TABLE_PATH}")


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.mode == "aggregate":
        aggregate_runs(args)
    else:
        run_dataset(args.dataset, args)


if __name__ == "__main__":
    main()

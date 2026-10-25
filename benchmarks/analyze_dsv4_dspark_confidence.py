# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate and summarize DeepSeek-V4 DSpark confidence-head dumps."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _ece(probs: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> float:
    if probs.numel() == 0:
        return math.nan
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    total = probs.numel()
    value = 0.0
    for index in range(bins):
        if index + 1 == bins:
            mask = (probs >= edges[index]) & (probs <= edges[index + 1])
        else:
            mask = (probs >= edges[index]) & (probs < edges[index + 1])
        count = int(mask.sum())
        if count:
            gap = (probs[mask].mean() - targets[mask].mean()).abs()
            value += float(gap) * count / total
    return value


def _soft_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() == 0:
        return math.nan
    return float(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets))


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() < 2:
        return 1.0
    # A dense one-dimensional search is deterministic, stable for fractional
    # TV labels, and avoids adding scipy/sklearn to the benchmark environment.
    temperatures = torch.logspace(-1.0, 1.0, 801, dtype=torch.float64)
    scaled = logits.to(torch.float64).unsqueeze(0) / temperatures.unsqueeze(1)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        scaled,
        targets.to(torch.float64).expand_as(scaled),
        reduction="none",
    ).mean(dim=1)
    return float(temperatures[int(losses.argmin())])


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return math.nan
    # Proposal blocks are at most five positions in the production contract;
    # this exact pairwise definition is small and handles ties correctly.
    comparisons = positives[:, None] - negatives[None, :]
    return float((comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean())


def _summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    observed_logits: torch.Tensor | None = None,
    observed_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    scaled_logits = logits / temperature
    probs = scaled_logits.sigmoid()
    result: dict[str, Any] = {
        "count": int(logits.numel()),
        "temperature": temperature,
        "predicted_mean": float(probs.mean()) if probs.numel() else math.nan,
        "target_mean": float(targets.mean()) if targets.numel() else math.nan,
        "ece": _ece(probs, targets),
        "brier": (
            float(torch.square(probs - targets).mean()) if probs.numel() else math.nan
        ),
        "soft_nll": _soft_nll(scaled_logits, targets),
    }
    if (
        observed_logits is not None
        and observed_labels is not None
        and observed_labels.numel()
    ):
        result["observed_count"] = int(observed_labels.numel())
        result["observed_acceptance"] = float(observed_labels.mean())
        result["observed_auc"] = _binary_auc(
            torch.sigmoid(observed_logits / temperature), observed_labels
        )
    return result


def _load_rows(input_dir: Path, rank: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("spec_alignment_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload.get("rank", -1)) != rank:
            continue
        required = (
            "draft_confidence_logits",
            "conditional_acceptance_targets",
            "num_draft_tokens",
            "output_valid_counts",
        )
        if not all(key in payload for key in required):
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def analyze(rows: list[dict[str, Any]], calibration_fraction: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("no rank-matching confidence alignment dumps were found")
    max_position = max(max(row["num_draft_tokens"], default=0) for row in rows)
    by_position: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    prefix_by_position: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row_index, row in enumerate(rows):
        logits = row["draft_confidence_logits"].to(torch.float64)
        targets = row["conditional_acceptance_targets"].to(torch.float64)
        valid_counts = row["output_valid_counts"].to(torch.int64)
        offset = 0
        for request_index, count_raw in enumerate(row["num_draft_tokens"]):
            count = int(count_raw)
            request_logits = logits[offset : offset + count]
            request_targets = targets[offset : offset + count]
            accepted = max(0, min(count, int(valid_counts[request_index]) - 1))
            split = (
                "calibration"
                if row_index < len(rows) * calibration_fraction
                else "evaluation"
            )
            cumulative_pred_logits = request_logits
            cumulative_targets = request_targets.cumprod(dim=0)
            for position in range(count):
                bucket = by_position[position]
                bucket[f"{split}_logits"].append(
                    request_logits[position : position + 1]
                )
                bucket[f"{split}_targets"].append(
                    request_targets[position : position + 1]
                )
                if position <= accepted and position < count:
                    observed = torch.tensor(
                        [1.0 if position < accepted else 0.0], dtype=torch.float64
                    )
                    bucket[f"{split}_observed_logits"].append(
                        request_logits[position : position + 1]
                    )
                    bucket[f"{split}_observed"].append(observed)

                prefix_bucket = prefix_by_position[position]
                prefix_bucket[f"{split}_logits"].append(
                    cumulative_pred_logits[: position + 1]
                )
                prefix_bucket[f"{split}_targets"].append(
                    cumulative_targets[position : position + 1]
                )
            offset += count

    temperatures: list[float] = []
    position_rows: list[dict[str, Any]] = []
    for position in range(max_position):
        bucket = by_position[position]
        calibration_logits = torch.cat(bucket["calibration_logits"])
        calibration_targets = torch.cat(bucket["calibration_targets"])
        temperature = _fit_temperature(calibration_logits, calibration_targets)
        temperatures.append(temperature)
        evaluation_logits = torch.cat(
            bucket["evaluation_logits"] or bucket["calibration_logits"]
        )
        evaluation_targets = torch.cat(
            bucket["evaluation_targets"] or bucket["calibration_targets"]
        )
        observed_parts = bucket["evaluation_observed"] or bucket["calibration_observed"]
        observed_logits_parts = (
            bucket["evaluation_observed_logits"]
            or bucket["calibration_observed_logits"]
        )
        observed = torch.cat(observed_parts) if observed_parts else None
        observed_logits = (
            torch.cat(observed_logits_parts) if observed_logits_parts else None
        )
        position_rows.append(
            {
                "position": position + 1,
                "raw": _summary(
                    evaluation_logits,
                    evaluation_targets,
                    1.0,
                    observed_logits,
                    observed,
                ),
                "calibrated": _summary(
                    evaluation_logits,
                    evaluation_targets,
                    temperature,
                    observed_logits,
                    observed,
                ),
            }
        )

    prefix_rows: list[dict[str, Any]] = []
    for position in range(max_position):
        bucket = prefix_by_position[position]
        logits_parts = bucket["evaluation_logits"] or bucket["calibration_logits"]
        target_parts = bucket["evaluation_targets"] or bucket["calibration_targets"]
        prefix_probs = torch.stack(
            [
                torch.sigmoid(part / torch.tensor(temperatures[: part.numel()])).prod()
                for part in logits_parts
            ]
        )
        prefix_targets = torch.cat(target_parts)
        prefix_rows.append(
            {
                "position": position + 1,
                "count": int(prefix_probs.numel()),
                "predicted_mean": float(prefix_probs.mean()),
                "target_mean": float(prefix_targets.mean()),
                "ece": _ece(prefix_probs, prefix_targets),
                "brier": float(torch.square(prefix_probs - prefix_targets).mean()),
            }
        )

    return {
        "schema_version": 1,
        "dump_count": len(rows),
        "rank": int(rows[0].get("rank", -1)),
        "step_range": [int(rows[0]["step"]), int(rows[-1]["step"])],
        "calibration_fraction": calibration_fraction,
        "sequential_temperatures": temperatures,
        "per_conditional_position": position_rows,
        "per_prefix_position": prefix_rows,
        "source_files": [row["_path"] for row in rows],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must be in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.input_dir, args.rank)
    result = analyze(rows, args.calibration_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def json_safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    args.output.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps({key: result[key] for key in ("dump_count", "rank", "step_range")})
    )


if __name__ == "__main__":
    main()

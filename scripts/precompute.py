"""Sinh dữ liệu đánh giá và artifacts từ cấu hình."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from darp.analysis import (
    acc_by_nrecs,
    acc_by_sector,
    acc_by_year,
    acc_overall,
    reliability_by_sector,
    reliability_by_year,
    reliability_heatmap,
    reliability_quintiles,
    spearman_c_true_q,
    topk_table,
    toy_example,
)
from darp.config import Config, load_config
from darp.dataset import Dataset
from darp.events import build_events
from darp.evaluation import MODELS, run_rolling
from darp.metrics import bootstrap_ci, acc_cost
from darp.relevance import FEATS


def _config_for_summary(cfg: Config) -> dict[str, Any]:
    values: dict[str, Any] = {
        key: getattr(cfg.data, key)
        for key in cfg.data.__dataclass_fields__
        if key != "year_shock"
    }
    values.update({key: getattr(cfg.eval, key) for key in cfg.eval.__dataclass_fields__})
    values.update(
        {
            key: getattr(cfg.model, key)
            for key in cfg.model.__dataclass_fields__
            if key != "ridge"
        }
    )
    return values


def _summary(
    cfg: Config,
    predictions: pd.DataFrame,
    reliability: pd.DataFrame,
    etas: np.ndarray,
    ablation: dict[str, float | None],
    sensitivity: pd.DataFrame,
) -> dict[str, Any]:
    overall = acc_overall(predictions)
    confidence_intervals = {
        model: bootstrap_ci(predictions[model], predictions.y)
        for model in MODELS
    }
    by_year = acc_by_year(predictions)
    eta_mean = pd.Series(np.nanmean(etas, axis=0), index=FEATS)
    topk = topk_table(predictions)
    rel_year = reliability_by_year(reliability)
    rel_sector = reliability_by_sector(reliability)
    heatmap = reliability_heatmap(reliability)
    quintiles = reliability_quintiles(reliability)
    by_sector = acc_by_sector(predictions)
    by_nrecs = acc_by_nrecs(predictions)
    results: dict[str, Any] = {
        "config": _config_for_summary(cfg),
        "n_test_events": int(len(predictions)),
        "ACC_overall": overall.ACC.round(4).to_dict(),
        "ACC_CI95": {
            model: [round(low, 4), round(high, 4)]
            for model, (low, high) in confidence_intervals.items()
        },
        "ACC_by_year": by_year.round(4).to_dict(),
        "eta_mean": eta_mean.round(3).to_dict(),
        "fig7_topk": topk.ACC.round(4).to_dict(),
        "fig8_ablation": {
            key: round(value, 4) if value is not None else None
            for key, value in ablation.items()
        },
        "fig9_sensitivity": sensitivity.round(4).to_dict(orient="records"),
        "reliability_by_year": rel_year["mean"].round(4).to_dict(),
        "spearman_c_vs_true_q": round(spearman_c_true_q(reliability), 4),
        "acc_by_sector": by_sector.round(4).to_dict(),
        "acc_by_nrecs": by_nrecs.round(4).to_dict(),
        "reliability_by_sector": rel_sector.round(4).to_dict(),
        "reliability_heatmap": heatmap.round(4).to_dict(),
        "reliability_quintiles": quintiles.round(4).to_dict(),
        "toy_example": toy_example(),
    }
    return results


def precompute(
    config_path: Path,
    output_dir: Path,
    skip_ablation: bool = False,
    skip_sensitivity: bool = False,
) -> dict[str, Any]:
    """Chạy pipeline tái lập và ghi toàn bộ artifacts."""
    cfg = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.synthetic(cfg)
    events, nodes = build_events(ds, cfg.eval.OB, cfg.eval.PD, cfg.eval.min_recs)
    predictions, reliability, etas = run_rolling(events, nodes, ds, cfg, label="DARP full")
    predictions["n_bin"] = pd.cut(
        predictions.n_recs, [2, 3, 4, 5, 6, 100], labels=["3", "4", "5", "6", "7+"]
    )
    ablation: dict[str, float | None] = {
        "DARP": acc_cost(predictions.DARP, predictions.y)
    }
    if skip_ablation:
        ablation.update({f"-{feature}": None for feature in FEATS})
        ablation["DARP_C"] = None
    else:
        for index, feature in enumerate(FEATS):
            selected = np.ones(6, dtype=bool)
            selected[index] = False
            prediction, _, _ = run_rolling(
                events, nodes, ds, cfg, use=selected, label=f"-{feature}", verbose=False
            )
            ablation[f"-{feature}"] = acc_cost(prediction.DARP, prediction.y)
        prediction, _, _ = run_rolling(
            events, nodes, ds, cfg, random_cross=True, label="DARP_C", verbose=False
        )
        ablation["DARP_C"] = acc_cost(prediction.DARP, prediction.y)
    ablation["MV"] = acc_cost(predictions.MV, predictions.y)
    ablation["WM"] = acc_cost(predictions.WM, predictions.y)

    sensitivity_rows: list[dict[str, Any]] = []
    if skip_sensitivity:
        sensitivity = pd.DataFrame(columns=["param", "value", *MODELS])
    else:
        all_months = sorted(events.month.unique())
        months = all_months[::2] if cfg.experiment.fast else all_months
        settings = cfg.experiment.sensitivity or {}
        for parameter, values in settings.items():
            for value in values:
                variant = cfg.with_overrides(**{f"eval.{parameter}": value})
                if parameter in {"OB", "PD"}:
                    variant_events, variant_nodes = build_events(
                        ds, variant.eval.OB, variant.eval.PD, variant.eval.min_recs
                    )
                else:
                    variant_events, variant_nodes = events, nodes
                result, _, _ = run_rolling(
                    variant_events,
                    variant_nodes,
                    ds,
                    variant,
                    months=months,
                    label=f"{parameter}={value}",
                    verbose=False,
                )
                sensitivity_rows.append(
                    {
                        "param": parameter,
                        "value": value,
                        **{
                            model: round(float(acc_cost(result[model], result.y)), 4)
                            for model in MODELS
                        },
                    }
                )
        sensitivity = pd.DataFrame(sensitivity_rows)

    results = _summary(cfg, predictions, reliability, etas, ablation, sensitivity)
    events.to_parquet(output_dir / "events.parquet", index=False)
    nodes.to_parquet(output_dir / "nodes.parquet", index=False)
    ds.recs.to_parquet(output_dir / "recs.parquet", index=False)
    ds.stocks.to_parquet(output_dir / "stocks.parquet", index=False)
    ds.prices.to_parquet(output_dir / "prices.parquet")
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    reliability.to_parquet(output_dir / "reliability.parquet", index=False)
    pd.DataFrame(etas, columns=FEATS).assign(month=sorted(events.month.unique())[cfg.eval.TP :]).to_csv(
        output_dir / "eta.csv", index=False
    )
    sensitivity.to_csv(output_dir / "sensitivity.csv", index=False)
    pd.DataFrame([ablation]).to_csv(output_dir / "ablation.csv", index=False)
    with (output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"\nArtifacts đã ghi vào {output_dir.resolve()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    precompute(args.config, args.out, args.skip_ablation, args.skip_sensitivity)


if __name__ == "__main__":
    main()

"""Fit and evaluate surface models under the shared protocol; log results to MLflow."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

from volsurf.config import from_dict, load_yaml
from volsurf.eval.runner import EvalConfig, flatten, key_columns, load_eval_days, run, summarize


def git_revision() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "configs"], capture_output=True, text=True
    )
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def log_mlflow(experiment, source, years, spec, eval_dict, summary_row, daily_path) -> None:
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{source}/{spec['name']}"):
        mlflow.log_params(
            {"source": source, "years": str(list(years)), "git": git_revision()}
            | flatten(spec, "model.")
            | flatten(eval_dict, "eval.")
        )
        metrics = {k.replace("/", "."): float(v) for k, v in summary_row.items() if pd.notna(v)}
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(daily_path))


def for_panel(spec, panel: str):
    """Fill ``{panel}`` in string parameters (e.g. a checkpoint trained on that panel)."""
    if isinstance(spec, dict):
        return {k: for_panel(v, panel) for k, v in spec.items()}
    if isinstance(spec, list):
        return [for_panel(v, panel) for v in spec]
    return spec.replace("{panel}", panel) if isinstance(spec, str) else spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--models", nargs="*", help="model names to run (default: all)")
    parser.add_argument("--sources", nargs="*", help="sources to run (default: all)")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-days", type=int, help="cap sampled days (smoke runs)")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--experiment", help="override the experiment name (results directory)")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries"
    )
    args = parser.parse_args()

    spec = load_yaml(args.config)
    for item in args.set:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = spec
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    if args.experiment:
        spec["experiment"] = args.experiment
    eval_dict = spec.get("eval", {})
    cfg = from_dict(EvalConfig, eval_dict)
    if args.max_days:
        cfg.sampling.max_days = args.max_days
    models = [m for m in spec["models"] if not args.models or m["name"] in args.models]
    out_root = Path("results") / spec["experiment"]

    for source, years in spec["sources"].items():
        if args.sources and source not in args.sources:
            continue
        start = time.perf_counter()
        days, pairs, info = load_eval_days(source, years, cfg)
        panel_models = [for_panel(m, source.split(":")[-1]) for m in models]
        daily, stability = run(
            panel_models, days, pairs, cfg, args.workers or spec.get("workers", 4)
        )
        summary = summarize(daily, stability)

        out = out_root / source.replace(":", "_")
        out.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out / "daily.csv", index=False)
        stability.to_csv(out / "stability.csv", index=False)
        summary.to_csv(out / "summary.csv")

        print(
            f"\n== {source} {list(years)}: {info['screened_days']} of {info['sampled_days']} "
            f"sampled days after screening, {len(pairs)} consecutive pairs, "
            f"{time.perf_counter() - start:.0f}s"
        )
        print(key_columns(summary).T.to_string())
        if "error" in daily and daily["error"].notna().any():
            print("errors:", daily.loc[daily["error"].notna(), ["model", "date"]].to_string())

        if not args.no_mlflow:
            for spec_m in panel_models:
                if spec_m["name"] not in summary.index:
                    continue
                path = out / f"daily_{spec_m['name']}.csv"
                daily[daily["model"] == spec_m["name"]].to_csv(path, index=False)
                log_mlflow(spec["experiment"], source, years, spec_m, eval_dict,
                           summary.loc[spec_m["name"]], path)  # fmt: skip


if __name__ == "__main__":
    main()

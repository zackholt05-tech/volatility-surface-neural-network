"""Train the cross-day conditional surface model; the checkpoint goes to artifacts/conditional/."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from volsurf.config import from_dict, load_yaml
from volsurf.eval.runner import flatten
from volsurf.training.conditional import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, help="override (smoke runs)")
    parser.add_argument("--max-days", type=int, help="cap training days (smoke runs)")
    parser.add_argument("--name", help="override the run name (checkpoint and MLflow run)")
    parser.add_argument("--latent-dim", type=int, help="override model.latent_dim")
    parser.add_argument("--threads", type=int, help="override torch threads")
    parser.add_argument("--out", help="checkpoint path (default: artifacts/conditional/<name>.pt)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    spec = load_yaml(args.config)
    name = args.name or spec.pop("name")
    spec.pop("name", None)
    if args.latent_dim:
        spec["model"] = {**spec.get("model", {}), "latent_dim": args.latent_dim}
    if args.threads:
        spec["threads"] = args.threads
    cfg = from_dict(TrainConfig, spec)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.max_days:
        cfg.max_days = args.max_days
    out = Path(args.out or f"artifacts/conditional/{name}.pt")

    net, best = train(cfg)
    net.save(out, net.meta)  # before logging, so a busy tracking store cannot lose the model
    if not args.no_mlflow:  # logged after training: parallel runs share one SQLite store
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("phase6-train")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(flatten(spec) | {"checkpoint": str(out)})
            for row in net.meta["history"]:
                mlflow.log_metrics({k: float(v) for k, v in row.items()}, step=row["epoch"])
            mlflow.log_metrics({f"best_{k}": float(v) for k, v in best.items()})
    print(f"best epoch {best['epoch']}: val loss {best['loss']:.4f}, "
          f"rmse {best['rmse_vp']:.3f} vp -> {out}")  # fmt: skip


if __name__ == "__main__":
    main()

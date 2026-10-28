"""Generate synthetic option-chain panels defined in a YAML config."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from volsurf.config import deep_merge, load_yaml
from volsurf.data.synthetic import SyntheticConfig, generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--only", nargs="*", help="dataset names to build (default: all)")
    parser.add_argument("--days", type=int, help="override n_days for quick runs")
    args = parser.parse_args()

    spec = load_yaml(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, overrides in spec["datasets"].items():
        if args.only and name not in args.only:
            continue
        cfg_dict = deep_merge(spec.get("defaults", {}), overrides or {})
        if args.days:
            cfg_dict["n_days"] = args.days
        cfg = SyntheticConfig.from_dict(cfg_dict)

        start = time.perf_counter()
        quotes, states = generate(cfg)
        quotes.to_parquet(out / f"{name}.parquet", index=False)
        states.to_parquet(out / f"{name}_states.parquet", index=False)

        n_days = quotes["date"].nunique()
        labels = quotes["label"].value_counts(normalize=True)
        print(
            f"{name:>13}: {len(quotes):,} quotes over {n_days} days "
            f"({len(quotes) / n_days:.0f}/day), "
            f"mispriced {1 - labels.get(0, 0.0):.2%}, "
            f"{time.perf_counter() - start:.1f}s -> {out / name}.parquet"
        )


if __name__ == "__main__":
    main()

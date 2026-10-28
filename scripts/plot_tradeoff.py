"""Fit-versus-arbitrage trade-off across penalty weights, with baselines on the same days."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

VIOLATION = ["arb_data/calendar_frac", "arb_data/butterfly_frac", "arb_ext/calendar_frac",
             "arb_ext/butterfly_frac"]  # fmt: skip


def lam(name: str) -> float | None:
    match = re.fullmatch(r"pnn_l(\d+(?:\.\d+)?)", name)
    return float(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/phase4-ablation")
    parser.add_argument("--source", default="spx")
    parser.add_argument("--baselines", nargs="*", default=["ssvi", "svi", "spline", "mlp"])
    args = parser.parse_args()

    source = args.source.replace(":", "_")
    summary = pd.read_csv(Path(args.results) / source / "summary.csv", index_col="model")
    summary["lambda"] = [lam(n) for n in summary.index]
    curve = summary.dropna(subset=["lambda"]).sort_values("lambda")
    base = summary.loc[[b for b in args.baselines if b in summary.index]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, domain in zip(axes, ["arb_data", "arb_ext"], strict=True):
        x = curve[f"{domain}/calendar_frac"] + curve[f"{domain}/butterfly_frac"]
        ax.plot(x, curve["interp/rms_norm"], "o-", color="C0", label="penalized network")
        for lam_value, xi, yi in zip(curve["lambda"], x, curve["interp/rms_norm"], strict=True):
            ax.annotate(f"λ={lam_value:g}", (xi, yi), textcoords="offset points", xytext=(4, 4),
                        fontsize=8)  # fmt: skip
        for name, row in base.iterrows():
            bx = row[f"{domain}/calendar_frac"] + row[f"{domain}/butterfly_frac"]
            ax.plot(bx, row["interp/rms_norm"], "s", label=name)
        ax.set(
            xlabel="calendar + butterfly violation rate",
            ylabel="held-out interior RMS (half-spreads)",
            title=f"{'quoted' if domain == 'arb_data' else 'extended'} domain",
        )
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlim(left=-1e-4)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Fit vs no-arbitrage trade-off ({args.source})")
    fig.tight_layout()
    out = Path(args.results) / source / "tradeoff.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()

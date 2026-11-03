"""Run the panel fits and mispricing reports on Modal CPU containers instead of this machine.

The scripts run unchanged: each container's working directory mirrors the repository, with the code
in the image and data/, results/ and artifacts/ linked to the `volsurf` Volume. Each call is one
script run in its own container, so panels fit side by side. Fill the Volume from the repository
first, launch detached so the run outlives this machine, and copy the outputs back after:
  bash scripts/modal_sync.sh up
  modal run --detach scripts/modal_run.py::warm_test
  bash scripts/modal_sync.sh down
Only the Python version and the locked dependencies are shared with the local environment: the
containers are Linux x86_64, so fits agree with local ones to rounding, not bit for bit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = "/root/repo"
LINKED = ("data", "results", "artifacts")

app = modal.App("volsurf")
volume = modal.Volume.from_name("volsurf", create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .env({"PYTHONPATH": f"{REPO}/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("src", f"{REPO}/src", ignore=["**/__pycache__"])
    .add_local_dir("scripts", f"{REPO}/scripts", ignore=["**/__pycache__"])
    .add_local_dir("configs", f"{REPO}/configs")
)
WORKERS = 15  # robust refits in parallel; the chain itself uses --threads


def _script(script: str, args: list[str], log: str) -> None:
    """Run scripts/<script> in the mirrored repository, copying its output to results/modal-logs."""
    os.chdir(REPO)
    for name in LINKED:
        Path("/vol", name).mkdir(parents=True, exist_ok=True)
        if not Path(name).exists():
            Path(name).symlink_to(f"/vol/{name}")
    logs = Path("results/modal-logs")
    logs.mkdir(exist_ok=True)
    cmd = [sys.executable, f"scripts/{script}", *args]
    with open(logs / f"{log}.log", "w") as out:
        out.write(" ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            out.write(line)
            out.flush()
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"{script} exited with {code}; see results/modal-logs/{log}.log")


# Billed on the larger of request and use: most of a run is the date-ordered chain on 4 threads,
# so request that and let the parallel refits burst to the limits.
@app.function(
    image=image, volumes={"/vol": volume}, cpu=(4.0, 20.0), memory=(8192, 32768),
    timeout=10 * 3600,
)  # fmt: skip
def fit(args: list[str], log: str) -> None:
    _script("fit_panel.py", args, log)


@app.function(image=image, volumes={"/vol": volume}, cpu=2, memory=16384, timeout=2 * 3600)
def report(args: list[str], log: str) -> None:
    _script("mispricing_report.py", args, log)


def _warm(config: str, out: str, source: str, years: list[int], seed: str, *extra: str):
    return [
        "--config", config, "--out", out, "--warm-start", "--threads", "4",
        "--workers", str(WORKERS), "--source", source, "--years", *map(str, years),
        "--seed-cache", seed, *extra,
    ]  # fmt: skip


@app.function(image=image, timeout=12 * 3600)
def warm_check() -> None:
    """The first six SPX 2022 validation days, to compare with the local validation fits."""
    args = _warm(
        "configs/phase7_warm.yaml", "results/modal-check", "spx", [2022],
        "results/phase7b-fits/spx/fits.pkl", "--max-days", "6",
    )  # fmt: skip
    args[args.index("--workers") + 1] = "3"
    fit.remote(args, "check_spx")


@app.function(image=image, timeout=12 * 3600)
def warm_test() -> None:
    """The test panels of configs/phase7_warm.yaml and configs/replication_warm.yaml."""
    phase7 = ("configs/phase7_warm.yaml", "results/phase7-warm-fits")
    cond = (
        "--checkpoint", "cond=artifacts/conditional/heston_d8.pt",
        "cond_multi=artifacts/conditional/heston_multi_d8.pt",
    )  # fmt: skip
    seeds = "results/phase7b-fits/{}/fits.pkl"
    panels = [
        fit.spawn(_warm(*phase7, "spx", [2023], seeds.format("spx")), "warmtest_spx"),
        fit.spawn(
            _warm(*phase7, "synthetic:heston", [2022], seeds.format("synthetic_heston"), *cond),
            "warmtest_heston",
        ),
        fit.spawn(
            _warm(
                *phase7,
                "synthetic:heston_fresh",
                [2023],
                seeds.format("synthetic_heston_fresh"),
                *cond,
            ),
            "warmtest_heston_fresh",
        ),
    ]
    replication = fit.spawn(
        _warm(
            "configs/replication_warm.yaml",
            "results/replication-warm-fits",
            "spx",
            [2016, 2017],
            "results/replication-fits/spx/fits.pkl",
        ),
        "repwarm_spx",
    )
    for call in panels:
        call.get()
    reports = [
        report.spawn(["--config", "configs/phase7_warm.yaml", "--split", "test"], "warmtest_report")
    ]
    replication.get()
    reports.append(
        report.spawn(
            ["--config", "configs/replication_warm.yaml", "--split", "test"], "repwarm_report"
        )
    )
    for call in reports:
        call.get()

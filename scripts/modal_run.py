"""Run the panel fits and mispricing reports on Modal CPU containers instead of this machine.

The scripts run unchanged: each container's working directory mirrors the repository, with the code
in the image and data/, results/ and artifacts/ linked to the `volsurf` Volume. Each call is one
script run in its own container, so panels fit side by side. Fill the Volume from the repository,
deploy the app and spawn the fits on it, so the calls outlive this machine; once all are done, run
the reports and copy the outputs back:
  bash scripts/modal_sync.sh up
  modal deploy scripts/modal_run.py
  modal run scripts/modal_run.py::launch    # spawns the test-panel fits and returns
  modal run scripts/modal_run.py::status    # running, done or failed, by panel
  modal run scripts/modal_run.py::reports
  bash scripts/modal_sync.sh down
A fit that Modal preempts restarts on the same input and resumes from the checkpoints
scripts/fit_panel.py saves each day. Only the Python version and the locked dependencies are shared
with the local environment: the containers are Linux x86_64, so fits agree with local ones to
rounding, not bit for bit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = "/root/repo"
LINKED = ("data", "results", "artifacts")
CALLS = Path("results/modal-calls.json")  # local: the spawned fits' call ids, by panel

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
    # appended, so the log of a preempted and restarted call keeps both attempts
    with open(logs / f"{log}.log", "a") as out:
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


PHASE7 = ("configs/phase7_warm.yaml", "results/phase7-warm-fits")
COND = (
    "--checkpoint", "cond=artifacts/conditional/heston_d8.pt",
    "cond_multi=artifacts/conditional/heston_multi_d8.pt",
)  # fmt: skip
SEED = "results/phase7b-fits/{}/fits.pkl"
# The test panels of configs/phase7_warm.yaml and configs/replication_warm.yaml, by log name.
PANELS = {
    "warmtest_spx": _warm(*PHASE7, "spx", [2023], SEED.format("spx")),
    "warmtest_heston": _warm(
        *PHASE7, "synthetic:heston", [2022], SEED.format("synthetic_heston"), *COND
    ),
    "warmtest_heston_fresh": _warm(
        *PHASE7, "synthetic:heston_fresh", [2023], SEED.format("synthetic_heston_fresh"), *COND
    ),
    "repwarm_spx": _warm(
        "configs/replication_warm.yaml", "results/replication-warm-fits", "spx", [2016, 2017],
        "results/replication-fits/spx/fits.pkl",
    ),
}  # fmt: skip
REPORTS = {
    "warmtest_report": ["--config", "configs/phase7_warm.yaml", "--split", "test"],
    "repwarm_report": ["--config", "configs/replication_warm.yaml", "--split", "test"],
}


@app.local_entrypoint()
def check() -> None:
    """The first six SPX 2022 validation days, to compare with the local validation fits."""
    args = _warm(
        "configs/phase7_warm.yaml", "results/modal-check", "spx", [2022],
        "results/phase7b-fits/spx/fits.pkl", "--max-days", "6",
    )  # fmt: skip
    args[args.index("--workers") + 1] = "3"
    fit.remote(args, "check_spx")


@app.local_entrypoint()
def launch(only: str = "") -> None:
    """Spawn the test-panel fits (``only``: a comma-separated subset of the log names) on the
    deployed app, and save their call ids."""
    deployed = modal.Function.from_name(app.name, "fit")
    names = only.split(",") if only else list(PANELS)
    calls = json.loads(CALLS.read_text()) if CALLS.exists() else {}
    for name in names:
        calls[name] = deployed.spawn(PANELS[name], name).object_id
        print(name, calls[name])
    CALLS.parent.mkdir(exist_ok=True)
    CALLS.write_text(json.dumps(calls, indent=1))


@app.local_entrypoint()
def status() -> None:
    """Whether each spawned fit is running, done or failed."""
    for name, call_id in json.loads(CALLS.read_text()).items():
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
            state = "done"
        except TimeoutError:
            state = "running"
        except Exception as error:  # the fit's own exception, re-raised here
            state = f"failed: {error}"
        print(f"{name:24s} {state}")


@app.local_entrypoint()
def reports() -> None:
    """The mispricing reports of the test panels, once every fit is done."""
    for call in [report.spawn(args, name) for name, args in REPORTS.items()]:
        call.get()

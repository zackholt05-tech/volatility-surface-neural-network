#!/bin/bash
# Copy the inputs of the warm-start test panels into the `volsurf` Modal Volume (up), or the
# outputs back into the repository (down). See scripts/modal_run.py.
#   bash scripts/modal_sync.sh up | down
set -euo pipefail
cd "$(dirname "$0")/.."
V=volsurf

put() { modal volume put -f "$V" "$1" "$1" > /dev/null && echo "up   $1"; }
# A folder is downloaded beside the repository and copied over its local copy, which keeps the local
# files the Volume lacks (a forced download in place replaces the whole folder).
get() {
  local tmp
  tmp=$(mktemp -d)
  if ! { modal volume get "$V" "$1" "$tmp" > /dev/null && mkdir -p "$1" \
         && cp -R "$tmp/$(basename "$1")/." "$1/"; }; then
    rm -rf "$tmp"
    echo "could not download $1" >&2
    return 1
  fi
  rm -rf "$tmp"
  echo "down $1"
}

case "${1:-}" in
  up)
    modal volume create --version 2 "$V" > /dev/null 2>&1 || true
    put data/processed/spx/days
    put data/processed/spx/forwards
    put data/processed/spx/summary.csv
    for f in data/processed/spx/quotes/{2016,2017,2022,2023}*.parquet; do put "$f"; done
    for f in heston heston_states heston_fresh heston_fresh_states; do put "data/synthetic/$f.parquet"; done
    for p in spx synthetic_heston synthetic_heston_fresh; do put "results/phase7b-fits/$p/fits.pkl"; done
    put results/replication-fits/spx/fits.pkl
    # the chains of the test years continue from these validation fits
    put results/phase7-warm-fits/spx/fits.pkl
    put results/phase7-warm-fits/spx/2022.parquet
    put results/phase7-warm-fits/synthetic_heston/fits.pkl
    put results/phase7-warm-fits/synthetic_heston/2021.parquet
    put artifacts/conditional/heston_d8.pt
    put artifacts/conditional/heston_multi_d8.pt
    ;;
  down)
    for d in results/phase7-warm-fits results/phase7-warm-test results/replication-warm-fits \
             results/replication-warm-test results/modal-logs; do
      get "$d"
    done
    ;;
  *)
    echo "usage: $0 up | down" >&2
    exit 1
    ;;
esac

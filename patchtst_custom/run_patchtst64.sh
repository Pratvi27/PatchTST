#!/usr/bin/env bash
# Run PatchTST under the patchtst64 preset (seq_len=512) across the four
# standard forecast horizons {96, 192, 336, 720}. Each horizon goes into a
# fresh timestamped subdirectory under ./result/, so reruns never overwrite.
#
# Usage:
#   ./run_patchtst64.sh                                  # default csv path
#   ./run_patchtst64.sh /path/to/traffic.csv             # custom csv path
#   ./run_patchtst64.sh /path/to/traffic.csv "96 192"    # subset of horizons

set -euo pipefail

# Resolve script location so relative paths work regardless of cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

CSV_PATH="${1:-/model/patchTST/all_six_datasets/traffic/traffic.csv}"
HORIZONS="${2:-96 192 336 720}"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-2021}"
CONFIG="${CONFIG:-patchtst64}"

TS="$(date +%Y%m%d-%H%M%S)"
SUMMARY_DIR="./result/patchtst64_${TS}"
mkdir -p "$SUMMARY_DIR"

echo "============================================================"
echo " PatchTST64 sweep"
echo "------------------------------------------------------------"
echo "  csv_path : $CSV_PATH"
echo "  horizons : $HORIZONS"
echo "  config   : $CONFIG"
echo "  seed     : $SEED"
echo "  outputs  : $SUMMARY_DIR/h{H}/"
echo "============================================================"

if [[ ! -f "$CSV_PATH" ]]; then
    echo "ERROR: csv not found at $CSV_PATH" >&2
    exit 1
fi

for H in $HORIZONS; do
    RUN_DIR="${SUMMARY_DIR}/h${H}"
    echo
    echo ">>> [horizon=${H}] -> ${RUN_DIR}"
    "$PYTHON" main.py \
        --csv_path "$CSV_PATH" \
        --config "$CONFIG" \
        --forecast_len "$H" \
        --result_dir "$RUN_DIR" \
        --seed "$SEED"
done

echo
echo "All horizons finished. Summary directory: $SUMMARY_DIR"

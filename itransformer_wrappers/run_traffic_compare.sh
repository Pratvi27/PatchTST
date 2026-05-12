#!/usr/bin/env bash
# Run iTransformer on Traffic across 4 horizons {96, 192, 336, 720}, aligned
# with our PatchTST reproduction:
#   - train_epochs = 20            (official default is 10)
#   - same csv as PatchTST         (symlinked into iTransformer/dataset/traffic/)
#   - itr = 1                      (single run)
#
# Other hyperparameters follow scripts/multivariate_forecasting/Traffic/iTransformer.sh:
#   d_model=512, d_ff=512, e_layers=4, batch_size=16, lr=0.001
#
# NOTE: Traffic uses e_layers=4 (one more than Weather/ECL). Don't accidentally
# change this — it's a per-dataset hyperparameter chosen by the iTransformer authors.
#
# This script does NOT modify any official file. Drop-in addition only.
#
# Usage:
#   bash run_traffic_compare.sh                  # all 4 horizons
#   bash run_traffic_compare.sh "96 192"         # subset of horizons
#
# Env overrides:
#   CSV_PATH    absolute csv path (default: PatchTST notebook path)
#   EPOCHS      train_epochs (default: 20)
#   SEQ_LEN     lookback (default: 96)
#   GPU         CUDA_VISIBLE_DEVICES (default: 0)
#   PYTHON      python interpreter

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# ─── Config ────────────────────────────────────────────────────────────
CSV_PATH="${CSV_PATH:-/model/patchTST/all_six_datasets/traffic/traffic.csv}"
EPOCHS="${EPOCHS:-20}"
SEQ_LEN="${SEQ_LEN:-96}"
HORIZONS="${1:-96 192 336 720}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"

# Traffic has 862 variables and uses one extra encoder layer.
VARS=862
E_LAYERS=4
BATCH_SIZE=16
LR=0.001

export CUDA_VISIBLE_DEVICES="$GPU"

# ─── Symlink csv into the location iTransformer expects ────────────────
mkdir -p "${SCRIPT_DIR}/dataset/traffic"
if [[ ! -f "$CSV_PATH" ]]; then
    echo "ERROR: csv not found at $CSV_PATH" >&2
    echo "       Override with: CSV_PATH=/your/path bash $0" >&2
    exit 1
fi
ln -sfn "$CSV_PATH" "${SCRIPT_DIR}/dataset/traffic/traffic.csv"
echo "Symlinked: ${SCRIPT_DIR}/dataset/traffic/traffic.csv -> $CSV_PATH"

# ─── Output dir ────────────────────────────────────────────────────────
TS=$(date +%Y%m%d-%H%M%S)
SUMMARY_DIR="${SCRIPT_DIR}/result/traffic_e${EPOCHS}_${TS}"
mkdir -p "$SUMMARY_DIR"

echo "============================================================"
echo " iTransformer Traffic sweep (aligned with PatchTST settings)"
echo "------------------------------------------------------------"
echo "  csv         : $CSV_PATH"
echo "  epochs      : $EPOCHS"
echo "  seq_len     : $SEQ_LEN"
echo "  horizons    : $HORIZONS"
echo "  variables   : $VARS"
echo "  e_layers    : $E_LAYERS  (note: +1 vs Weather/ECL)"
echo "  batch_size  : $BATCH_SIZE"
echo "  lr          : $LR"
echo "  GPU         : $GPU"
echo "  outputs     : $SUMMARY_DIR/h{H}/"
echo "============================================================"

for H in $HORIZONS; do
    RUN_DIR="${SUMMARY_DIR}/h${H}"
    mkdir -p "$RUN_DIR"
    LOG_FILE="${RUN_DIR}/run.log"

    MODEL_ID="traffic_${SEQ_LEN}_${H}"
    DES="Exp_h${H}_e${EPOCHS}"

    echo
    echo ">>> [horizon=${H}] -> ${RUN_DIR}"

    "$PYTHON" -u run.py \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id "$MODEL_ID" \
        --model iTransformer \
        --data custom \
        --features M \
        --seq_len "$SEQ_LEN" \
        --pred_len "$H" \
        --e_layers $E_LAYERS \
        --enc_in $VARS \
        --dec_in $VARS \
        --c_out $VARS \
        --d_model 512 \
        --d_ff 512 \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --train_epochs "$EPOCHS" \
        --itr 1 \
        --des "$DES" \
        2>&1 | tee "$LOG_FILE"

    # Parse the metric line iTransformer prints at the end of test():
    #   "mse:0.395123, mae:0.268312"
    METRIC_LINE=$(grep -E '^mse:[0-9]' "$LOG_FILE" | tail -n 1 || true)
    if [[ -z "$METRIC_LINE" ]]; then
        echo "WARNING: no mse/mae line found in $LOG_FILE"
        MSE="null"
        MAE="null"
    else
        MSE=$(echo "$METRIC_LINE" | sed -E 's/mse:([0-9.]+),.*/\1/')
        MAE=$(echo "$METRIC_LINE" | sed -E 's/.*mae:([0-9.]+).*/\1/')
    fi

    cat > "${RUN_DIR}/results.json" <<EOF
{
  "model": "iTransformer",
  "dataset": "traffic",
  "seq_len": ${SEQ_LEN},
  "pred_len": ${H},
  "epochs": ${EPOCHS},
  "d_model": 512,
  "d_ff": 512,
  "e_layers": ${E_LAYERS},
  "batch_size": ${BATCH_SIZE},
  "learning_rate": ${LR},
  "test_mse": ${MSE},
  "test_mae": ${MAE}
}
EOF
    echo "Wrote ${RUN_DIR}/results.json  (mse=${MSE}, mae=${MAE})"
done

# ─── Aggregate per-horizon results into one summary.json ───────────────
"$PYTHON" - <<PYEOF
import glob, json, os, re
summary_dir = r"${SUMMARY_DIR}"
out = {}
for d in sorted(glob.glob(os.path.join(summary_dir, "h*"))):
    name = os.path.basename(d)
    if not re.fullmatch(r"h\d+", name):
        continue
    fp = os.path.join(d, "results.json")
    if os.path.isfile(fp):
        with open(fp) as f:
            out[name] = json.load(f)
with open(os.path.join(summary_dir, "summary.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSummary written to {summary_dir}/summary.json")
print(json.dumps(out, indent=2))
PYEOF

echo
echo "Done. All artifacts under: $SUMMARY_DIR"

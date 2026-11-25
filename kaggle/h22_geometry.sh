#!/bin/bash
# Taxonomy geometry of each encoder on H22-clean (PLAN_ANALYSIS.md §4.1) on a Kaggle CPU session (no GPU quota):
# tools/feature_geometry.py on the train features of h22_extract, one encoder after another.
# Output: results/geometry/h22clean_<enc>_geometry{,_shuf0}_class.csv and _geometry.json, logs/geometry_<enc>.log.
# The per-class link to C2 (tools/c2_class_analysis.py) then runs locally on the saved predictions.
set -eo pipefail
ENCODERS=${ENCODERS:-dinov2_b14,bioclip2,clip_b32}
MAX_EVAL=${MAX_EVAL:-2000}
INPUT=${INPUT:-/kaggle/input}
WORK=${WORK:-/kaggle/working}
CODE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1
cd "$WORK"
mkdir -p results/geometry logs configs
git -C "$CODE" log --oneline -1 || true
echo "CPUs: $(nproc)  RAM: $(free -g | awk '/Mem/{print $2}') GB"
FEAT=""
for m in $(find "$INPUT" -maxdepth 8 -path "*/h22clean/train/*/manifest.json" -print 2>/dev/null); do
  FEAT="${m%/train/*/manifest.json}"; break
done
[ -n "$FEAT" ] || { echo "No H22 features under $INPUT: add Input > Your Work > h22_extract"; exit 1; }
echo "features: $FEAT"
rc=0
for enc in $(echo "$ENCODERS" | tr ',' ' '); do
  echo; echo "=== [$(date +%H:%M:%S)] $enc"
  python "$CODE/tools/feature_geometry.py" --config "$CODE/configs/h22clean_${enc}.json" --feature_root "$FEAT" \
    --taxonomy "$CODE/taxonomy/h22.csv" --c2_config "$CODE/results/configs/h22clean_${enc}_tau_norm+c2f.json" \
    --out results/geometry --max_eval "$MAX_EVAL" --threads "$(nproc)" 2>&1 | tee "logs/geometry_${enc}.log" || rc=1
done
echo; echo "=== [$(date +%H:%M:%S)] DONE (exit $rc)"; ls -la results/geometry
exit $rc

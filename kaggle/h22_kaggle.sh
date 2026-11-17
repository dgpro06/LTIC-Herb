#!/bin/bash
# H22-clean (Herbarium 2022, leakage-free split, plan §5.2) on Kaggle. Guide: KAGGLE_H22.md.
#
#   split + V-parts -> chunked features (optionally one PART of the chunks per Kaggle account) -> optional baselines
#
# Several accounts can extract at the same time: account K of N sets PART="K/N" and extracts every N-th chunk.
# The split files are deterministic, so every account builds identical ones (their sha1 is printed: compare).
# Afterwards ONE notebook attaches every account's Output (PREV=auto copies all of them) and has all chunks.
set -eo pipefail

ENCODERS=${ENCODERS:-dinov2_b14,bioclip2,clip_b32}
PART=${PART:-0/1}                           # K/N: this account extracts chunks i with i % N == K
DEADLINE_MIN=${DEADLINE_MIN:-640}           # stop starting new chunks after ~10.7 h (Kaggle kills at 12 h)
CHUNK=25000                                 # must be the same on every account (it is checked)
BASELINES=${BASELINES:-}
BASELINE_ENCODERS=${BASELINE_ENCODERS:-$ENCODERS}
TRIALS=${TRIALS:-10}
SEEDS=${SEEDS:-0,1,2}
BATCH=${BATCH:-128}
WORKERS=${WORKERS:-4}
INPUT=${INPUT:-/kaggle/input}
WORK=${WORK:-/kaggle/working}
PREV=${PREV:-}                              # "auto" = copy every attached earlier H22 output
SKIP_INSTALL=${SKIP_INSTALL:-0}

CODE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1
mkdir -p "$WORK"
cd "$WORK"
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

stamp "STEP 1  GPU and code"
if ! nvidia-smi --query-gpu=name,memory.total --format=csv; then
  echo "No GPU. Notebook sidebar: Session options > Accelerator > GPU T4 x2, then run again."; exit 1
fi
git -C "$CODE" log --oneline -1 || true
if [ "$SKIP_INSTALL" != 1 ]; then pip install -q --upgrade open_clip_torch; fi

# prune the image folders: H22 has ~1M files and walking them takes minutes
PRUNE=( \( -name images -o -name train_images -o -name test_images \) -prune -o )
META=$(find "$INPUT" -maxdepth 4 "${PRUNE[@]}" -type f -name train_metadata.json -print 2>/dev/null | head -1)
if [ -z "$META" ]; then
  echo "train_metadata.json not found under $INPUT. Join the competition (kaggle.com/competitions/herbarium-2022-fgvc9"
  echo "> Join Competition / I Understand and Accept), then Input > + Add Input > Competitions > Herbarium 2022."; exit 1
fi
H22_ROOT=$(dirname "$META")
echo "H22 data: $H22_ROOT"
ls "$H22_ROOT" | head

if [ "$PREV" = auto ]; then
  # every attached earlier output (this or another account) that holds H22 splits or features
  for hit in $(find "$INPUT" -maxdepth 7 "${PRUNE[@]}" -type f -path "*/splits/h22clean_train.txt" -print 2>/dev/null); do
    d="${hit%/splits/h22clean_train.txt}"
    stamp "Copying earlier output $d"
    for sub in splits taxonomy feats results; do
      if [ -d "$d/$sub" ]; then mkdir -p "$WORK/$sub"; cp -rn "$d/$sub/." "$WORK/$sub/"; echo "copied $sub"; fi
    done
  done
fi

stamp "STEP 2  H22-clean split (plan 5.2) and V-tune / V-select / V-cal"
if [ ! -f splits/h22clean_vcal.txt ]; then
  python "$CODE/tools/make_h22_clean.py" --metadata "$META" --out splits --path_prefix train_images/ \
    --taxonomy_out taxonomy/h22.csv
  python "$CODE/tools/make_val_parts.py" --train splits/h22clean_train.txt --val splits/h22clean_val.txt \
    --out splits --prefix h22clean --few_thr 20 --auto_ratios
else
  echo "splits already there"
fi
echo "split-file sha1 (must be identical on every account):"
sha1sum splits/h22clean_{train,vtune,vselect,vcal,test}.txt | sed 's/^/  /'
wc -l splits/h22clean_{train,vtune,vselect,vcal,test}.txt | sed 's/^/  /'

stamp "STEP 3  Features: $ENCODERS, part $PART (chunked, resumable, stops before ${DEADLINE_MIN} min)"
python "$CODE/tools/extract_features.py" --root "$H22_ROOT" --out feats/h22clean --encoders "$ENCODERS" \
  --batch "$BATCH" --workers "$WORKERS" --chunk "$CHUNK" --part "$PART" --deadline_min "$DEADLINE_MIN" \
  --split train=splits/h22clean_train.txt --split vtune=splits/h22clean_vtune.txt \
  --split vselect=splits/h22clean_vselect.txt --split vcal=splits/h22clean_vcal.txt \
  --split test=splits/h22clean_test.txt

stamp "Feature store status (all parts together)"
COMPLETE=$(python - "$ENCODERS" <<'EOF'
import json, os, sys
ok = True
for enc in sys.argv[1].split(','):
    for split in ('train', 'vtune', 'vselect', 'vcal', 'test'):
        d = os.path.join('feats/h22clean', split, enc)
        try:
            m = json.load(open(os.path.join(d, 'manifest.json'), encoding='utf-8'))
        except FileNotFoundError:
            print('  {:10s} {:8s} not started'.format(enc, split), file=sys.stderr); ok = False; continue
        have = {int(f[6:11]) for f in os.listdir(d) if f.startswith('chunk_') and f.endswith('.pt')}
        miss = sorted(set(range(m['num_chunks'])) - have)
        ok &= not miss
        print('  {:10s} {:8s} {}/{} chunks{}'.format(enc, split, len(have), m['num_chunks'],
              '' if not miss else '  missing: {}'.format(miss[:12])), file=sys.stderr)
print(1 if ok else 0)
EOF
)
[ "$COMPLETE" = 1 ] && echo "ALL CHUNKS PRESENT" || echo "NOT COMPLETE: run again (same PART), or attach the other accounts' outputs"

mkdir -p configs
for enc in $(echo "$ENCODERS,$BASELINE_ENCODERS" | tr ',' ' ' | xargs -n1 | sort -u); do
  python - "$CODE/configs/h22clean_clip_b32.json" "configs/h22clean_${enc}.json" "$enc" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1], encoding='utf-8')); c['encoder'] = sys.argv[3]
c['num_classes'] = len(json.load(open('splits/h22clean_label_map.json', encoding='utf-8')))   # 15,501
json.dump(c, open(sys.argv[2], 'w', encoding='utf-8'), indent=1)
EOF
done

if [ -n "$BASELINES" ]; then
  if [ "$COMPLETE" != 1 ]; then
    echo "Baselines skipped: the features are not complete yet."
  else
    for enc in $(echo "$BASELINE_ENCODERS" | tr ',' ' '); do
      stamp "Baselines on H22-clean / $enc: $BASELINES"
      python "$CODE/tools/run_baselines.py" --config "configs/h22clean_${enc}.json" --methods "$BASELINES" \
        --trials "$TRIALS" --seeds "$SEEDS" --skip_done
    done
  fi
fi

stamp "DONE"
du -sh feats splits taxonomy results 2>/dev/null || true

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/4] Required files check...\n'
required=(
  models/training_dataset_cache_v2.npz
  models/ensemble_config.pkl
  models/val_pred_candidate_v8.npz
  models/val_pred_pca_xgb.npz
  models/preprocess_pca_xgb.pkl
)
missing=0
for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

printf '\n[2/4] Syntax check...\n'
python -m py_compile train_v15_safe_exact.py v16_stress_safe_exact.py

printf '\n[3/4] Exact V8 integration with safe alphas...\n'
python train_v15_safe_exact.py | tee logs/v15_safe_exact_prepare.log

python - <<'PY'
import pickle
from pathlib import Path
p = Path('models/val_pred_candidate_v15.pkl')
if not p.exists():
    raise SystemExit('missing preparation artifact')
with open(p, 'rb') as f:
    d = pickle.load(f)
if not bool(d.get('passed', False)):
    raise SystemExit('safe exact integration gate did not pass; stop here')
print('safe exact integration PASS; continue to final stress')
PY

printf '\n[4/4] Re-check stress with exact-safe alphas...\n'
python v16_stress_safe_exact.py | tee logs/v16_safe_exact.log

printf '\nDone. Logs:\n'
printf '  logs/v15_safe_exact_prepare.log\n'
printf '  logs/v16_safe_exact.log\n'
printf '\nNothing is activated automatically. Online V8 remains unchanged.\n'

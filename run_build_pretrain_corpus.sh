#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/4] Check processed external data...\n'
[ -d external_data/processed/kaist_runs ] || { echo 'missing external_data/processed/kaist_runs'; exit 1; }
[ -f external_data/processed/metropt_core.csv ] || { echo 'missing external_data/processed/metropt_core.csv'; exit 1; }

printf '\n[2/4] Syntax/import check...\n'
python -m py_compile build_pretrain_corpus.py src/deep/corpus_builder.py src/deep/__init__.py
python - <<'PY'
from src.deep import DeepPretrainCorpusBuilder
print('deep corpus imports OK')
PY

printf '\n[3/4] Build masked 512->96 pretraining corpus...\n'
START=$(date +%s)
python build_pretrain_corpus.py --kaist-windows 8 --metro-windows 12 --min-group-rows 128 | tee logs/pretrain_corpus_build.log
END=$(date +%s)
printf '\ncorpus build elapsed: %d seconds\n' "$((END-START))"

printf '\n[4/4] Validate corpus integrity and group split...\n'
python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('.')
corpus_dir = root / 'external_data' / 'corpus'
pretrain_path = corpus_dir / 'pretrain_corpus_v1.npz'
official_path = corpus_dir / 'official_finetune_v1.npz'
manifest_path = corpus_dir / 'pretrain_manifest_v1.json'

for p in (pretrain_path, official_path, manifest_path):
    if not p.is_file():
        raise SystemExit(f'missing: {p}')

z = np.load(pretrain_path, allow_pickle=False)
X = z['X']
Y = z['Y']
mask = z['mask']
split = z['split']
group = z['group_id']
source = z['source']
targets = z['targets']

print('X shape           :', X.shape)
print('Y shape           :', Y.shape)
print('mask shape        :', mask.shape)
print('groups            :', len(np.unique(group)))
print('sources           :', {str(s): int((source == s).sum()) for s in np.unique(source)})
print('split samples     :', {s: int((split == s).sum()) for s in ['train','val','test']})

if X.ndim != 3 or X.shape[1:] != (512, 6):
    raise SystemExit(f'bad X shape: {X.shape}')
if Y.ndim != 3 or Y.shape[1:] != (96, 6):
    raise SystemExit(f'bad Y shape: {Y.shape}')
if mask.shape != (len(X), 6):
    raise SystemExit(f'bad mask shape: {mask.shape}')
if not np.isfinite(X).all() or not np.isfinite(Y).all():
    raise SystemExit('X/Y contains non-finite values')
if not np.isin(mask, [0.0, 1.0]).all():
    raise SystemExit('mask is not binary')

# One physical group must belong to exactly one split: no train/val leakage across overlapping windows.
for gid in np.unique(group):
    gs = np.unique(split[group == gid])
    if len(gs) != 1:
        raise SystemExit(f'group leakage: {gid} -> {gs.tolist()}')

for s in ['train','val','test']:
    if (split == s).sum() == 0:
        raise SystemExit(f'empty split: {s}')

print('modality samples  :')
for j, name in enumerate(targets.tolist()):
    print(f'  {name:16s}: {int((mask[:,j] > 0.5).sum())}')

of = np.load(official_path, allow_pickle=False)
print('official X shape  :', of['X'].shape)
print('official Y shape  :', of['Y'].shape)
if of['X'].shape[1:] != (512, 6) or of['Y'].shape[1:] != (96, 6):
    raise SystemExit('official finetune shape invalid')
if len(of['X']) != 5:
    raise SystemExit(f'expected 5 official example sequences, got {len(of["X"])}')
if not np.all(of['mask'] == 1.0):
    raise SystemExit('official finetune should contain all six modalities')

manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
print('manifest samples  :', manifest.get('samples'))
print('PRETRAIN CORPUS VALIDATION: PASS')
PY

printf '\nDone. Log: logs/pretrain_corpus_build.log\n'
printf 'NOTE: online V8/API/callback files were not modified.\n'

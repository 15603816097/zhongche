from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.deep import DeepPretrainCorpusBuilder


ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "external_data" / "corpus"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kaist-windows", type=int, default=8)
    parser.add_argument("--metro-windows", type=int, default=12)
    parser.add_argument("--min-group-rows", type=int, default=128)
    args = parser.parse_args()

    builder = DeepPretrainCorpusBuilder(
        ROOT,
        seed=args.seed,
        kaist_windows_per_group=args.kaist_windows,
        metro_windows_per_group=args.metro_windows,
        min_group_rows=args.min_group_rows,
    )

    pretrain_path = CORPUS_DIR / "pretrain_corpus_v1.npz"
    official_path = CORPUS_DIR / "official_finetune_v1.npz"
    manifest_path = CORPUS_DIR / "pretrain_manifest_v1.json"

    print("=" * 100)
    print("BUILD DEEP PRETRAIN CORPUS")
    print("=" * 100)
    print(f"root               : {ROOT}")
    print(f"kaist windows/group: {args.kaist_windows}")
    print(f"metro windows/group: {args.metro_windows}")
    print(f"min group rows     : {args.min_group_rows}")

    summary = builder.save(pretrain_path, official_path, manifest_path)

    print("\n" + "=" * 100)
    print("CORPUS BUILD COMPLETE")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"pretrain npz       : {pretrain_path}")
    print(f"official finetune  : {official_path}")
    print(f"manifest           : {manifest_path}")
    print("NOTE: no online V8/API/callback files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

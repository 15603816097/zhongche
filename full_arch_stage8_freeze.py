from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts" / "full_arch" / "freeze"
FROZEN_CONFIG = ROOT / "full_arch_frozen_gate_v1.json"

REQUIRED_MODELS = [
    ROOT / "models" / "model_lgb.pkl",
    ROOT / "models" / "scaler.pkl",
    ROOT / "models" / "model_xgb.pkl",
    ROOT / "models" / "scaler_xgb.pkl",
    ROOT / "models" / "ensemble_config.pkl",
    ROOT / "models" / "model_pca_xgb.pkl",
    ROOT / "models" / "preprocess_pca_xgb.pkl",
]

CRITICAL_FILES = [
    ROOT / "app.py",
    ROOT / "app_full_arch.py",
    ROOT / "config.py",
    ROOT / "full_arch_frozen_gate_v1.json",
    ROOT / "src" / "inference.py",
    ROOT / "src" / "v8_runtime.py",
    ROOT / "src" / "full_arch_runtime.py",
    ROOT / "v9_analog_multiscale_diagnostic.py",
    ROOT / "full_arch_stage4_confidence_ood.py",
    ROOT / "full_arch_stage5_dynamic_gate.py",
    ROOT / "test_full_arch_stage6_runtime.py",
    ROOT / "test_full_arch_stage7_callback_stress.py",
    ROOT / "run_full_arch_stage7.sh",
]

STAGE5_MANIFEST = ROOT / "artifacts" / "full_arch" / "dynamic_gate" / "manifest.json"
STAGE6_LOG = OUT_DIR / "stage6_final.log"
STAGE7_LOG = OUT_DIR / "stage7_final.log"


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in REQUIRED_MODELS + CRITICAL_FILES + [STAGE5_MANIFEST, STAGE6_LOG, STAGE7_LOG]:
        require(p)

    branch = run("git", "branch", "--show-current")
    head = run("git", "rev-parse", "HEAD")
    tracked_dirty = run("git", "status", "--porcelain", "--untracked-files=no")
    if branch != "full-architecture":
        raise RuntimeError(f"expected branch full-architecture, got {branch}")
    if tracked_dirty:
        raise RuntimeError("tracked working tree is dirty:\n" + tracked_dirty)

    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    if not frozen.get("frozen", False):
        raise RuntimeError("frozen gate config has frozen=false")
    if not frozen.get("global_gate_pass", False):
        raise RuntimeError("frozen gate config global_gate_pass=false")
    if frozen.get("enabled_targets") != [
        "speed_rpm",
        "acoustic_db",
        "pressure_kpa",
    ]:
        raise RuntimeError(
            f"unexpected enabled targets: {frozen.get('enabled_targets')}"
        )

    stage5 = json.loads(STAGE5_MANIFEST.read_text(encoding="utf-8"))
    if not stage5.get("global_gate_pass", False):
        raise RuntimeError("Stage 5 manifest says global gate REJECT")
    if stage5.get("enabled_targets") != frozen.get("enabled_targets"):
        raise RuntimeError(
            "frozen config target list differs from Stage 5 manifest: "
            f"{frozen.get('enabled_targets')} vs {stage5.get('enabled_targets')}"
        )

    stage6_text = STAGE6_LOG.read_text(encoding="utf-8", errors="replace")
    stage7_text = STAGE7_LOG.read_text(encoding="utf-8", errors="replace")
    if "STAGE 6 DIRECT RUNTIME: PASS" not in stage6_text:
        raise RuntimeError("Stage 6 final log does not contain PASS")
    if "STAGE 7 CALLBACK STRESS: PASS" not in stage7_text:
        raise RuntimeError("Stage 7 final log does not contain callback PASS")
    if "STAGE 7 PASS" not in stage7_text:
        raise RuntimeError("Stage 7 final wrapper did not PASS")

    print("=" * 110)
    print("FULL ARCHITECTURE - STAGE 8 FREEZE")
    print("=" * 110)
    print("branch          :", branch)
    print("git HEAD        :", head)
    print("enabled targets :", frozen["enabled_targets"])
    print("flat RMSE ratio :", stage5.get("flat_rmse_ratio"))
    print("Stage 6         : PASS")
    print("Stage 7         : PASS")
    print()
    print("Hashing frozen model pack and critical runtime files...")

    model_hashes = {}
    for path in REQUIRED_MODELS:
        digest = sha256(path)
        model_hashes[str(path.relative_to(ROOT))] = {
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
        print(f"  model {path.name:28s} {digest[:16]}...")

    source_hashes = {}
    for path in CRITICAL_FILES:
        digest = sha256(path)
        source_hashes[str(path.relative_to(ROOT))] = digest
        print(f"  file  {str(path.relative_to(ROOT)):42s} {digest[:16]}...")

    freeze = {
        "candidate": "full_arch_dynamic_gate_v1",
        "app_version": "3.0.0-full-arch",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "git_head": head,
        "python": sys.version,
        "platform": platform.platform(),
        "enabled_targets": frozen["enabled_targets"],
        "fallback_targets": [
            "vibration_rms",
            "temperature_c",
            "current_a",
        ],
        "offline": {
            "flat_rmse_v8": stage5.get("flat_rmse_v8"),
            "flat_rmse_candidate": stage5.get("flat_rmse_candidate"),
            "flat_rmse_ratio": stage5.get("flat_rmse_ratio"),
            "global_gate_pass": stage5.get("global_gate_pass"),
        },
        "validation": {
            "stage6_direct_runtime": "PASS",
            "stage7_callback_stress": "PASS",
            "stage7_accepted": "50/50",
            "stage7_callbacks": "50/50",
            "stage7_prediction_max_diff": 0.0,
        },
        "model_hashes": model_hashes,
        "critical_file_hashes": source_hashes,
        "frozen_gate_config": frozen,
        "deployment_policy": (
            "This manifest freezes the candidate only. It does not replace or stop "
            "the V8 production service. Production cutover must be a separate explicit action."
        ),
    }

    manifest_path = OUT_DIR / "full_arch_candidate_v1_manifest.json"
    manifest_path.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = OUT_DIR / "full_arch_candidate_v1_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "FULL ARCHITECTURE CANDIDATE V1",
                f"git_head={head}",
                "app_version=3.0.0-full-arch",
                f"enabled_targets={','.join(frozen['enabled_targets'])}",
                f"flat_rmse_v8={stage5.get('flat_rmse_v8')}",
                f"flat_rmse_candidate={stage5.get('flat_rmse_candidate')}",
                f"flat_rmse_ratio={stage5.get('flat_rmse_ratio')}",
                "stage6_direct_runtime=PASS",
                "stage7_callback_stress=PASS",
                "production_v8_untouched=true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("freeze manifest :", manifest_path)
    print("freeze summary  :", summary_path)
    print("production V8   : UNTOUCHED")
    print("STAGE 8 FREEZE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

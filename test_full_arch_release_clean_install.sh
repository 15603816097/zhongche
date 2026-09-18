#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="/root/full_arch_release/full_arch_v1_official_52.895_20260918.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"
TEST_ROOT="/root/full_arch_release_test"
NAME="full_arch_v1_official_52.895_20260918"
TEST_DIR="$TEST_ROOT/$NAME"
TEST_PORT=8890

echo "======================================================================"
echo "FULL ARCHITECTURE V1 - PORTABLE RELEASE CLEAN INSTALL TEST"
echo "======================================================================"
echo "archive   : $ARCHIVE"
echo "test dir  : $TEST_DIR"
echo "test port : $TEST_PORT"
echo

if [ ! -f "$ARCHIVE" ]; then
  echo "missing archive: $ARCHIVE" >&2
  exit 2
fi
if [ ! -f "$ARCHIVE_SHA" ]; then
  echo "missing archive checksum: $ARCHIVE_SHA" >&2
  exit 2
fi

echo "[1/6] Verify archive SHA256..."
(
  cd "$(dirname "$ARCHIVE")"
  sha256sum -c "$(basename "$ARCHIVE_SHA")"
)

echo "[2/6] Extract into a completely new directory..."
rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"
tar -xzf "$ARCHIVE" -C "$TEST_ROOT"

if [ ! -d "$TEST_DIR" ]; then
  echo "expected extracted directory missing: $TEST_DIR" >&2
  exit 2
fi

echo "[3/6] Run clean installer..."
cd "$TEST_DIR"
bash install.sh

echo "[4/6] Start isolated API on port $TEST_PORT..."
PORT="$TEST_PORT" bash start.sh

cleanup() {
  cd "$TEST_DIR" 2>/dev/null || true
  bash stop.sh >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[5/6] Verify health..."
PORT="$TEST_PORT" bash health.sh

echo "[6/6] Verify root and predict routes exist..."
"$TEST_DIR/.runtime/bin/python" - "$TEST_PORT" <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])

with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
    root = json.load(r)
print("ROOT:", json.dumps(root, ensure_ascii=False))

with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
    health = json.load(r)
print("HEALTH:", json.dumps(health, ensure_ascii=False))

assert root.get("status") == "ok"
assert root.get("predict_endpoint") == "/predict"
assert health.get("status") == "ok"
assert health.get("version") == "3.0.0-full-arch"
assert health.get("ensemble_version") == 8
assert health.get("model_loaded") is True

print("ROUTE CHECK: PASS")
PY

echo
echo "======================================================================"
echo "PORTABLE RELEASE CLEAN INSTALL TEST: PASS"
echo "======================================================================"
echo "Archive is ready to download and later redeploy on a fresh server."
echo "Test installation: $TEST_DIR"

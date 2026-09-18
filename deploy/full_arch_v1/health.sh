#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.runtime/bin/python"
PORT="${PORT:-8800}"

if [ ! -x "$PY" ]; then
  echo "runtime environment missing" >&2
  exit 2
fi

"$PY" - "$PORT" <<'PY'
import json, sys, urllib.request
port = int(sys.argv[1])
with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
    data = json.load(r)
print(json.dumps(data, ensure_ascii=False, indent=2))
PY

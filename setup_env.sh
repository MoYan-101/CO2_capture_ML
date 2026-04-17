#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt

cat <<'EOF'

Environment setup complete.

Activate it with:
  source .venv/bin/activate

Run a quick pipeline test with:
  bash run.sh test

EOF

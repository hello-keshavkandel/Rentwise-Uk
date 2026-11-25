#!/usr/bin/env bash
# RentWise UK — create a reproducible Python 3.11 environment in .venv
# (the live app, Docker and CI all use Python 3.11; some pinned packages do not support 3.13+).
set -euo pipefail
cd "$(dirname "$0")/.."
REQS=(-r requirements.txt -r requirements-api.txt -r requirements-dev.txt)
[ "${WITH_DATA:-0}" = "1" ] && REQS+=(-r requirements-data.txt)

if command -v uv >/dev/null 2>&1; then
  echo "==> Using uv (downloads Python 3.11 if needed)"
  uv venv .venv --python 3.11
  uv pip install --python .venv/bin/python "${REQS[@]}"
elif command -v python3.11 >/dev/null 2>&1; then
  echo "==> Using python3.11"
  python3.11 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install "${REQS[@]}"
else
  cat <<'MSG'
Python 3.11 was not found. Install ONE of these, then run `make setup` again:
  * uv (recommended, installs Python 3.11 for you):   brew install uv
        or: curl -LsSf https://astral.sh/uv/install.sh | sh
  * Python 3.11 itself:                               brew install python@3.11
MSG
  exit 1
fi

[ -f .env ] || cp .env.example .env
echo
echo "==> Done. Python: $(.venv/bin/python --version)"
echo "    Next:  make verify     (checks everything)"
echo "           make demo       (dashboard + API with a demo account)"

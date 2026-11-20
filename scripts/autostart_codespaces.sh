#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f catboost-v3-artifact.zip && ! -f artifacts/catboost-v3/model.cbm ]]; then
  echo "Antifraud: catboost-v3-artifact.zip is missing from GitHub; refusing synthetic demo."
  echo "Add the archive to the repository once; future Codespaces will launch automatically."
  exit 0
fi
if pgrep -f '[u]vicorn antifraud.app:app' >/dev/null; then
  echo "Antifraud API already running."
  exit 0
fi
nohup python scripts/start_codespaces.py > /tmp/antifraud-codespaces.log 2>&1 < /dev/null &
sleep 4
if ! kill -0 "$!" 2>/dev/null; then
  echo "Antifraud startup failed; see /tmp/antifraud-codespaces.log"
  tail -25 /tmp/antifraud-codespaces.log || true
else
  echo "Antifraud started; open forwarded port 8000 /docs"
fi

#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/hj153lee/PalmClaw"
readonly PYTHON_BIN="$REPO_ROOT/.conda/ubuntu-agent/bin/python"
readonly EVALUATION_SESSION="v2-hybrid-s21-s25-evaluation"
readonly EVALUATION_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid-state-evolution-s21-s25-evaluation"
readonly CONTROLLER="$REPO_ROOT/ubuntu/evaluation/experiment-scripts/run_vehiclemembench_v2_hybrid_s26_s50_controller.sh"

printf '[%s] Waiting for S21-S25 evaluation\n' "$(date -u +%FT%TZ)"
while tmux has-session -t "$EVALUATION_SESSION" 2>/dev/null; do
  sleep 60
done

"$PYTHON_BIN" - "$EVALUATION_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = (
    root / "update-audit" / "dual-judge-summary.json",
    root / "answerability" / "answerability-summary.json",
    root / "agent" / "agent-summary.json",
)
for path in paths:
    if not path.is_file():
        raise SystemExit(f"S21-S25 evaluation artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETED":
        raise SystemExit(f"S21-S25 evaluation is incomplete: {path}")
    if payload.get("failure_count", 0) != 0 or payload.get("failures"):
        raise SystemExit(f"S21-S25 evaluation has runtime failures: {path}")
PY

printf '[%s] S21-S25 evaluation gate passed; starting S26-S50\n' \
  "$(date -u +%FT%TZ)"
exec "$CONTROLLER"

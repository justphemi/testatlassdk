#!/usr/bin/env bash
# Replace the YOUR-ORG/YOUR-REPO placeholder everywhere. Usage: scripts/set_repo.sh my-org/my-repo
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 ORG/REPO"; exit 1; }
cd "$(dirname "$0")/.."
grep -rl "YOUR-ORG/YOUR-REPO" . --exclude-dir=.git --exclude-dir=.venv --exclude=set_repo.sh | while read -r f; do
  python3 - "$f" "$1" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); p.write_text(p.read_text().replace("YOUR-ORG/YOUR-REPO", sys.argv[2]))
PY
  echo "updated $f"
done

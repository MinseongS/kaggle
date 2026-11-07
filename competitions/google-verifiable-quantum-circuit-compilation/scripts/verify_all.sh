#!/usr/bin/env bash
# Usage: scripts/verify_all.sh <submission_dir> <out_dir> [tasks...]
# submission_dir 에 invsqrt.kmb, dqi.kmb, dfthcsos/ 가 있어야 함. 각 verifier 의 JSON 과 소요 시간을 out_dir 에 저장.
set -u
C="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$(cd "$1" && pwd)"; OUT="$2"; shift 2
TASKS="${*:-dfthcsos invsqrt dqi}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
cd "$C/data/verifiers"
for t in $TASKS; do
  case $t in
    dfthcsos) args=(dfthcsos/verify.py "$SUB/dfthcsos");;
    invsqrt)  args=(invsqrt/verify_submission_invsqrt.py "$SUB/invsqrt.kmb");;
    dqi)      args=(dqi/verify_submission_dqi.py "$SUB/dqi.kmb");;
  esac
  s=$(date +%s)
  uv run --project "$C" python "${args[@]}" > "$OUT/$t.json" 2> "$OUT/$t.err"
  echo "$t exit=$? sec=$(( $(date +%s) - s ))" | tee -a "$OUT/timing.txt"
done

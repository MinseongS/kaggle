#!/usr/bin/env bash
# launchd 가 매일 밤 실행: Claude 헤드리스 연구 세션 (시간·예산 제한).
set -u
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/opt/homebrew/Caskroom/miniconda/base/bin:/usr/bin:/bin:/usr/sbin:/sbin"
C="$(cd "$(dirname "$0")/.." && pwd)"
cd "$C"
TOKF="$HOME/.config/vqcc/claude_oauth_token"
if [ -r "$TOKF" ]; then export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$TOKF")"; fi
mkdir -p outputs/research_logs
LOGF="outputs/research_logs/$(date +%Y%m%d-%H%M).log"
HOURS="${RESEARCH_HOURS:-5}"
BUDGET="${RESEARCH_BUDGET_USD:-15}"
/usr/bin/caffeinate -i timeout "${HOURS}h" claude -p "$(cat scripts/research_prompt.md)" \
  --dangerously-skip-permissions --max-budget-usd "$BUDGET" --output-format text > "$LOGF" 2>&1
echo "exit=$? $(date)" >> "$LOGF"

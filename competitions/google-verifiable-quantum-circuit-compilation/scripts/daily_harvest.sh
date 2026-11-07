#!/usr/bin/env bash
# Orca 자동화가 매일 실행: 공개 노트북 수집·검증만 (기준점 확인용). 2026-11-07 사용자 지시로 자동 제출 중단 (--no-submit).
set -u
export PATH="/Applications/Orca.app/Contents/Resources/bin:$HOME/.local/bin:/opt/homebrew/bin:/opt/homebrew/Caskroom/miniconda/base/bin:/usr/bin:/bin:/usr/sbin:/sbin"
C="$(cd "$(dirname "$0")/.." && pwd)"
cd "$C"
mkdir -p outputs/harvest
/usr/bin/caffeinate -i timeout 3h uv run python src/harvest.py --top 15 --no-submit >> outputs/harvest/launchd.log 2>&1
uv run python scripts/healthcheck.py > /dev/null 2>> outputs/harvest/launchd.log

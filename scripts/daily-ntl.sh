#!/bin/zsh
# Daily autonomous run for name-that-llm. Invoked by launchd (~/Library/LaunchAgents/com.minseong.ntl-daily.plist).
export PATH="/Users/minseong/.local/bin:/opt/homebrew/Caskroom/miniconda/base/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
COMP=/Users/minseong/project/kaggle/competitions/name-that-llm
LOGDIR=$COMP/outputs/daily; mkdir -p $LOGDIR
TS=$(date +%Y%m%d-%H%M)
LOCK=/tmp/ntl-daily.lock
if [[ -e $LOCK ]] && kill -0 $(cat $LOCK) 2>/dev/null; then echo "$TS previous run still active" >> $LOGDIR/skipped.log; exit 0; fi
echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
cd /Users/minseong/project/kaggle
# long-lived token from `claude setup-token`, stored outside git
[[ -f ~/.claude/ntl-oauth-token ]] && export CLAUDE_CODE_OAUTH_TOKEN=$(cat ~/.claude/ntl-oauth-token)
caffeinate -i claude -p --dangerously-skip-permissions \
  "name-that-llm 대회 일일 자동 실행이다 (시각 $TS KST). competitions/name-that-llm/DAILY.md 의 절대 규칙과 절차를 그대로 따라라. 리포트는 competitions/name-that-llm/outputs/daily/report-$TS.md 에 써라." \
  > $LOGDIR/run-$TS.log 2>&1
RC=$?
# alert the user when a run fails (auth expiry, crash) — otherwise silent failures go unnoticed
LOG=$LOGDIR/run-$TS.log
if [[ $RC -ne 0 ]] || grep -qiE "invalid api key|not logged in|please run /login|authentication_error" $LOG || [[ ! -f $LOGDIR/report-$TS.md ]]; then
  echo "$TS FAILED rc=$RC" >> $LOGDIR/failures.log
  osascript -e "display notification \"자동 실행 실패 ($TS). outputs/daily/run-$TS.log 확인\" with title \"name-that-llm\" sound name \"Basso\""
fi

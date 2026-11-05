#!/bin/zsh
# Daily autonomous run for enveda-CASMI26. Invoked by launchd (~/Library/LaunchAgents/com.minseong.casmi-daily.plist).
export PATH="/Users/minseong/.local/bin:/opt/homebrew/Caskroom/miniconda/base/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
COMP=/Users/minseong/project/kaggle/competitions/enveda-CASMI26-molecule-id-mass-spectra
LOGDIR=$COMP/outputs/daily; mkdir -p $LOGDIR
TS=$(date +%Y%m%d-%H%M)
LOCK=/tmp/casmi-daily.lock
if [[ -e $LOCK ]] && kill -0 $(cat $LOCK) 2>/dev/null; then echo "$TS previous run still active" >> $LOGDIR/skipped.log; exit 0; fi
echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
cd /Users/minseong/project/kaggle
# long-lived token from `claude setup-token`, stored outside git (shared with the ntl automation)
[[ -f ~/.claude/ntl-oauth-token ]] && export CLAUDE_CODE_OAUTH_TOKEN=$(cat ~/.claude/ntl-oauth-token)
caffeinate -i claude -p --dangerously-skip-permissions \
  "enveda-CASMI26 대회 일일 자동 실행이다 (시각 $TS KST). competitions/enveda-CASMI26-molecule-id-mass-spectra/DAILY.md 의 절대 규칙과 절차를 그대로 따라라. 작업 목록은 같은 폴더의 QUEUE.md. 리포트는 competitions/enveda-CASMI26-molecule-id-mass-spectra/outputs/daily/report-$TS.md 에 써라." \
  > $LOGDIR/run-$TS.log 2>&1

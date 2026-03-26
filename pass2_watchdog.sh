#!/bin/zsh

set -u

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_CODEX_THREAD_ID="019d2a03-548c-76a2-8ae3-806ce486e587"
CODEX_THREAD_ID="${1:-$DEFAULT_CODEX_THREAD_ID}"
MAIN_LOG="$REPO_DIR/run/pass2-main.log"
WATCHDOG_LOG="$REPO_DIR/run/pass2-watchdog.log"
CODEX_LOG="$REPO_DIR/run/pass2-codex.log"
LAST_MESSAGE_LOG="$REPO_DIR/run/pass2-codex-last.txt"

cd "$REPO_DIR" || exit 1
mkdir -p run
[[ -n "$CODEX_THREAD_ID" ]] || {
  print -u2 -- "CODEX_THREAD_ID must be non-empty."
  exit 1
}

while true; do
  checkpoint="$(wc -l < all_consumer_health_questions.pass2.jsonl 2>/dev/null || print 0)"
  printf '[%s] starting main | checkpoint=%s\n' "$(date -Iseconds)" "$checkpoint" >> "$WATCHDOG_LOG"
  uv run --env-file .env -m main >> "$MAIN_LOG" 2>&1
  exit_code="$?"
  checkpoint="$(wc -l < all_consumer_health_questions.pass2.jsonl 2>/dev/null || print 0)"
  printf '[%s] main exited | code=%s | checkpoint=%s\n' "$(date -Iseconds)" "$exit_code" "$checkpoint" >> "$WATCHDOG_LOG"
  if [[ "$exit_code" -eq 0 ]]; then
    exit 0
  fi

  while true; do
    printf '[%s] invoking codex resume | thread_id=%s | checkpoint=%s\n' "$(date -Iseconds)" "$CODEX_THREAD_ID" "$checkpoint" >> "$WATCHDOG_LOG"
    # `-C/--cd` is an option on `codex exec`, not on the `resume` subcommand.
    # Keep it before `resume` or the CLI exits with usage error code 2.
    codex exec \
      --dangerously-bypass-approvals-and-sandbox \
      -C "$REPO_DIR" \
      resume \
      -o "$LAST_MESSAGE_LOG" \
      "$CODEX_THREAD_ID" \
      - >> "$CODEX_LOG" 2>&1 <<EOF
The pass-2 batch crashed in $REPO_DIR.

Read the latest traceback from run/pass2-main.log, fix the root cause with a minimal patch, run \`uv run --env-file .env -m ruff check\`, commit the fix with a focused commit message, and then exit.

Do not launch the long batch yourself; this watchdog will relaunch \`uv run --env-file .env -m main\` automatically after you finish.

Current checkpoint: $checkpoint lines in all_consumer_health_questions.pass2.jsonl.
EOF
    codex_exit_code="$?"
    printf '[%s] codex exited | code=%s\n' "$(date -Iseconds)" "$codex_exit_code" >> "$WATCHDOG_LOG"
    if [[ "$codex_exit_code" -eq 0 ]]; then
      break
    fi
    sleep 10
  done
done

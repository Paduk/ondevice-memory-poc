#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MARKER_FILE TMUX_SESSION" >&2
  exit 2
fi

marker=$1
session=$2
while [[ ! -s "$marker" ]]; do
  sleep 10
done

command=$(tmux list-panes -t "$session":0.0 -F '#{pane_start_command}')
command=${command#\"}
command=${command%\"}
tmux send-keys -t "$session":0.0 C-c
sleep 5
if tmux has-session -t "$session" 2>/dev/null; then
  tmux respawn-pane -k -t "$session":0.0 "$command"
else
  tmux new-session -d -s "$session" "$command"
fi

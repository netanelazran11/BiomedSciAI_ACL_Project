#!/usr/bin/env bash
# Compare methyl/paper/ (Claude, git-tracked) against the Overleaf copy
# (ChatGPT/Codex, not tracked). Reports word counts and diffs per section.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OTHER="$HERE/../outputs/overleaf/MethylLlama_Publication"
[ -d "$OTHER" ] || { echo "Overleaf copy not found at $OTHER"; exit 1; }

wc_() { sed 's/%.*//' "$1" 2>/dev/null | sed 's/\\[a-zA-Z]*//g' | wc -w | tr -d ' '; }

printf "%-34s %10s %10s   %s\n" "section" "paper/" "overleaf/" "status"
printf '%.0s-' {1..76}; echo
for f in "$HERE"/sections/*.tex; do
  b="$(basename "$f")"; o="$OTHER/sections/$b"
  if [ ! -f "$o" ]; then
    printf "%-34s %10s %10s   %s\n" "$b" "$(wc_ "$f")" "-" "only here"
  elif diff -q "$f" "$o" >/dev/null 2>&1; then
    printf "%-34s %10s %10s   %s\n" "$b" "$(wc_ "$f")" "$(wc_ "$o")" "identical"
  else
    printf "%-34s %10s %10s   %s\n" "$b" "$(wc_ "$f")" "$(wc_ "$o")" "DIFFERS"
  fi
done
for f in "$OTHER"/sections/*.tex; do
  b="$(basename "$f")"
  [ -f "$HERE/sections/$b" ] || printf "%-34s %10s %10s   %s\n" "$b" "-" "$(wc_ "$f")" "only in overleaf"
done

if [ "${1:-}" = "--full" ]; then
  echo; echo "===== unified diffs ====="
  for f in "$HERE"/sections/*.tex; do
    o="$OTHER/sections/$(basename "$f")"
    [ -f "$o" ] && ! diff -q "$f" "$o" >/dev/null 2>&1 && {
      echo; echo "--- $(basename "$f") ---"; diff -u "$o" "$f" | head -60; }
  done
fi

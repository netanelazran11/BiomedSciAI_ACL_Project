#!/usr/bin/env bash
# Compare methyl/paper/ (Claude, git-tracked) against the Overleaf copy
# (ChatGPT/Codex, not tracked). Reports word counts and diffs per section.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OTHER="$HERE/../outputs/overleaf/MethylLlama_Publication"
[ -d "$OTHER" ] || { echo "Overleaf copy not found at $OTHER"; exit 1; }

wc_() { sed 's/%.*//' "$1" 2>/dev/null | sed 's/\\[a-zA-Z]*//g' | wc -w | tr -d ' '; }

# A figure's identity is what it draws plus its size. Plain md5 is useless here:
# matplotlib writes a /CreationDate into every PDF, so re-running a script with
# no code change produces a different md5 and every figure reads as STALE.
fig_id() { printf '%s %s' "$(pdftotext "$1" - 2>/dev/null | md5 -q)" "$(wc -c < "$1" | tr -d ' ')"; }

# Divergence is usually visible in the mtimes before it is visible in the text,
# so print them: an overleaf/ section newer than its paper/ twin means the other
# copy moved after this one was last synced.
mt_() { date -r "$1" "+%m-%d %H:%M" 2>/dev/null || echo "-"; }

printf "%-34s %8s %8s  %-11s %-11s  %s\n" \
  "section" "paper/" "overleaf/" "paper mtime" "ovl mtime" "status"
printf '%.0s-' {1..92}; echo
for f in "$HERE"/sections/*.tex; do
  b="$(basename "$f")"; o="$OTHER/sections/$b"
  if [ ! -f "$o" ]; then
    printf "%-34s %8s %8s  %-11s %-11s  %s\n" "$b" "$(wc_ "$f")" "-" "$(mt_ "$f")" "-" "only here"
  elif diff -q "$f" "$o" >/dev/null 2>&1; then
    printf "%-34s %8s %8s  %-11s %-11s  %s\n" "$b" "$(wc_ "$f")" "$(wc_ "$o")" "$(mt_ "$f")" "$(mt_ "$o")" "identical"
  else
    printf "%-34s %8s %8s  %-11s %-11s  %s\n" "$b" "$(wc_ "$f")" "$(wc_ "$o")" "$(mt_ "$f")" "$(mt_ "$o")" "DIFFERS"
  fi
done
for f in "$OTHER"/sections/*.tex; do
  b="$(basename "$f")"
  [ -f "$HERE/sections/$b" ] || printf "%-34s %8s %8s  %-11s %-11s  %s\n" "$b" "-" "$(wc_ "$f")" "-" "$(mt_ "$f")" "only in overleaf"
done


# --- figures: compare the rendered PDFs against the canonical build output ---
CANON="$HERE/../figures/paper"
if [ -d "$CANON" ]; then
  echo
  printf "%-44s %s\n" "figure" "vs canonical build in figures/paper/"
  printf '%.0s-' {1..76}; echo
  for c in "$CANON"/*.pdf; do
    b="$(basename "$c")"; ref="$(fig_id "$c")"; line=""
    for where in "$HERE/figures/$b" "$OTHER/figures/$b"; do
      if [ ! -f "$where" ]; then line="$line missing;"
      elif [ "$(fig_id "$where")" = "$ref" ]; then line="$line current;"
      else line="$line STALE;"; fi
    done
    printf "%-44s paper:%-9s overleaf:%s\n" "$b" \
      "$(echo "$line" | cut -d';' -f1 | tr -d ' ')" \
      "$(echo "$line" | cut -d';' -f2 | tr -d ' ')"
  done
fi

if [ "${1:-}" = "--full" ]; then
  echo; echo "===== unified diffs ====="
  for f in "$HERE"/sections/*.tex; do
    o="$OTHER/sections/$(basename "$f")"
    [ -f "$o" ] && ! diff -q "$f" "$o" >/dev/null 2>&1 && {
      echo; echo "--- $(basename "$f") ---"; diff -u "$o" "$f" | head -60; }
  done
fi

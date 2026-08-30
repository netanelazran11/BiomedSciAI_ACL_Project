# Two manuscript sources — how to keep them in sync

There are two working copies of this manuscript, on purpose:

| Copy | Path | Edited by | Git history |
|---|---|---|---|
| **This one** | `methyl/paper/` | Claude | yes — every change is a commit |
| Overleaf copy | `methyl/outputs/overleaf/MethylLlama_Publication/` | ChatGPT / Codex | **no** (`outputs/` is gitignored) |

`methyl/paper/` was seeded from the Overleaf copy on 2026-08-30, then the
Methods, Discussion, supplementary figures, and the scConcept framing in the
Introduction were ported in from the chapter-review draft, which the Overleaf
copy was missing.

## See what has diverged

    bash methyl/paper/diff_sources.sh

Section filenames differ between the two copies only in that this one is the
Overleaf layout; the script maps them and reports per-section word counts and
a unified diff for any section that differs.

## Rule that makes this work

Commit after editing here. `git log -p paper/sections/<file>` then answers
"who changed what, and when" without reconstructing it from timestamps.

# Source from methyl/ after the module loads:   source ablation_infonce/env.sh
# Sets PY to a Python 3.11 that sees the bmfm_methyl_env packages, or exits.
#
# Why: bmfm_methyl_env was built on /usr/bin/python3.11 (3.11.2). Since Oct 2026 the cluster OS
# ships only /usr/bin/python3.13 (login node, salmon-07, catfish-04, ...), so the venv's python
# resolves to 3.13 and cannot see its 3.11 site-packages. Any CPython 3.11.x can load those
# packages (same ABI), so we run a standalone 3.11 with the venv site-packages on PYTHONPATH.
#
# Order: (1) venv python if it is 3.11; (2) /usr/bin/python3.11; (3) $PY311 if set;
#        (4) a uv-managed CPython 3.11 under $TOOLS/python (see README in the commit message).

TOOLS="${TOOLS:-/sci/labs/benjamin.yakir/netanel.azran/tools}"
_VENV_SITE="$(pwd)/bmfm_methyl_env/lib/python3.11/site-packages"
source bmfm_methyl_env/bin/activate

_is311() { "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; }

PY=""
if _is311 python; then
    PY="$(command -v python)"
else
    for cand in /usr/bin/python3.11 "${PY311:-}" $(ls -d "${TOOLS}"/python/cpython-3.11*/bin/python3.11 2>/dev/null | sort -V | tail -1); do
        if [ -n "${cand}" ] && [ -x "${cand}" ] && _is311 "${cand}"; then PY="${cand}"; break; fi
    done
    if [ -z "${PY}" ]; then
        echo "ERROR: no Python 3.11 on $(hostname) (venv python: $(python --version 2>&1);" \
             "/usr/bin/python3 -> $(readlink -f /usr/bin/python3)). Install one with uv into ${TOOLS}/python" \
             "(see ablation_infonce/env.sh) or set PY311=/path/to/python3.11."
        exit 1
    fi
    echo "WARN: venv python on $(hostname) is $(python --version 2>&1); using ${PY} + venv site-packages"
    export PYTHONPATH="${_VENV_SITE}:${PYTHONPATH:-}"
    export PYTHONNOUSERSITE=1
fi

if ! "${PY}" -c 'import numpy, pandas, anndata, torch' 2>/dev/null; then
    echo "ERROR: ${PY} on $(hostname) cannot import numpy/pandas/anndata/torch:"
    "${PY}" -c 'import numpy, pandas, anndata, torch' 2>&1 | tail -5
    exit 1
fi
echo "Python: $("${PY}" --version 2>&1) (${PY}) on $(hostname); torch $("${PY}" -c 'import torch; print(torch.__version__)')"
export PY

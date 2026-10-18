# Source from methyl/ after the module loads:   source ablation_infonce/env.sh
# Sets PY to a Python 3.11 that sees the bmfm_methyl_env packages, or exits.
#
# Why: bmfm_methyl_env was built from /usr/bin/python3 (3.11.2). On nodes where /usr/bin/python3
# is a different version (seen: 3.13 on moriah-gw-02; job 46408952 on salmon-07 could not import
# numpy), the venv's python runs without the venv's 3.11 site-packages. Fallback: /usr/bin/python3.11
# with the venv site-packages on PYTHONPATH (same interpreter version the packages were built for).

source bmfm_methyl_env/bin/activate
_VENV_SITE="$(pwd)/bmfm_methyl_env/lib/python3.11/site-packages"

if python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; then
    PY="$(command -v python)"
elif [ -x /usr/bin/python3.11 ]; then
    echo "WARN: venv python on $(hostname) is $(python --version 2>&1); using /usr/bin/python3.11 + venv site-packages"
    PY=/usr/bin/python3.11
    export PYTHONPATH="${_VENV_SITE}:${PYTHONPATH:-}"
else
    echo "ERROR: no Python 3.11 on $(hostname) (venv python: $(python --version 2>&1);" \
         "/usr/bin/python3 -> $(readlink -f /usr/bin/python3)). Resubmit with --exclude=$(hostname)."
    exit 1
fi

if ! "${PY}" -c 'import numpy, pandas, anndata, torch' 2>/dev/null; then
    echo "ERROR: ${PY} on $(hostname) cannot import numpy/pandas/anndata/torch. Resubmit with --exclude=$(hostname)."
    exit 1
fi
echo "Python: $("${PY}" --version 2>&1) (${PY}) on $(hostname)"
export PY

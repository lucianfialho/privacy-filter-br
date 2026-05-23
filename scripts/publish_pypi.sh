#!/bin/bash
# Publish br-pii-guardrail to PyPI.
#
# Prereqs:
#   1. PyPI account: https://pypi.org/account/register/
#   2. Create API token (project-scoped or global): https://pypi.org/manage/account/token/
#   3. Install: pip install --upgrade build twine
#
# Token can be passed via:
#   - env var: TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-xxx bash scripts/publish_pypi.sh
#   - or ~/.pypirc with [pypi] entry
#
# Usage:
#   bash scripts/publish_pypi.sh           # publish to PyPI
#   PYPI_TEST=1 bash scripts/publish_pypi.sh   # publish to TestPyPI first (recommended)

set -euo pipefail

LIB_DIR="${LIB_DIR:-br-pii-guardrail}"

cd "$LIB_DIR"

echo "=== Cleaning old builds ==="
rm -rf dist build src/*.egg-info

echo ""
echo "=== Building distribution ==="
python3 -m build --sdist --wheel

echo ""
echo "=== Built artifacts ==="
ls -lh dist/

echo ""
if [ "${PYPI_TEST:-}" = "1" ]; then
    echo "=== Uploading to TestPyPI ==="
    python3 -m twine upload --repository testpypi dist/*
    echo ""
    echo "Install test:"
    echo "  pip install --index-url https://test.pypi.org/simple/ br-pii-guardrail"
else
    echo "=== Uploading to PyPI (production) ==="
    read -p "Confirm upload to production PyPI? [y/N] " confirm
    if [ "$confirm" != "y" ]; then
        echo "Aborted. To test first: PYPI_TEST=1 bash scripts/publish_pypi.sh"
        exit 0
    fi
    python3 -m twine upload dist/*
    echo ""
    echo "Done. Install with:"
    echo "  pip install br-pii-guardrail"
fi

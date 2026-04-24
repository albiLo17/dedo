#!/usr/bin/env bash
# =============================================================================
# Generate RGB screenshots for all observability conditions.
# Output: experiments/hang_obs_exp/viz_output/
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
SCRIPT="$(dirname "$0")/gen_screenshots.py"

echo "=== Generating Observability Visualizations ==="
cd "${REPO_ROOT}"
${PYTHON} "${SCRIPT}"
echo "=== Done. Output at experiments/hang_obs_exp/viz_output/ ==="

#!/usr/bin/env bash
# Runs the full paper benchmark on a Google Colab GPU runtime and zips the
# results for download. From a Colab cell, with the runtime set to a GPU:
#
#   !git clone -q --branch v0.2.1 https://github.com/udaykishore-resu/gpu-boundary /content/gpu-boundary && bash /content/gpu-boundary/bench/colab/run.sh
#
# It installs the latest Go if missing, runs `make bench-paper` with a label
# built from the GPU name and the checked-out tag, and writes
# /content/results-<label>.zip. Colab deletes /content when the runtime
# resets, so download the zip in the same cell.
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! nvidia-smi >/dev/null 2>&1; then
  echo "No GPU visible. In Colab: Runtime -> Change runtime type -> T4 GPU -> Save, then run again." >&2
  exit 1
fi

if [ ! -x /usr/local/go/bin/go ]; then
  V=$(curl -fsSL 'https://go.dev/VERSION?m=text' | head -1)
  curl -fsSL -o "/tmp/$V.linux-amd64.tar.gz" "https://go.dev/dl/$V.linux-amd64.tar.gz"
  tar -C /usr/local -xzf "/tmp/$V.linux-amd64.tar.gz"
  rm -f "/tmp/$V.linux-amd64.tar.gz"
fi
export PATH=$PATH:/usr/local/go/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

TAG=$(git describe --tags --always)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
LABEL="colab-${GPU}-${TAG}"
echo "label: $LABEL"

# Keep going on failure so whatever finished still gets zipped; the exit
# status still reports the failure.
status=0
make bench-paper LABEL="$LABEL" || status=$?

ZIP="/content/results-${LABEL}.zip"
rm -f "$ZIP"
if ls results/"${LABEL}"-*.json >/dev/null 2>&1; then
  (cd results && zip -q -j "$ZIP" "${LABEL}"-*.json)
  ls -l "$ZIP"
fi
if [ "$status" -ne 0 ]; then
  echo "bench-paper FAILED (exit $status); partial results, if any, are in $ZIP" >&2
fi
exit "$status"

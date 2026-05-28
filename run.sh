#!/bin/bash
# HTCondor job wrapper — uses the micromamba env Python directly.
# No shell hook needed; the full path to the env's python3 is unambiguous.

set -e

PYTHON=/eos/user/m/mmarriot/micromamba/envs/cluestering/bin/python3
WORKDIR=/afs/cern.ch/user/m/mmarriot/private/cluestering_patatune

echo "=== Job started at $(date) ==="
echo "Running on host: $(hostname)"
echo "Python: $PYTHON"
echo "Working dir: $WORKDIR"
echo ""

cd $WORKDIR

# PYTHONUNBUFFERED=1 forces every print() to flush immediately.
# Direct redirect (no pipe) so there is no pipe buffer to fill before writing.
export PYTHONUNBUFFERED=1
$PYTHON optimize.py > logs/live.out 2>&1

echo ""
echo "=== Job finished at $(date) ==="

#!/bin/bash
# HTCondor job wrapper for physics_plots.py
# Arguments are forwarded directly from condor_physics.sub

set -e

PYTHON=/eos/user/m/mmarriot/micromamba/envs/cluestering/bin/python3
WORKDIR=/afs/cern.ch/user/m/mmarriot/private/cluestering_patatune

echo "=== Job started at $(date) ==="
echo "Running on host: $(hostname)"
echo "Python: $PYTHON"
echo "Working dir: $WORKDIR"
echo "Args: $@"
echo ""

cd $WORKDIR

export PYTHONUNBUFFERED=1
$PYTHON physics_plots.py "$@"

echo ""
echo "=== Job finished at $(date) ==="

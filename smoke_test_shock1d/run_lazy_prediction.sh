#!/bin/bash

# MIRGE env vars used to setup cache locations
MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache/"}
export XDG_CACHE_ROOT=${XDG_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/xdg-cache"}

echo "$MIRGE_CACHE_ROOT"
echo "$XDG_CACHE_ROOT"

nproc=1
run_cmd="mpiexec -n $nproc"

# turn off optimization for CI (enabled asserts)
$run_cmd bash -c 'XDG_CACHE_HOME=$XDG_CACHE_ROOT/$OMPI_COMM_WORLD_RANK python -u -m mpi4py ./driver.py -i run_params_quads_3d.yaml --lazy --overintegration > mirge-0.out'

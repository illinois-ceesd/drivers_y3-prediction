#!/bin/bash

#flux: --nodes=1024
#flux: --time=240
#flux: --bank=uiuc
#flux: --output=scal4096.txt

export MIRGE_CACHE_ROOT="./mirge-cache_1024node"
export MIRGE_CACHE_DISABLE=1

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 1024 -n 4096 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak_np4096.yaml --lazy --overintegration -c shock1d_weak_np4096

#!/bin/bash

#flux: --nodes=1024
#flux: --time=240
#flux: --bank=uiuc
#flux: --output=strong4096_1d.txt

export MIRGE_CACHE_ROOT="./mirge-cache_1024_1d"
export MIRGE_CACHE_DISABLE=1
source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 1024 -n 4096 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_strong.yaml --lazy --overintegration -c shock1d_strong_np4096_1d

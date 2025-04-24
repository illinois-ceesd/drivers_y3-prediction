#!/bin/bash

#flux: --nodes=512
#flux: --time=240
#flux: --bank=uiuc
#flux: --output=scal2048.txt

export MIRGE_CACHE_ROOT="./mirge-cache_512node"
export MIRGE_CACHE_DISABLE=1

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 512 -n 2048 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak_np2048.yaml --lazy --overintegration -c shock1d_weak_np2048

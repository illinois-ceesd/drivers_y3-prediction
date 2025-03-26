#!/bin/bash

#flux: --nodes=8
#flux: --time=60
#flux: --bank=uiuc
#flux: --output=scal8.txt

export MIRGE_CACHE_ROOT="./mirge-cache_8node"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 8 -n 32 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak.yaml -s 32 --lazy --overintegration -c shock1d_weak_np32

#!/bin/bash

#flux: --nodes=256
#flux: --time=240
#flux: --bank=uiuc
#flux: --output=scal256.txt

export MIRGE_CACHE_ROOT="./mirge-cache_256node"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 256 -n 1024 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak.yaml -s 1024 --lazy --overintegration -c shock1d_weak_np1024

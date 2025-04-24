#!/bin/bash

#flux: --nodes=16
#flux: --time=60
#flux: --bank=uiuc
#flux: --output=comcat_1dpart_64.txt

export MIRGE_CACHE_ROOT="./mirge-cache_16node"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 16 -n 64 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak_1dpart.yaml -s 64 --lazy --overintegration -c shock1d_weak_1dpart_np64

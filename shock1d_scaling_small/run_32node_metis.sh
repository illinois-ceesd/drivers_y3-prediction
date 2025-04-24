#!/bin/bash

#flux: --nodes=32
#flux: --time=120
#flux: --bank=uiuc
#flux: --output=comcat_metis_128.txt

export MIRGE_CACHE_ROOT="./mirge-cache_32node"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 32 -n 128 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak.yaml -s 128 --lazy --overintegration -c shock1d_weak_np128

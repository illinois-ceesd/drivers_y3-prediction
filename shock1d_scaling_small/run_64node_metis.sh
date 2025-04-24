#!/bin/bash

#flux: --nodes=64
#flux: --time=120
#flux: --bank=uiuc
#flux: --output=comcat_metis_256.txt

export MIRGE_CACHE_ROOT="./mirge-cache_64node"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 64 -n 256 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak.yaml -s 256 --lazy --overintegration -c shock1d_weak_metis_np256

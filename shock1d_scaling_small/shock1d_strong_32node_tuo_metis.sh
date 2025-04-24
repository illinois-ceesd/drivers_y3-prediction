#!/bin/bash

#flux: --nodes=32
#flux: --time=60
#flux: --bank=uiuc
#flux: --output=strong128_metis.txt

export MIRGE_CACHE_ROOT="./mirge-cache_128_metis"
export MIRGE_CACHE_DISABLE=1
source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 32 -n 128 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_strong_metis.yaml --lazy --overintegration -c shock1d_strong_np128_metis

#!/bin/bash

#flux: --nodes=64
#flux: --time=240
#flux: --bank=uiuc
#flux: --output=strong256_metis.txt

export MIRGE_CACHE_ROOT="./mirge-cache_512_metis"
export MIRGE_CACHE_DISABLE=1
source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 64 -n 256 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_strong_metis.yaml --lazy --overintegration -c shock1d_strong_np256_metis

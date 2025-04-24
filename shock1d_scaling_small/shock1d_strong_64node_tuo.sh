#!/bin/bash

#flux: --nodes=64
#flux: --time=180
#flux: --bank=uiuc
#flux: --output=strong256_1d.txt

export MIRGE_CACHE_ROOT="./mirge-cache_64node"
export MIRGE_CACHE_DISABLE=1

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 64 -n 256 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_strong.yaml --lazy --overintegration -c shock1d_strong_np256_1d

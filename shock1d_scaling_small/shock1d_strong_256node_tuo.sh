#!/bin/bash

#flux: --nodes=256
#flux: --time=120
#flux: --bank=uiuc
#flux: --output=strong1024_1d.txt

export MIRGE_CACHE_ROOT="./mirge-cache_256_1d"
export MIRGE_CACHE_DISABLE=1
source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 256 -n 1024 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_strong.yaml --lazy --overintegration -c shock1d_strong_np1024_1d

#!/bin/bash

#flux: --nodes=128
#flux: --time=180
#flux: --bank=uiuc
#flux: --output=scal128.txt

export MIRGE_CACHE_ROOT="./mirge-cache_128node"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 128 -n 512 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak.yaml -s 512 --lazy --overintegration -c shock1d_weak_np512

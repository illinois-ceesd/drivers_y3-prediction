#!/bin/bash

#flux: --nodes=64
#flux: --time=80
#flux: --bank=uiuc
#flux: --output=scal64.txt

export MIRGE_CACHE_ROOT="./mirge-cache_64node"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 64 -n 256 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i scalability_3d.yaml -s 256 --lazy --overintegration -c shock1d_weak_np256

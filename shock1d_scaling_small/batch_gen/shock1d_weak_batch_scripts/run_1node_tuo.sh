#!/bin/bash

#flux: --nodes=1
#flux: --time=60
#flux: --bank=uiuc
#flux: --output=scal1.txt

export MIRGE_CACHE_ROOT="./mirge-cache_1node"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 1 -n 1 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak.yaml -s 1 --lazy --overintegration -c shock1d_weak_np1
$MIRGE_MPI_EXEC -N 1 -n 2 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak.yaml -s 2 --lazy --overintegration -c shock1d_weak_np2
$MIRGE_MPI_EXEC -N 1 -n 4 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i shock1d_weak.yaml -s 4 --lazy --overintegration -c shock1d_weak_np4

#!/bin/bash

#flux: --nodes=1
#flux: --time=60
#flux: --bank=uiuc
#flux: --output=scal1_4.txt

export MIRGE_CACHE_ROOT="./mirge-cache_1node"

source /p/vast1/mtcampbe/CEESD/Experimental/mirge-03.18/config/activate_env.sh
source /p/vast1/mtcampbe/CEESD/Experimental/mirge-03.18/mirgecom/scripts/mirge-testing-env.sh

#$MIRGE_MPI_EXEC -N 1 -n 1 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i scalability_3d.yaml -s 1 --lazy --overintegration -c shock1d_weak_np1
#$MIRGE_MPI_EXEC -N 1 -n 2 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i scalability_3d.yaml -s 2 --lazy --overintegration -c shock1d_weak_np2
$MIRGE_MPI_EXEC -N 1 -n 4 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i scalability_3d.yaml -s 4 --lazy --overintegration -c shock1d_weak_np4

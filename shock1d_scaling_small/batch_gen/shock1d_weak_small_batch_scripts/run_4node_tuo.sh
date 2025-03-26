#!/bin/bash

#flux: --nodes=4
#flux: --time=60
#flux: --bank=uiuc
#flux: --output=scal4.txt

export MIRGE_CACHE_ROOT="./mirge-cache_4node"

source /p/vast1/mtcampbe/CEESD/Experimental/mirge-03.18/config/activate_env.sh
source /p/vast1/mtcampbe/CEESD/Experimental/mirge-03.18/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 4 -n 16 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i scalability_3d.yaml -s 16 --lazy --overintegration -c shock1d_weak_np16

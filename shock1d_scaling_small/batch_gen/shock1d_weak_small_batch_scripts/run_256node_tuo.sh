#!/bin/bash

#flux: --nodes=256
#flux: --time=240
#flux: --bank=uiuc
#flux: --output=scal256.txt

export MIRGE_CACHE_ROOT="./mirge-cache_256node"

source /p/vast1/mtcampbe/CEESD/Experimental/mirge-03.18/config/activate_env.sh
source /p/vast1/mtcampbe/CEESD/Experimental/mirge-03.18/mirgecom/scripts/mirge-testing-env.sh

$MIRGE_MPI_EXEC -N 256 -n 1024 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py ./driver.py -i scalability_3d.yaml -s 1024 --lazy --overintegration -c shock1d_weak_np1024

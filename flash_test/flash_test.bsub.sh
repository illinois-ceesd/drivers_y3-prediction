#! /bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 360
#BSUB -J flash10
#BSUB -q pbatch
#BSUB -o flash10.out

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

set -x
$MIRGE_MPI_EXEC -n 1 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py driver.py -c ${casename} -i run_params.yaml -r restart_data/prediction-001020000 --log --lazy
set +x


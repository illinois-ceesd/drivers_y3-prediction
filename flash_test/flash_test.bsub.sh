#! /bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 360
#BSUB -J flash9
#BSUB -q pbatch
#BSUB -o flash9.out

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh

set -x
$MIRGE_MPI_EXEC -n 1 $MIRGE_PARALLEL_SPAWNER python -u -O -m mpi4py driver.py -c ${casename} -i run_params.yaml -r restart_data/prediction-000965000 --log --lazy
set +x

# flash5 000483000
# flash6 000604000
# flash7 000726000
# flash8 000847000
# flash9 000965000

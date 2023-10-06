#!/bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc 
#BSUB -J unstart_test
#BSUB -W 00:20
#BSUB -q pdebug
##BSUB -W 08:00
##BSUB -q pbatch
#BSUB -o output.txt

exec="driver.py"


last_rstfile=$(ls -ltr restart_data/* | tail -n 1 | awk '{print $NF}')

last_rstfile="${last_rstfile%-*}"
echo '!!!!!!!!!!!!!!!!!!!!!! PRINTING FILE NAME !!!!!!!!!!!!!!'
echo $last_rstfile

source /p/gpfs1/lauer5/drivers_y3-prediction/emirge/config/activate_env.sh
source /p/gpfs1/lauer5/drivers_y3-prediction/emirge/mirgecom/scripts/mirge-testing-env.sh
#$MIRGE_MPI_EXEC -n 1 $MIRGE_PARALLEL_SPAWNER python -m mpi4py $exec --lazy -r $rst_file
#$MIRGE_MPI_EXEC $MIRGE_PARALLEL_SPAWNER python -m mpi4py $exec # -r $last_rstfile
$MIRGE_MPI_EXEC -n 1 $MIRGE_PARALLEL_SPAWNER python -m mpi4py $exec -i run_params.yaml -r $last_rstfile

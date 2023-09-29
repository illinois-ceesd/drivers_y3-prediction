#!/bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc 
#BSUB -J unstart_test
#BSUB -W 00:20
#BSUB -q pdebug
##BSUB -W 720
##BSUB -q pbatch
#BSUB -o output.txt

exec="driver.py"

#rst_file="./restart_data/burner_mix-d2p2e21350n1-230000-0000.pkl"

source /p/gpfs1/lauer5/drivers_y3-prediction/emirge/config/activate_env.sh

#$MIRGE_MPI_EXEC -n 1 $MIRGE_PARALLEL_SPAWNER python -m mpi4py $exec --lazy -r $rst_file
#$MIRGE_MPI_EXEC $MIRGE_PARALLEL_SPAWNER python -m mpi4py $exec # -r $rst_file
mpirun -n 2 python -u -m mpi4py $exec -i run_params.yaml

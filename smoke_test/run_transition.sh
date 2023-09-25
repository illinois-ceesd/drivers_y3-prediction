#!/bin/bash
#mpirun -n 2 python -u -O -m mpi4py prediction_scalar_to_multispecies.py -i run_params_trans.yaml -r restart_data/prediction-000010 --lazy
#mpirun -n 2 python -u -O -m mpi4py driver.py -i transition_params.yaml -r restart_data/prediction-restart-000000020 --casename=prediction-transition
mpirun -n 2 python -u -m mpi4py driver.py -i transition_params.yaml -r restart_data/prediction-restart-000000020 --casename=prediction-transition

#!/bin/bash
#mpirun -n 2 python -u -m mpi4py driver.py -i transition_params.yaml -r restart_data/prediction-restart-000000020 --casename=prediction-transition
mpirun -n 2 python -u -m mpi4py driver.py -i transition_params.yaml -r restart_data/prediction-restart-000000020 --casename=prediction-transition

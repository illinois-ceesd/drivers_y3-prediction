#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py prediction_scalar_to_multispecies.py -i run_params_trans.yaml -r restart_data/prediction-000010 --lazy

#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml -r restart_data/prediction-000000010 -t restart_data/prediction-000000000 --log

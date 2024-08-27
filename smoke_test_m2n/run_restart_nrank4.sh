#!/bin/bash
mpirun -n 4 python -u -m mpi4py driver.py -i run_params.yaml -r restart_data_4p/prediction-nrank3-000000010 -t restart_data_4p/prediction-nrank3-000000000 --casename=prediction-restart-nrank4 --numpy

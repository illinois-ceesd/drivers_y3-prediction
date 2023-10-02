#!/bin/bash
#mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --casename=prediction-eager
mpirun -n 2 python -u -m mpi4py driver.py -i run_params.yaml --casename=prediction-eager

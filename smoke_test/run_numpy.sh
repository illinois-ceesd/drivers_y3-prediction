#!/bin/bash
#mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --numpy --casename=prediction-numpy
mpirun -n 2 python -u -m mpi4py driver.py -i run_params.yaml --numpy --casename=prediction-numpy

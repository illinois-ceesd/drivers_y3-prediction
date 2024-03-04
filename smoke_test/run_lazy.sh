#!/bin/bash
#mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy --casename=prediction-lazy
mpirun -n 8 python -u -m mpi4py driver.py -i run_params.yaml --lazy --casename=prediction-lazy 2>&1 | tee out.txt

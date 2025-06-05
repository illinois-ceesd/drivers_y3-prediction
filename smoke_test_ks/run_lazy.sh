#!/bin/bash
#mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy
# turn off optimizations for CI (enables asserts)
mpirun -n 2 python -u -m mpi4py driver.py -i run_params.yaml --log --lazy --overintegration

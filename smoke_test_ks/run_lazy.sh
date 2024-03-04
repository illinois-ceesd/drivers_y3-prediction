#!/bin/bash
#mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy
# turn off optimizations for CI (enables asserts)
mpirun -n 1 --oversubscribe python -u -m mpi4py driver.py -i run_params.yaml --log --lazy 2>&1 | tee out.txt

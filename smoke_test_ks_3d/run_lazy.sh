#!/bin/bash

# Create the mesh for a small, simple test (size==mesh spacing)
#mpiexec -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy
# turn off optimization for CI (enabled asserts)
mpiexec -n 2 python -u -m mpi4py driver.py -i run_params.yaml --lazy

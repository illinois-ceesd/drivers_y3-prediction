#!/bin/bash
mpirun -n 4 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy

#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py driver.py -c step3_init -i init_params.yaml --log --lazy

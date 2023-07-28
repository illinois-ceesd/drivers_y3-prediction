#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py driver.py --esdg -i run_params.yaml

#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py driver.py --esdg --numpy -i run_params.yaml

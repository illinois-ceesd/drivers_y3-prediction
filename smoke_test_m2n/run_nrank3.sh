#!/bin/bash
mpirun -n 3 --oversubscribe python -u -m mpi4py driver.py -i run_params.yaml --numpy --casename=prediction-nrank3

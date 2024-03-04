#!/bin/bash
#mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy --casename=prediction-lazy
mpirun -n 4 zsh run_lazy_rank.sh 2>&1 | tee out.txt

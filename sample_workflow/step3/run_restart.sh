#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml -r restart_data/step1_final -t ../target_states/single_species/restart_data/target --log --lazy

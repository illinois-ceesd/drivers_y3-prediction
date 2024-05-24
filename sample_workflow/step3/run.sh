#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy -t init_data/step3_init-000000000 -r init_data/step3_transfer_n0_to_n7-000000200
#mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy -r init_data/step3_transfer_n0_to_n7-000000200
#mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy -t init_data/step3_init-000000000
#mpirun -n 2 python -u -O -m mpi4py driver.py -i run_params.yaml --log --lazy

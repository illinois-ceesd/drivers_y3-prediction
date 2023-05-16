#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py prediction_scalar_to_multispecies.py -i init_params.yaml -r init_data/prediction-000000200 -c step3_transfer_n0_to_n7

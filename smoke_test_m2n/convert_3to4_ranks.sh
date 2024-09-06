#!/bin/bash
# first create 3 and 4 rank mesh distributions
mpirun -n 1 python -m mpi4py meshdist.py -w -1 -d 2 -n 3 -s data/actii_2d.msh -o actii_2d_3p
mpirun -n 1 python -m mpi4py meshdist.py -w -1 -d 2 -n 4 -s data/actii_2d.msh -o actii_2d_4p

# transfer date from the 3p run to 4p restart files
mpirun -n 1 python -m mpi4py redist.py -m 3 -n 4 -i restart_data/prediction-nrank3-000000010 -s actii_2d_3p/mirgecom_np3 -t actii_2d_4p/mirgecom_np4 -o restart_data_4p
mpirun -n 1 python -m mpi4py redist.py -m 3 -n 4 -i restart_data/prediction-nrank3-000000000 -s actii_2d_3p/mirgecom_np3 -t actii_2d_4p/mirgecom_np4 -o restart_data_4p

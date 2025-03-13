#!/bin/bash

#flux: --nodes=1
#flux: --time=360
#flux: --output=scal4.txt
##BSUB -J scale4

cd /p/lustre5/mtcampbe/CEESD/Experimental/main.2025.02.18/drivers_y3-prediction/scalability_test

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -n 4 -N 1




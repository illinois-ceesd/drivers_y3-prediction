#!/bin/bash
#BSUB -nnodes 4
#BSUB -G uiuc
#BSUB -W 120
#BSUB -J scale16
#BSUB -q pdebug
#BSUB -o scal16.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 16 -n 16

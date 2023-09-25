#!/bin/bash
#BSUB -nnodes 2
#BSUB -G uiuc
#BSUB -W 120
#BSUB -J scale8
#BSUB -q pdebug
#BSUB -o scal8.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 8 -n 8


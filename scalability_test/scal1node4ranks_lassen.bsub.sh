#!/bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 80
#BSUB -J scale4
#BSUB -q pdebug
#BSUB -o scal4.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 4 -n 4



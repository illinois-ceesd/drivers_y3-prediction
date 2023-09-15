#!/bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 20
#BSUB -J scale1
#BSUB -q pdebug
#BSUB -o scal1.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 1 -n 1



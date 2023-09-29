#!/bin/bash
#BSUB -nnodes 8
#BSUB -G uiuc
#BSUB -W 120
#BSUB -J scale32
#BSUB -q pbatch
#BSUB -o scal32.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 32 -n 32

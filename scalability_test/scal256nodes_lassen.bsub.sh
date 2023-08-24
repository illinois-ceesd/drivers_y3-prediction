#!/bin/bash
#BSUB -nnodes 256
#BSUB -G uiuc
#BSUB -W 300
#BSUB -J scale1024
#BSUB -q pbatch
#BSUB -o scal1024.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 1024 -n 1024

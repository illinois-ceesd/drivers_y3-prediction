#!/bin/bash
#BSUB -nnodes 32
#BSUB -G uiuc
#BSUB -W 180
#BSUB -J scale128
#BSUB -q pbatch
#BSUB -o scal128.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 128 -n 128

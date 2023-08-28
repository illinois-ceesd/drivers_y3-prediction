#!/bin/bash
#BSUB -nnodes 64
#BSUB -G uiuc
#BSUB -W 180
#BSUB -J scale256
#BSUB -q pbatch
#BSUB -o scal256.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 256 -n 256

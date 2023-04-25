#!/bin/bash
#BSUB -nnodes 4
#BSUB -G uiuc
#BSUB -W 180
#BSUB -J scale16
#BSUB -q pbatch
#BSUB -o scal16.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 1 -n 16

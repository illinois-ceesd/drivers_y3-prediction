#!/bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 30
#BSUB -J iotest4
#BSUB -q pbatch
#BSUB -o iotest4.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -n 4



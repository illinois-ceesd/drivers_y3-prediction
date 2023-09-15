#!/bin/bash
#BSUB -nnodes 2
#BSUB -G uiuc
#BSUB -W 30
#BSUB -J iotest8
#BSUB -q pbatch
#BSUB -o iotest8.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 8 -n 8



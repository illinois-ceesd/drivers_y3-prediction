#!/bin/bash
#BSUB -nnodes 4
#BSUB -G uiuc
#BSUB -W 30
#BSUB -J iotest16
#BSUB -q pbatch
#BSUB -o iotest16.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 16 -n 16



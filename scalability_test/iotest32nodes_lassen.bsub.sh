#!/bin/bash
#BSUB -nnodes 32
#BSUB -G uiuc
#BSUB -W 60
#BSUB -J iotest128
#BSUB -q pbatch
#BSUB -o iotest128.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 128 -n 128



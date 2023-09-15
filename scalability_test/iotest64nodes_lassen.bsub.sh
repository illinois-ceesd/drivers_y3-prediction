#!/bin/bash
#BSUB -nnodes 64
#BSUB -G uiuc
#BSUB -W 60
#BSUB -J iotest256
#BSUB -q pbatch
#BSUB -o iotest256.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 256 -n 256



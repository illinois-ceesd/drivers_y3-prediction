#!/bin/bash
#BSUB -nnodes 8
#BSUB -G uiuc
#BSUB -W 30
#BSUB -J iotest32
#BSUB -q pbatch
#BSUB -o iotest32.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 32 -n 32



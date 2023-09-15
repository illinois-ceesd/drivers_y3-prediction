#!/bin/bash
#BSUB -nnodes 16
#BSUB -G uiuc
#BSUB -W 30
#BSUB -J iotest64
#BSUB -q pbatch
#BSUB -o iotest64.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 64 -n 64



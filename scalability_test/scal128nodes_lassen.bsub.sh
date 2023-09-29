#!/bin/bash
#BSUB -nnodes 128
#BSUB -G uiuc
#BSUB -W 240
#BSUB -J scale512
#BSUB -q pbatch
#BSUB -o scal512.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 512 -n 512

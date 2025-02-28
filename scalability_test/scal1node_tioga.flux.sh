#!/bin/bash

#flux: --nodes=1
#flux: --time=360
#flux: --output=scal8.txt
##BSUB -J scale4

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -n 8 -N 1




#!/bin/bash

#flux: --nodes=16
#flux: --time=60
#flux: --output=scal64.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 64 -n 64 -N 16




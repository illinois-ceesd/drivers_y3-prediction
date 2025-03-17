#!/bin/bash

#flux: --nodes=8
#flux: --time=60
#flux: --output=scal32.txt

export MIRGE_CACHE_ROOT="./mirge-cache_8nodes"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 32 -n 32 -N 8




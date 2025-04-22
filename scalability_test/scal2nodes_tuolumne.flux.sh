#!/bin/bash

#flux: --nodes=2
#flux: --time=60
#flux: --output=scal8.txt

export MIRGE_CACHE_ROOT="./mirge-cache_2nodes"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
env | grep MIRGE
env | grep CACHE
source ../scripts/multi_scalability.sh -p ../ -s 8 -n 8 -N 2




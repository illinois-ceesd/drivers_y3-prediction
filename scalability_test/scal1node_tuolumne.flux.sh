#!/bin/bash

#flux: --nodes=1
#flux: --time=120
#flux: --output=scal4.txt

export MIRGE_CACHE_ROOT="./mirge-cache_1node"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -n 4 -N 1




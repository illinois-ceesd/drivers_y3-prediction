#!/bin/bash

#flux: --nodes=4
#flux: --time=60
#flux: --output=scal16.txt

export MIRGE_CACHE_ROOT="./mirge-cache_4nodes"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 16 -n 16 -N 4




#!/bin/bash

#flux: --nodes=64
#flux: --time=60
#flux: --output=scal256.txt

export MIRGE_CACHE_ROOT="./mirge-cache_64nodes"

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 256 -n 256 -N 64






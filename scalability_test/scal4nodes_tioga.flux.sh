#!/bin/bash

#flux: --nodes=4
#flux: --time=60
#flux: --output=scal32.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 32 -n 32 -N 4


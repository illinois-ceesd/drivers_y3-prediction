#!/bin/bash

#flux: --nodes=2
#flux: --time=60
#flux: --output=scal16.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 16 -n 16 -N 2


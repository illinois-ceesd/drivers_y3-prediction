#!/bin/bash

#flux: --nodes=32
#flux: --time=60
#flux: --output=scal128.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 128 -n 128 -N 32





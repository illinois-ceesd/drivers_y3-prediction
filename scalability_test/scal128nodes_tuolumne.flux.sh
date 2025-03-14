#!/bin/bash

#flux: --nodes=128
#flux: --time=60
#flux: --output=scal512.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 512 -n 512 -N 128






#!/bin/bash

#flux: --nodes=256
#flux: --time=240
#flux: --output=scal1024.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s 1024 -n 1024 -N 256






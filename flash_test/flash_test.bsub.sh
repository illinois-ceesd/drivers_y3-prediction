#! /bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 240
#BSUB -J flash1
#BSUB -q pbatch
#BSUB -o flash1.out

source ../emirge/config/activate_env.sh
export PYOPENCL_CTX="port:tesla"
export XDG_CACHE_ROOT="/tmp/$USER/xdg-scratch"
export POCL_CACHE_DIR_ROOT="/tmp/$USER/pocl-cache"
jsrun_cmd="jsrun -g 1 -a 1 -n 1"
$jsrun_cmd bash -c 'XDG_CACHE_HOME=$XDG_CACHE_ROOT/$$ POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./driver.py -i run_params.yaml --log --lazy'

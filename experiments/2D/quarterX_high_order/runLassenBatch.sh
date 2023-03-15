#! /bin/bash --login
#BSUB -nnodes 8
#BSUB -G uiuc
#BSUB -W 720
#BSUB -J 2d_pred_quarterX_n0_p3
#BSUB -q pbatch
#BSUB -o runOutput.txt
#BSUB -e runOutput.txt

module load gcc/8.3.1
module load spectrum-mpi
conda deactivate
conda activate mirgeDriver.Y3prediction

export PYOPENCL_CTX="port:tesla"
jsrun_cmd="jsrun -g 1 -a 1 -n 32"

export XDG_CACHE_DIR_ROOT="/tmp/$USER/xdg-scratch"
export POCL_CACHE_DIR_ROOT="/tmp/$USER/pocl-cache"
$jsrun_cmd js_task_info

$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ XDG_CACHE_DIR=$XDG_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./driver.py -i run_params.yaml --lazy --log > mirge-0.out'


#! /bin/bash --login
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 720
#BSUB -J 2d_pred_step1
#BSUB -q pbatch
#BSUB -o runOutput.txt
#BSUB -e runOutput.txt

module load gcc/8.3.1
module load spectrum-mpi
conda deactivate
conda activate mirgeDriver.Y3prediction

export PYOPENCL_CTX="port:tesla"
jsrun_cmd="jsrun -g 1 -a 1 -n 4"

export XDG_CACHE_ROOT="/tmp/$USER/xdg-scratch"
export POCL_CACHE_ROOT="/tmp/$USER/pocl-cache"
$jsrun_cmd js_task_info

$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_ROOT/$OMPI_COMM_WORLD_RANK XDG_CACHE_HOME=$XDG_CACHE_ROOT/$OMPI_COMM_WORLD_RANK python -O -u -m mpi4py ./driver.py -i run_params.yaml --log --lazy > mirge-1.out'


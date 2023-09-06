#! /bin/bash --login
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 60
#BSUB -J pred_smoke
#BSUB -q pdebug
#BSUB -o runOutput.txt
#BSUB -e runOutput.txt

module load gcc/8.3.1
module load spectrum-mpi
conda deactivate
conda activate mirgeDriver.Y3prediction
export PYOPENCL_CTX="port:tesla"
#export PYOPENCL_CTX="0:2"
jsrun_cmd="jsrun -g 1 -a 1 -n 1"
export XDG_CACHE_ROOT="/tmp/$USER/xdg-scratch"
export POCL_CACHE_ROOT="/tmp/$USER/pocl-cache"
$jsrun_cmd js_task_info
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_ROOT/$OMPI_COMM_WORLD_RANK XDG_CACHE_HOME=$XDG_CACHE_ROOT/$OMPI_COMM_WORLD_RANK python -u -m mpi4py ./driver.py -i run_params.yaml --lazy --log > mirge-0.out'


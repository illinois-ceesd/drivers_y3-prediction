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
conda activate mirgeDriver.Y2prediction
export PYOPENCL_CTX="port:tesla"
#export PYOPENCL_CTX="0:2"
jsrun_cmd="jsrun -g 1 -a 1 -n 1"
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
export POCL_CACHE_DIR_ROOT="/tmp/$USER/pocl-cache"
$jsrun_cmd js_task_info

# Create the mesh for a small, simple test (size==mesh spacing)
cd data
./mkmsh --size=48 --link  # will not overwrite existing actii.msh
cd ../

$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./driver.py -i run_params.yaml --log --lazy > mirge-1.out'

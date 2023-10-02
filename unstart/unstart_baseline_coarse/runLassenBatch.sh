#! /bin/bash --login
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 00:20
#BSUB -J unstart_test
#BSUB -q pdebug
#BSUB -o runOutput.txt
#BSUB -e runOutput.txt

module load gcc/8.3.1
module load spectrum-mpi
__conda_setup="$('${CONDA_PATH}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
  eval "$__conda_setup"
else
  if [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    . "${CONDA_PATH}/etc/profile.d/conda.sh"
  else
    export PATH="${CONDA_PATH}/bin:$PATH"
  fi
fi
unset __conda_setup
conda deactivate
conda activate mirgeDriver.Y3prediction
export PYOPENCL_CTX="port:tesla"
#export PYOPENCL_CTX="0:2"
jsrun_cmd="jsrun -g 1 -a 1 -n 4"
export XDG_CACHE_ROOT="/tmp/$USER/xdg-scratch"
export POCL_CACHE_ROOT="/tmp/$USER/pocl-cache"
$jsrun_cmd js_task_info
$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_ROOT/$OMPI_COMM_WORLD_RANK XDG_CACHE_HOME=$XDG_CACHE_ROOT/$OMPI_COMM_WORLD_RANK python -O -u -m mpi4py ./driver.py -i run_params.yaml --log --lazy --casename=prediction-eager > mirge-1.out'

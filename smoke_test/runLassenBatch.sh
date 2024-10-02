#! /bin/bash --login
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 60
#BSUB -J pred_smoke
#BSUB -q pdebug
#BSUB -o runOutput.txt
#BSUB -e runOutput.txt

module load gcc/12.2.1
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

nnodes=$(echo $LSB_MCPU_HOSTS | wc -w)
nnodes=$((nnodes/2-1))
nproc=$((4*nnodes)) # 4 ranks per node, 1 per GPU
echo nnodes=$nnodes nproc=$nproc
jsrun_cmd="jsrun -g 1 -a 1 -n $nproc"

# Reenable CUDA cache
export CUDA_CACHE_DISABLE=0

# MIRGE env vars used to setup cache locations
MIRGE_CACHE_ROOT=${MIRGE_CACHE_ROOT:-"$(pwd)/.mirge-cache/"}
XDG_CACHE_ROOT=${XDG_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/xdg-cache"}
CUDA_CACHE_ROOT=${CUDA_CACHE_ROOT:-"${MIRGE_CACHE_ROOT}/cuda-cache"}

echo "$MIRGE_CACHE_ROOT"
echo "$XDG_CACHE_ROOT"
echo "$CUDA_CACHE_ROOT"

# These vars are used by pocl, pyopencl, loopy, and cuda for cache location
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${XDG_CACHE_ROOT}/rank$OMPI_COMM_WORLD_RANK"}
CUDA_CACHE_PATH=${CUDA_CACHE_DIR:-"${CUDA_CACHE_ROOT}/rank$OMPI_COMM_WORLD_RANK"}
# The system sets a default CUDA_CACHE_PATH which is node-local :(
# User still has full path control, but we discard the system default
# CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-"${CUDA_CACHE_ROOT}/rank$OMPI_COMM_WORLD_RANK"}

export XDG_CACHE_HOME
export CUDA_CACHE_PATH

$jsrun_cmd js_task_info
$jsrun_cmd bash -c 'CUDA_CACHE_PATH=$CUDA_CACHE_PATH$OMPI_COMM_WORLD_RANK XDG_CACHE_HOME=$XDG_CACHE_HOME$OMPI_COMM_WORLD_RANK python -O -u -m mpi4py ./driver.py -i run_params.yaml --log --lazy --casename=prediction-eager > mirge-1.out'

#!/bin/bash
#SBATCH --nodes=1                # number of nodes
#SBATCH -t 00:30:00              # walltime (hh:mm:ss)
#SBATCH --partition=gpuA40x4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=single:1
#SBATCH --account=bcbs-delta-gpu
#SBATCH --exclusive              # dedicated node for this job
#SBATCH --no-requeue
#SBATCH --gpus-per-task=1
#SBATCH -o runOutput.txt
#SBATCH -e runOutput.txt

# Put any environment activation here, e.g.:
source ../emirge/config/activate_env.sh

# OpenCL device selection:
export PYOPENCL_CTX="port:nvidia"     # Run on Nvidia GPU with pocl
# export PYOPENCL_CTX="port:pthread"  # Run on CPU with pocl

nnodes=$SLURM_JOB_NUM_NODES
nproc=$((4*nnodes)) # 4 ranks per node, 1 per GPU

echo nnodes=$nnodes nproc=$nproc

srun_cmd="srun -N $nnodes -n $nproc"

export XDG_CACHE_HOME_ROOT="/tmp/$USER/xdg-scratch/rank"
export POCL_CACHE_DIR_ROOT="/tmp/$USER/pocl-cache/rank"

# Run application
$srun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT$SLURM_PROCID XDG_CACHE_HOME=$XDG_CACHE_HOME_ROOT$SLURM_PROCID python -u -O -m mpi4py driver.py -i run_params.yaml --casename=prediction-lazy --lazy > mirge-0.out'
mpirun -n 2 python -u -m mpi4py driver.py -i run_params.yaml --casename=prediction-eager

#!/bin/bash --login
#SBATCH --job-name=mesh_quarterX
#SBATCH --nodes=1
#SBATCH --partition=pbatch
#SBATCH --time=24:00:00
#SBATCH --output=out
#SBATCH --account=uiuc
#SBATCH --output=out

module load gcc/10.2.1
module load openmpi/4.1.0
conda deactivate
conda activate mirgeDriver.Y2isolator

export OMP_NUM_THREADS=36
srun -n 1 make_mesh.sh

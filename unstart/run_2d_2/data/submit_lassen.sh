#!/bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc 
#BSUB -J mesh
#BSUB -W 20
#BSUB -q pdebug
##BSUB -W 720
##BSUB -q pbatch
#BSUB -o output.txt

source /p/gpfs1/lauer5/drivers_y3-prediction/emirge/config/activate_env.sh


export OMP_NUM_THREADS=36
sh make_mesh.sh

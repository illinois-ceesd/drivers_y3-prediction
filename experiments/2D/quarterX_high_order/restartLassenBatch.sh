#! /bin/bash --login
#BSUB -nnodes 7
#BSUB -G uiuc
#BSUB -W 720
#BSUB -J 3d_pred_quarterX_noslip_n0_p3
#BSUB -q pbatch
#BSUB -o runOutput.txt
#BSUB -e runOutput.txt

module load gcc/8.3.1
module load spectrum-mpi
conda deactivate
conda activate mirgeDriver.Y2prediction

export PYOPENCL_CTX="port:tesla"
jsrun_cmd="jsrun -g 1 -a 1 -n 28"

export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
export POCL_CACHE_DIR_ROOT="/tmp/$USER/pocl-cache"

#
# get latest restart dump
#
fileList=(`ls -a restart_data/prediction-*.pkl`)
file=${fileList[${#fileList[@]}-1]}
export dumpFile=${file::-9}
#
# file for stdout
#
export outputFile=`$HOME/bin/getProgOutName.sh mirge`
echo "Writing output to ${outputFile}"

$jsrun_cmd js_task_info

$jsrun_cmd bash -c 'POCL_CACHE_DIR=$POCL_CACHE_DIR_ROOT/$$ python -O -u -m mpi4py ./prediction.py -i run_params.yaml -r ${dumpFile} -t restart_data/prediction-000000000 --lazy --log > ${outputFile}'


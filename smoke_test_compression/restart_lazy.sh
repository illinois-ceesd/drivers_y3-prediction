#!/bin/bash

#### MAKE SURE THAT THE LATEST RESTART FILE IS THE BIGGEST NUMBER

fileList=(`ls -a restart_data/prediction-*.pkl`)
file=${fileList[${#fileList[@]}-1]}
export dumpFile=${file::-9}
echo $dumpFile

mpirun -n 1 python -u -O -m mpi4py ./driver.py -i run_params.yaml -r restart_data/prediction-000116700 -t restart_data/prediction-000000000 --lazy >> mirge-0.out
# mpirun -n 2 python -u -O -m mpi4py ./driver.py -i run_params.yaml -r ${dumpFile} -t restart_data/prediction-000000000 > ${mirge-0.out}

#!/bin/bash
#
restart_files=`ls restart_data/*-0000.pkl`
for file in ${restart_files[@]}; do
  # remove the last 9 characters, the rank identifier and file extension
  # this typically "-0000.pkl"
  restart_file=`echo ${file} | sed 's/.\{9\}$//'`
  echo "making viz data from ${restart_file}"
  mpirun -n 2 python -u -O -m mpi4py driver.py -i make_viz_params.yaml -r $restart_file --log > /dev/null 2>&1 
done

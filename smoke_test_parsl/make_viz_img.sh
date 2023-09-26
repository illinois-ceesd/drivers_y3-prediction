#!/bin/bash
#
restart_files=`ls viz_data/*fluid*.pvtu`
for file in ${restart_files[@]}; do
  echo "making viz images from ${file}"
  ./run-paraview.sh viz_data/${file} > /dev/null 2>&1 
done

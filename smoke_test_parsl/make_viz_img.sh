#!/bin/bash
#
export pvpath="/Applications/ParaView-5.11.0.app/Contents/bin"
cd viz_data
viz_files=`ls *fluid*.pvtu`
cd ..
directory="`pwd`/viz_data"

for file in ${viz_files[@]}; do
  echo "making viz images from ${file}"
  dump=${file%.pvtu}
  dump_index=`echo $dump | cut -d - -f 3`
  echo "Running paraview in ${directory} on dump $dump_index"
  ${pvpath}/pvpython paraview-driver.py -f $directory/$file -p $directory -d $dump_index -i viz_config.py
done

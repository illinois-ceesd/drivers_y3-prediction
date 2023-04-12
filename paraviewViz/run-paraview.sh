#!/bin/bash

export pvpath="/Applications/ParaView-5.11.0.app/Contents/bin"
directory="`pwd`/viz_data"
file=$1
dump=${file%.pvtu}
dump_index=`echo $dump | cut -d - -f 3`
echo "Running paraview in ${directory} on dump $dump_index"
${pvpath}/pvpython paraview-driver.py -p $directory -d $dump_index



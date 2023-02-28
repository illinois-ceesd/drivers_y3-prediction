#!/bin/bash

export pvpath="/Applications/ParaView-5.9.1.app/Contents/bin"

############################################## Define function #################################################

# Each processor calls pvsingle
# Check consistency with startIter, stopIter outside this function definition
export SHELL=$(type -p bash)
function pvsingle() {
  base="/p/lscratchh/wang83/y6-runs/SimData"


  #prefix="upstream_full"
  #prefix="N2-injector"
  #prefix="arc-heater-3d"

  # Downstream
  #runName="Downstream/HalfX/VerticalInjection"
  #runName="Downstream/HalfX/HorizontalInjection"
  #runName="Downstream/1X/VerticalInjection"
  #runName="Downstream/1X/HorizontalInjection"
  runName="Downstream/1X/HorizontalInjectionLIB"

  #prefix="cavity-vert-3d"
  prefix="cavity-horz-3d"
  #prefix="cavity_z"
  #prefix="nozzle_down_z"
  #prefix="precavity_bl"
  #prefix="nozzle_to_cavity"

  # Downstream horizontal injection

  # Run Paraview
  #echo "pvbatch /g/g19/wang83/research/paraview/paraview-driver.py $base/$runName $prefix $1"
  #${pvpath}/pvbatch paraview-driver.py $base/$runName $prefix $1
  ${pvpath}/pvbatch paraview-driver.py 
}
export -f pvsingle # So GNU parallel can see it.


echo "sequence=" $(seq $startIter $skipIter $stopIter)
#parallel -j $numProcs pvsingle ::: $(seq $startIter $skipIter $stopIter)
pvsingle ::: $(seq $startIter $skipIter $stopIter)



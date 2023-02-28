#!/bin/bash

# Forward environment variables
#MSUB -V

# Notify by email
#MSUB -m e

# Partition and queue
#MSUB -l partition=pascal
#MSUB -q pdebug
#MSUB -A uiuc

# Combine stdout and stderr
#MSUB -j oe -o stdoe-pv-%j.log

# Request resources
#MSUB -l walltime=00:30:00
#MSUB -l nodes=1

# Upstream 30kw
#startIter=1442500
#skipIter=2500
#stopIter=1477500
#ppn=20 # ppn=20 to avoid OOM for large upstream solution

# Upstream 45kw
#startIter=1412500
#skipIter=2500
#stopIter=1565000
#ppn=7 # To avoid OOM, use ppn=20 for upstream slices, ppn=7 for upstream 3D contours

# ====== Downstream vertical injection ======
# (HalfX)
#startIter=133000
#skipIter=1000
#stopIter=133000
#ppn=12 # Use ppn=12 to avoid OOM with 3D contours

# (1X)
#startIter=430000
#skipIter=10000
#stopIter=500000
#ppn=3

# ====== Downstream horizontal injection ======

# (HalfX)
#startIter=0
#skipIter=1000
#stopIter=127000
#ppn=12

# (1X)
#startIter=130000
#skipIter=10000
#stopIter=220000
#ppn=3

# (1X with LIB)
startIter=4000
skipIter=1000
stopIter=10000
ppn=1

numProcs=$(($SLURM_NNODES*$ppn))

module load paraview/5.7.0

############################################## Define function #################################################

# Each processor calls pvsingle
# Check consistency with startIter, stopIter outside this function definition
export SHELL=$(type -p bash)
function pvsingle() {
  #base="/usr/workspace/wsa/xpacc/Y6/SimData"
  base="/p/lscratchh/wang83/y6-runs/SimData"

  # Upstream
  #runName="Upstream/HalfX/RampArc30kw"
  #runName="Upstream/HalfX/RampArc45kw"

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
  pvbatch /g/g19/wang83/research/paraview/paraview-driver.py $base/$runName $prefix $1
}
export -f pvsingle # So GNU parallel can see it.

############################################## Run the job #################################################

scontrol show job $SLURM_JOBID

# Announce
echo "--------------------------------------------------------"
echo "Starting run script "
echo "Number of nodes = "$SLURM_NNODES
echo "Number of processors = "$numProcs
echo "JOBID = "$SLURM_JOBID
echo "Hostname = "$(hostname)
echo "Submit directory = "$SLURM_SUBMIT_DIR
echo "Date = "$(date)
echo "--------------------------------------------------------"

echo "sequence=" $(seq $startIter $skipIter $stopIter)
parallel -j $numProcs pvsingle ::: $(seq $startIter $skipIter $stopIter)

# Announce
echo "--------------------------------------------------------"
echo "Finishing run script "
echo "Number of nodes = "$SLURM_NNODES
echo "Number of processors = "$numProcs
echo "JOBID = "$SLURM_JOBID
echo "Hostname = "$(hostname)
echo "Submit directory = "$SLURM_SUBMIT_DIR
echo "Date = "$(date)
echo "--------------------------------------------------------"


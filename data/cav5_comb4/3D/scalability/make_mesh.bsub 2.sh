#!/bin/bash
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 120
#BSUB -J mkmsh512
#BSUB -q pdebug
#BSUB -o mkmsh512.txt

source ../../../../emirge/config/activate_env.sh
./mkmsh --size=3.6386 --np=30
./mkmsh --size=3.6388 --np=30





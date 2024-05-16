#!/bin/bash

NCPUS=$(getconf _NPROCESSORS_ONLN)
gmsh -setnumber size 10 \
     -setnumber blrationozzle 2 \
     -setnumber blratiomodel 3 \
     -setnumber nozzlefac 4 \
     -setnumber modelfac 6 \
     -setnumber plumefac 3 \
     -setnumber spillfac 3 \
     -o actii_2d.msh -nopopup -format msh2 ./pseudoY0_2d_axi.geo -2 -nt $NCPUS

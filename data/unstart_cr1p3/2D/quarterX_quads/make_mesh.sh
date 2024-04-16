#!/bin/bash

NCPUS=$(getconf _NPROCESSORS_ONLN)
gmsh -setnumber size 10 \
     -setnumber blrationozzle 4 \
     -setnumber blratiomodel 3 \
     -setnumber nozzlefac 8 \
     -setnumber throatfac 16 \
     -setnumber modelfac 6 \
     -setnumber plumefac 3 \
     -setnumber spillfac 3 \
     -o actii_2d.msh -nopopup -format msh2 ./pseudoY0_2d_axi_quads.geo -2 -nt $NCPUS

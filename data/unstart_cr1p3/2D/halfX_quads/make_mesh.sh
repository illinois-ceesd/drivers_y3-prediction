#!/bin/bash

NCPUS=$(getconf _NPROCESSORS_ONLN)
gmsh -setnumber size 10 \
     -setnumber blrationozzle 8 \
     -setnumber blratiomodel 4 \
     -setnumber nozzlefac 10 \
     -setnumber throatfac 18 \
     -setnumber modelfac 8 \
     -setnumber plumefac 3 \
     -setnumber spillfac 3 \
     -o actii_2d.msh -nopopup -format msh2 ./pseudoY0_2d_axi_quads.geo -2 -nt $NCPUS

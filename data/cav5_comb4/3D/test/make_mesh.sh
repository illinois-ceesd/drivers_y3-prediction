#!/bin/bash

#NCPUS=1
NCPUS=$(getconf _NPROCESSORS_ONLN)
gmsh -setnumber size 12 -setnumber blratio 4 -setnumber cavityfac 4 -setnumber isofac 2 -setnumber injectorfac 5 -setnumber blratiocavity 2 -setnumber blratioinjector 2 -setnumber blratiosample 4 -setnumber blratiosurround 2 -setnumber shearfac 4 -o actii_3d.msh -nopopup -format msh2 ./actii_from_brep.geo -3 -nt $NCPUS

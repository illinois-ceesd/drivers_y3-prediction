#!/bin/bash

NCPUS=$(getconf _NPROCESSORS_ONLN)
#NCPUS=1
gmsh -setnumber size 12.8 -setnumber blratio 3 -setnumber nozzlefac 9 -setnumber cavityfac 4 -setnumber isofac 2 -setnumber injectorfac 4 -setnumber samplefac 4 -setnumber blratiocavity 2 -setnumber blratiocorner 4 -setnumber blrationozzle 2 -setnumber blratiocomb 6 -setnumber blratioinjector 2 -setnumber blratiosample 4 -setnumber blratiosurround 2 -setnumber shearfac 4 -o actii_3d.msh -nopopup -format msh2 ./actii_from_brep.geo -3 -nt $NCPUS
#gmsh -setnumber size 12.8 -setnumber blratio 3 -setnumber nozzlefac 9 -setnumber cavityfac 4 -setnumber isofac 2 -setnumber injectorfac 4 -setnumber samplefac 4 -setnumber blratiocavity 2 -setnumber blrationozzle 2 -setnumber blratiocomb 6 -setnumber blratioinjector 1 -setnumber blratiosample 4 -setnumber blratiosurround 2 -setnumber shearfac 4 -o actii_3d.msh -nopopup -format msh2 ./actii_from_brep.geo -3 -nt $NCPUS

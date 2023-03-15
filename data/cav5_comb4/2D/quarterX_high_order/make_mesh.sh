#!/bin/bash

NCPUS=$(getconf _NPROCESSORS_ONLN)
gmsh -setnumber size 6.4 -setnumber blratio 4 -setnumber cavityfac 2 -setnumber nozzlefac 4 -setnumber isofac 1 -setnumber injectorfac 2 -setnumber blratiocavity 2 -setnumber blratioinjector 2 -setnumber samplefac 2 -setnumber blratiosample 8 -setnumber blratiosurround 2 -setnumber shearfac 3 -o actii_2d.msh -nopopup -format msh2 ./actii_from_brep.geo -2 -nt $NCPUS

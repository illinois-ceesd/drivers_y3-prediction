#!/bin/bash

NCPUS=1
gmsh -setnumber size 6.4 -setnumber blratio 4 -setnumber cavityfac 6 -setnumber isofac 2 -setnumber injectorfac 4 -setnumber blratiocavity 2 -setnumber blratioinjector 2 -setnumber samplefac 4 -setnumber blratiosample 8 -setnumber blratiosurround 2 -setnumber shearfac 6 -o unstart_2d_asym.msh -nopopup -format msh2 ./unstart_from_brep_2d_asym.geo -2 -nt $NCPUS

if [ -f "unstart_2d_asym.msh" ]; then
    NCELLS=$(grep -cP '^[0-9]+\s4\b(?!.*")' actii_2d.msh)
    printf "Created mesh with ${NCELLS} tets.\n"
else
    printf "Mesh creation failed."
fi

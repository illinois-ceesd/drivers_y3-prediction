#!/bin/bash

NCPUS=1
gmsh -setnumber size 6.4 -setnumber blratio 10 -setnumber cavityfac 12 -setnumber isofac 10 -setnumber blratiocavity 10 -setnumber shearfac 6 -o unstart_test.msh -nopopup -format msh2 ./unstart_from_brep_2d.geo -2 -nt $NCPUS

if [ -f "unstart_coarse.msh" ]; then
    NCELLS=$(grep -cP '^[0-9]+\s4\b(?!.*")' actii_2d.msh)
    printf "Created mesh with ${NCELLS} tets.\n"
else
    printf "Mesh creation failed."
fi

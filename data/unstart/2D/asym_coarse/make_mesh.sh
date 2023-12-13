#!/bin/bash

NCPUS=1
gmsh -setnumber size 12.8 -setnumber blratio 4 -setnumber cavityfac 4 -setnumber isofac 2 -setnumber blratiocavity 4 -setnumber shearfac 6 -o unstart_2d_asym.msh -nopopup -format msh2 ./unstart_from_brep_2d_asym.geo -2 -nt $NCPUS

if [ -f "unstart_2d_asym.msh" ]; then
    NCELLS=$(grep -cP '^[0-9]+\s4\b(?!.*")' unstart_2d_asym.msh)
    printf "Created mesh with ${NCELLS} tets.\n"
else
    printf "Mesh creation failed."
fi

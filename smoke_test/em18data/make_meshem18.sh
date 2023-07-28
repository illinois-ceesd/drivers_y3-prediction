#!/bin/bash

NCPUS=1
gmsh -o compressedRamp.msh -nopopup -format msh2 ./compressedRamp.geo -2 -nt $NCPUS
# gmsh -o ramp1.msh -nopopup -format msh2 ./ramp1.geo -2 -nt $NCPUS

if [ -f "compressedRamp.msh" ]; then
    NCELLS=$(grep -cP '^[0-9]+\s4\b(?!.*")' compressedRamp.msh)
    printf "Created mesh with ${NCELLS} tets.\n"
else
    printf "Mesh creation failed."
fi

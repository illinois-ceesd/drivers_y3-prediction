#!/bin/bash

NCPUS=1
gmsh -o compressionRamp.msh -nopopup -format msh2 ./compressionRamp.geo -2 -nt $NCPUS
# gmsh -o ramp1.msh -nopopup -format msh2 ./ramp1.geo -2 -nt $NCPUS

if [ -f "compressionRamp.msh" ]; then
    NCELLS=$(grep -cP '^[0-9]+\s4\b(?!.*")' compressionRamp.msh)
    printf "Created mesh with ${NCELLS} tets.\n"
else
    printf "Mesh creation failed."
fi
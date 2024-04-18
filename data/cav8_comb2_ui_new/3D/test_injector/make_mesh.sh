#!/bin/bash

NCPUS=$(getconf _NPROCESSORS_ONLN)
gmsh ./create_geometry.geo -parse_and_exit
gmsh -o actii_3d.msh -nopopup -format msh2 ./create_mesh.geo -nt $NCPUS -parse_and_exit
python ../../../checkMesh.py actii_3d.msh

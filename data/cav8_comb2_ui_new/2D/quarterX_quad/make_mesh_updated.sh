#!/bin/bash

NCPUS=$(getconf _NPROCESSORS_ONLN)
gmsh ./create_geometry_updated.geo -parse_and_exit
gmsh -o actii_2d.msh -nopopup -format msh2 ./create_mesh_updated.geo -nt $NCPUS -parse_and_exit
python ../../../checkMesh.py actii_2d.msh

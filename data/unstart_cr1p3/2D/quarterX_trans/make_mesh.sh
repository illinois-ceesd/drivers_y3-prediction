#!/bin/bash

NCPUS=$(getconf _NPROCESSORS_ONLN)
# read the the brep geometry and export it to unrolled_geo
#gmsh -o geometry_trans.geo_unrolled -nopopup -format geo_unrolled pseudoY0_2d_axi_export_geom_trans.geo
# now apply the transfinite meshing
gmsh -o actii_2d.msh -nopopup -format msh2 pseudoY0_2d_axi_tris_trans_import.geo -2
python ../../../checkMesh.py actii_2d.msh

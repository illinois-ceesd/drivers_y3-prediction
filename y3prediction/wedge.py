"""mirgecom driver initializer for wedge."""
import numpy as np
from functools import partial


def get_mesh(dim, size, bl_ratio,
             transfinite=False, use_wall=True, use_quads=False,
             use_gmsh=True):
    """Generate a grid using `gmsh`."""

    from meshmode.mesh.io import (
        generate_gmsh,
        ScriptSource
    )

    lcar0 = 0.040
    lcar1 = 0.020
    theta = 10.0/180*np.pi

    # for 2D, the line segments/surfaces need to be specified clockwise to
    # get the correct facing (right-handed) surface normals
    my_string = (
        f"Point(1) = {{ 0.0, -0.6, 0.0, {lcar1}}};\n"
        f"Point(2) = {{ 0.0, 0.0,  0.0, {lcar0}}};\n"
        f"Point(3) = {{ 2.8*Tan({theta}), 2.8,  0, {lcar0}}};\n"
        f"Point(4) = {{ 3.0, 2.8,  0.0, {lcar0}}};\n"
        f"Point(5) = {{ 3.0, 0.0,  0.0, {lcar0}}};\n"
        f"Point(6) = {{ 3.0, -0.6,  0.0, {lcar0}}};\n"
        "Line(1) = {1, 2};\n"
        "Line(2) = {2, 3};\n"
        "Line(3) = {3, 4};\n"
        "Line(4) = {4, 5};\n"
        "Line(5) = {5, 6};\n"
        "Line(6) = {6, 1};\n"
        "Line(7) = {5, 2};\n"
        "Line Loop(1) = {1, -7, 5, 6};\n"
        "Line Loop(2) = {2, 3, 4, 7};\n"
        "Plane Surface(1) = {1};\n"
        "Plane Surface(2) = {2};\n"
        "Physical Line('inflow') = {6};\n"
        "Physical Line('outflow') = {3};\n"
        "Physical Line('slip_wall') = {2, 4, 5};\n"
        "Physical Line('symmetry') = {1};\n"
        "Physical Surface('fluid') = {1, 2};\n"
        "Transfinite Line {1} = 10 Using Progression 1.0;\n"
        "Transfinite Line {2} = 46 Using Progression 1.0;\n"
        "Transfinite Line {3} = 51 Using Progression 1.0;\n"
        "Transfinite Line {4} = 46 Using Progression 1.0;\n"
        "Transfinite Line {5} = 10 Using Progression 1.0;\n"
        "Transfinite Line {6} = 51 Using Progression 1.0;\n"
        "Transfinite Line {7} = 51 Using Progression 1.0;\n"
        "Transfinite Surface {1, 2};\n"
        )

    my_string += ("""
        Mesh.MeshSizeExtendFromBoundary = 0;
        Mesh.MeshSizeFromPoints = 0;
        Mesh.MeshSizeFromCurvature = 0;

        Mesh.Algorithm = 5;
        Mesh.OptimizeNetgen = 1;
        Mesh.Smoothing = 100;
    """)

    if use_quads:
        my_string += ("""
        // Convert the triangles back to quads
        Mesh.Algorithm = 6;
        Mesh.Algorithm3D = 1;
        Mesh.RecombinationAlgorithm = 2;
        Mesh.RecombineAll = 1;
        Recombine Surface {1, 2};
        """)

    print(my_string)
    return partial(generate_gmsh, ScriptSource(my_string, "geo"),
                   force_ambient_dim=dim, dimensions=dim, target_unit="M",
                   return_tag_to_elements_map=True)

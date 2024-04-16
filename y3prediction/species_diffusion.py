"""mirgecom driver initializer for wedge."""
from functools import partial


def get_mesh(dim, size, transfinite=False, use_quads=False):
    """Generate a grid using `gmsh`."""

    from meshmode.mesh.io import (
        generate_gmsh,
        ScriptSource
    )

    numpts = int(1./size)+1

    # for 2D, the line segments/surfaces need to be specified clockwise to
    # get the correct facing (right-handed) surface normals
    my_string = (
        f"Point(1) = {{ -0.5, -0.5, 0.0, {size}}};\n"
        f"Point(2) = {{ 0.5, -0.5,  0.0, {size}}};\n"
        f"Point(3) = {{ 0.5, 0.5, 0.0, {size}}};\n"
        f"Point(4) = {{ -0.5, 0.5,  0.0, {size}}};\n"
        "Line(1) = {1, 2};\n"
        "Line(2) = {2, 3};\n"
        "Line(3) = {3, 4};\n"
        "Line(4) = {4, 1};\n"
        "Line Loop(1) = {1:4};\n"
        "Plane Surface(1) = {-1};\n"
        "Physical Line('wall') = {1,2,3,4};\n"
        "Physical Surface('fluid') = {1};\n"
        f"Transfinite Line {{1:4}} = {numpts};\n"
        "Transfinite Surface {1};\n"
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

    #print(my_string)
    return partial(generate_gmsh, ScriptSource(my_string, "geo"),
                   force_ambient_dim=dim, dimensions=dim, target_unit="M",
                   return_tag_to_elements_map=True)

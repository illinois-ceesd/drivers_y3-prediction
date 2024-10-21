"""mirgecom driver initializer for 1d shock."""
import numpy as np
from functools import partial

def get_mesh(dim, size, bl_ratio, interface_ratio,
             transfinite=False, use_wall=True, use_quads=False,
             use_gmsh=True):
    """Generate a grid using `gmsh`."""

    if use_gmsh:
        from meshmode.mesh.io import (
            generate_gmsh,
            ScriptSource
        )

        # for 2D, the line segments/surfaces need to be specified clockwise to
        # get the correct facing (right-handed) surface normals
        my_string = (f"""
            Point(1) = {{ 0., 0., 0., {size}}};
            Point(2) = {{ 0.6, 0.0, 0., {size}}};
            Point(3) = {{ 0.6, 0.2, 0., {size}}};
            Point(4) = {{ 3.0, 0.2, 0., {size}}};
            Point(5) = {{ 3.0, 1.0, 0., {size}}};
            Point(6) = {{ 0.6, 1.0, 0., {size}}};
            Point(7) = {{ 0.0, 1.0, 0., {size}}};
            Point(8) = {{ 0.0, 0.2, 0., {size}}};

            Point(9) = {{ 0.45, 0.0, 0., {size}}};
            Point(10) = {{ 0.45, 0.2, 0., {size}}};
            Point(11) = {{ 0.45, 1.0, 0., {size}}};
            Point(12) = {{ 0.0, 0.35, 0., {size}}};
            Point(13) = {{ 0.6, 0.35, 0., {size}}};
            Point(14) = {{ 3.0, 0.35, 0., {size}}};
            Point(15) = {{ 0.45, 0.35, 0., {size}}};
        """)

        my_string += ("""
            Line(1) = {1, 9};
            Line(2) = {9, 10};
            Line(3) = {10, 8};
            Line(4) = {8, 1};
            Line(6) = {9, 2};
            Line(7) = {2, 3};
            Line(8) = {3, 10};
            Line(9) = {10, 15};
            Line(10) = {15, 12};
            Line(11) = {12, 8};
            Line(12) = {3, 13};
            Line(13) = {13, 15};
            Line(14) = {15, 11};
            Line(15) = {11, 7};
            Line(16) = {7, 12};
            Line(17) = {13, 6};
            Line(18) = {6, 11};
            Line(19) = {3, 4};
            Line(20) = {4, 14};
            Line(21) = {14, 13};
            Line(22) = {14, 5};
            Line(23) = {5, 6};
        """)

        my_string += ("""
            Line Loop(1) = {1, 2, 3, 4};
            Line Loop(2) = {6, 7, 8, -2};
            Line Loop(3) = {3, -11, -10, -9};
            Line Loop(4) = {8, 9, -13, -12};
            Line Loop(5) = {10, -16, -15, -14};
            Line Loop(6) = {13, 14, -18, -17};
            Line Loop(7) = {19, 20, 21, -12};
            Line Loop(8) = {21, 17, -23, -22};
        """)

        my_string += ("""
            Surface(1) = {1};
            Surface(2) = {2};
            Surface(3) = {3};
            Surface(4) = {4};
            Surface(5) = {5};
            Surface(6) = {6};
            Surface(7) = {7};
            Surface(8) = {8};

        """)

        my_string += ("""
            Physical Surface('fluid') = {1:8};
            Physical Curve('inflow') = {4, 11, 16};
            Physical Curve('outflow') = {20, 22};
            Physical Curve('flow') = {4, 11, 16, 20, 22};
            Physical Curve('isothermal_wall') = {1, 6, 7, 19, 15, 18, 23};
        """)

        if transfinite:
            my_string += (f"""
                Transfinite Curve {{1, 3, 10, 15}} = {0.45} / {size} + 1;
                Transfinite Curve {{16, 14, 17, 22}} = {0.65} / {size} + 1;
                Transfinite Curve {{11, 9, 12, 20}} = {0.15} / {size} + 1;
                Transfinite Curve {{2, 4, 7}} = {0.2} / {size} + 1;
                Transfinite Curve {{6, 8, 13, 18}} = {0.15} / {size} + 1;
                Transfinite Curve {{19, 21, 23}} = {2} / {size} + 1;
            """)
            my_string += (
                "Transfinite Surface {1:8} Right;"
            )

        else:
            my_string += (f"""
                // Create distance field from curves, excludes cavity
                Field[1] = Distance;
                Field[1].CurvesList = {{1,3}};
                Field[1].NumPointsPerCurve = 100000;

                //Create threshold field that varies element size near boundaries
                Field[2] = Threshold;
                Field[2].InField = 1;
                Field[2].SizeMin = {size} / {bl_ratio};
                Field[2].SizeMax = {size};
                Field[2].DistMin = 0.0002;
                Field[2].DistMax = 0.005;
                Field[2].StopAtDistMax = 1;

                //  background mesh size
                Field[3] = Box;
                Field[3].XMin = 0.;
                Field[3].XMax = 1.0;
                Field[3].YMin = -1.0;
                Field[3].YMax = 1.0;
                Field[3].VIn = {size};

                // Create distance field from curves, excludes cavity
                Field[4] = Distance;
                Field[4].CurvesList = {{2}};
                Field[4].NumPointsPerCurve = 100000;

                //Create threshold field that varies element size near boundaries
                Field[5] = Threshold;
                Field[5].InField = 4;
                Field[5].SizeMin = {size} / {interface_ratio};
                Field[5].SizeMax = {size};
                Field[5].DistMin = 0.0002;
                Field[5].DistMax = 0.005;
                Field[5].StopAtDistMax = 1;

                // take the minimum of all defined meshing fields
                Field[100] = Min;
                Field[100].FieldsList = {{2, 3, 5}};
                Background Field = 100;
            """)

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
            Recombine Surface {1, 2, 3};
            """)

        #print(my_string)
        mesh_construction_kwargs = {
            "force_positive_orientation":  True,
            "skip_element_orientation_test":  True}
        return partial(generate_gmsh, ScriptSource(my_string, "geo"),
                       force_ambient_dim=dim, dimensions=dim, target_unit="M",
                       mesh_construction_kwargs=mesh_construction_kwargs,
                       return_tag_to_elements_map=True)

    else:
        from meshmode.mesh.generation import generate_regular_rect_mesh

        # this only works for non-slanty meshes
        def get_meshmode_mesh(a, b, nelements_per_axis, boundary_tag_to_face):

            if use_quads:
                from meshmode.mesh import TensorProductElementGroup
                group_cls = TensorProductElementGroup
            else:
                group_cls = None

            mesh = generate_regular_rect_mesh(
                a=a, b=b, nelements_per_axis=nelements_per_axis,
                group_cls=group_cls,
                boundary_tag_to_face=boundary_tag_to_face
                )

            mgrp = mesh.groups[0]
            x = mgrp.nodes[0, :, :]
            x_avg = np.sum(x, axis=1)/x.shape[1]
            tag_to_elements = {
                "fluid": np.where(x_avg < fluid_length)[0],
                "wall": np.where(x_avg > fluid_length)[0]}

            return mesh, tag_to_elements

        if dim == 2:
            a = (bottom_inflow[0], bottom_inflow[1])
            b = (top_wall[0], top_wall[1])
            boundary_tag_to_face = {
                "inflow": ["-x"],
                "outflow": ["+x"],
                "flow": ["-x", "+x"],
                "isothermal_wall": ["-y", "+y"],
                "periodic_y_top": ["+y"],
                "periodic_y_bottom": ["-y"],
                "wall_farfield": ["+x"],
            }
            nelements_per_axis = (int(fluid_length/size) + int(wall_length/size),
                                  int(height/size))
        else:
            a = (bottom_inflow[0], bottom_inflow[1], 0.)
            b = (top_wall[0], top_wall[1], 0.02)
            boundary_tag_to_face = {
                "inflow": ["-x"],
                "outflow": ["+x"],
                "flow": ["-x", "+x"],
                "isothermal_wall": ["-y", "+y", "-z", "+z"],
                "wall_farfield": ["+x", "-y", "+y", "-z", "+z"]}
            nelements_per_axis = (int(fluid_length/size) + int(wall_length/size),
                                  int(height/size),
                                  int(height/size))
            #nelements_per_axis = (3, 2, 2)
            #print(f"{nelements_per_axis=}")

        return partial(get_meshmode_mesh,
                       a=a, b=b, boundary_tag_to_face=boundary_tag_to_face,
                       nelements_per_axis=nelements_per_axis)

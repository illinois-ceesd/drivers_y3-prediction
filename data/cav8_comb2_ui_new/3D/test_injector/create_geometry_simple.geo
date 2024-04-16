SetFactory("OpenCASCADE");
//Merge "pseudoY0_2d.brep";
// Fluid volume
Cylinder(1) = {0.0, 0.0, 0.0, 0., 10., 0., 1.};
Box(2) = {-5.0, 10.0, -5.0, 10., 2., 10.};
Box(3) = {-5.0, 12.0, -5.0, 10., 8., 10.};
Rotate { {0., 1., 0.}, {0., 0., 0.}, Pi/4} { Volume{1}; }
//+
BooleanFragments{ Volume{1}; Delete; }{Volume{2}; Delete;}


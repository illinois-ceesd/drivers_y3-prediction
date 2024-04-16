SetFactory("OpenCASCADE");
//Merge "pseudoY0_2d.brep";
// Fluid volume
Cylinder(1) = {0.0, 0.0, 0.0, 0., 10., 0., 1.};
Box(2) = {-5.0, 10.0, -5.0, 10., 2., 10.};
Box(3) = {-5.0, 12.0, -5.0, 10., 8., 10.};
Rotate { {0., 1., 0.}, {0., 0., 0.}, Pi/4} { Volume{1}; }
//+
BooleanFragments{ Volume{1}; Delete; }{Volume{2}; Delete;}

// injector boundary layer 
Dilate { { 0., 5., 0.}, {.6, 1., .6}} { Duplicata{ Volume{1}; } }

// split injector/bottom plane
// we have to delete the surfaces and volumes and reconstruct them to include the new edges
Line(100) = {22, 25};
Line(101) = {21, 26};
BooleanFragments{ Curve{100}; Curve{101}; Delete; }{ Curve{28, 43}; Delete; }

// delete the original injector volumes
// we will remake them
Recursive Delete { Curve{50, 51, 54, 55}; }
Recursive Delete { Volume{1,4}; }
//+
// delete the original bl volume 
Delete { Volume{2};} 
//Delete{ Surface{21};}
Delete{ Surface{17};}
Delete{ Surface{19:24}; }
Recursive Delete { Curve{28}; }
// remove the top volume we'll just remake it. didn't really need it to begin with I guess
Recursive Delete{ Volume{3};}
Recursive Delete{ Curve{42, 33, 41, 37, 32, 34, 38, 36}; }
//Coherence;
//
//Surface Loop(10) = {19, 20, 22:24, 25:33};
//Volume(10) = {10};

// reconstruct surfaces and volumes
Curve Loop(26) = {31, 52, -60, -48};
Plane Surface(25) = {26};
Curve Loop(27) = {52, 61, 47, -40};
Plane Surface(26) = {27};
Curve Loop(28) = {47, -35, -57, -58};
Plane Surface(27) = {28};
Curve Loop(29) = {57, -39, 48, -59};
Plane Surface(28) = {29};
Curve Loop(30) = {59, 49, -63, 56};
Plane Surface(29) = {30};
Curve Loop(31) = {58, -56, -62, 46};
Plane Surface(30) = {31};
Curve Loop(32) = {46, -61, 53, 65};
Plane Surface(31) = {32};
Curve Loop(33) = {60, 53, -64, -49};
Plane Surface(32) = {33};
Curve Loop(34) = {64, 65, 62, 63};
Plane Surface(33) = {34};

// Remake the bottom bl plane
Extrude {0, 2, 0} { Surface{25:33}; }
//
// Remake the injector
Extrude {0, -10, 0} { Surface{29:33}; }

// Remake the top volume
Extrude {0, 8, 0} { Curve{78, 80, 68, 84}; }
Curve Loop(140) = {122, 125, 124, 120};
Plane Surface(140) = {140};
Surface Loop(15) = {140, 80, 81, 83, 82, 42, 38, 46, 49, 56, 59, 61, 53, 62};
Volume(15) = {15};
Delete{ Surface{80:83};}

// make bl meshes for the top volume
Extrude {0, 2, 0} { Surface{140}; }
Extrude {0, 0, 2} { Surface{141}; }
Extrude {2, 0, 0} { Surface{142}; }
Extrude {0, 0, -2} { Surface{143}; }
Extrude {-2, 0, 0} { Surface{144}; }

Save "geometry_trans.geo_unrolled";
//+
//+

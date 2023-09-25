Point(1) = { 0.0,  -0.01, 0, 0.002};
Point(2) = { 0.1, -0.01,  0, 0.002};
Point(3) = { 0.1, 0.01,    0, 0.002};
Point(4) = { 0.0,  0.01,    0, 0.002};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(1) = {-4, -3, -2, -1};
Plane Surface(1) = {1};

fluid_surf_vec[] = Extrude {0, 0, 0.02 } { Surface{1}; };

Coherence;

Physical Volume('fluid') = {1};
Physical Surface('fluid_wall') = {1, 25, 26, 17};

// Create distance field from surfaces, fluid sides
Field[1] = Distance;
Field[1].SurfacesList = {1, 25, 26, 17};
Field[1].Sampling = 1000;

//Create threshold field that varrries element size near boundaries
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.002 / 3.0;
Field[2].SizeMax = 0.002 / 2.0;
Field[2].DistMin = 0.0002;
Field[2].DistMax = 0.005;
Field[2].StopAtDistMax = 1;

// Create distance field from surfaces, fluid sides
Field[11] = Distance;
Field[11].CurvesList = {1, 3, 7, 9};
Field[11].Sampling = 1000;

//Create threshold field that varrries element size near boundaries
Field[12] = Threshold;
Field[12].InField = 11;
Field[12].SizeMin = 0.002 / 6.0;
Field[12].SizeMax = 0.002 / 3.0;
Field[12].DistMin = 0.00002;
Field[12].DistMax = 0.005;
Field[12].StopAtDistMax = 1;

//  background mesh size
Field[3] = Box;
Field[3].XMin = 0.;
Field[3].XMax = 2.0;
Field[3].YMin = -1.0;
Field[3].YMax = 1.0;
Field[3].ZMin = -1.0;
Field[3].ZMax = 1.0;
Field[3].VIn = 0.002;

// take the minimum of all defined meshing fields
Field[100] = Min;
Field[100].FieldsList = {2, 3, 12};
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 100;

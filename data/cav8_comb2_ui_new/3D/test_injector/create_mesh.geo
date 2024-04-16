Merge "geometry_trans.geo_unrolled";
//
Mesh.ScalingFactor = 0.001;
//
// injector bl
//
num_bl_radius_injector = 7;
num_bl_theta_injector = 15;
num_bl_height_injector = 81;
//Transfinite Curve {-93, 95, 87, -90, -46, 53, 49, -56, 114, 102, -105, -110} = num_bl_radius_injector Using Progression 1.2;
Transfinite Curve {-93, 95, 87, -90, -46, 53, 49, -56, 114, 102, -105, -110} = num_bl_radius_injector Using Progression 1.;
Transfinite Curve {107, 112, 116, 100, 109, 115, 117, 104, 58, 61, 60, 59, 62, 65, 64, 63, 83, 75, 72, 85, 92, 96, 97, 89} = num_bl_theta_injector Using Progression 1;
//Transfinite Curve {106, 98, 108, 103, 113, 101, 111, 99} = num_bl_height_injector Using Progression 1.05;
Transfinite Curve {106, 98, 108, 103, 113, 101, 111, 99} = num_bl_height_injector Using Progression 1.;
Transfinite Surface {
    71, 75, 78, 67, 29, 30, 32, 31, 53, 61, 59, 56, 64,
    66, 70, 73, 50, 55, 52, 57, 
    68, 63, 76, 72, 74, 69, 65, 77, 66, 36, 39, 45, 48, 54, 58, 60, 51};
Transfinite Volume{5:8,10:13};

//
// wall bl
//
num_bl_radial_wall = 25;
num_bl_height_wall = 11;
Transfinite Curve {69, 94, 71, 86, 91, 74, 88, 81, 76, 79, 66, 67} = num_bl_height_wall Using Progression 1.2;
Transfinite Curve {84, 80, 78, 68, 39, 35, 40, 31} = num_bl_theta_injector Using Progression 1;
Transfinite Curve {-73, 82, 77, -70, -48, 57, 47, -52} = num_bl_radial_wall Using Progression 1.1;
Transfinite Surface {34, 47, 43, 41, 42, 46, 49, 38, 44, 40, 35, 37, 28, 27, 26, 25};
Transfinite Volume{1:4};

//
// top bl
//
num_bl_width_wall = 5;
num_bl_height_wall = 21;
Transfinite Curve {119, 140, 136, 118, 144, 148, 121, 152, 156, 164, 160, 123} = num_bl_height_wall Using Progression 1;
Transfinite Curve {138, 141, 128, 120, 125, 132, 157, 154, 141, 162, 165, 124, 133, 130, 122, 149, 146} = num_bl_theta_injector Using Progression 1;
Transfinite Curve {158, 161, 159, 163, 142, 145, 147, 143, 135, 139, 137, 134, 150, 153, 151, 155, 129, 131, 127, 126} = num_bl_width_wall Using Progression 1;
Transfinite Surface {
    159, 142, 155, 158, 157, 156,
164, 143, 163, 162, 160, 161, 169,
144, 168, 166, 165, 167, 154, 141,
152, 150, 153, 151, 149, 140, 148,
145, 146, 147};
Transfinite Volume{16:20};

// background mesh size
Field[1] = Box;
Field[1].XMin = -100.;
Field[1].XMax = 100.;
Field[1].YMin = -100.;
Field[1].YMax = 100.;
Field[1].ZMin = -100;
Field[1].ZMax = 100;
Field[1].Thickness = 100;
Field[1].VIn = 2;
Field[1].VOut = 100;

// injector mesh size
Field[2] = Box;
Field[2].XMin = -100.;
Field[2].XMax = 100.;
Field[2].YMin = -100.;
Field[2].YMax = 12.;
Field[2].ZMin = -100;
Field[2].ZMax = 100;
Field[2].Thickness = 100;
Field[2].VIn = 0.1;
Field[2].VOut = 100;

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[3] = Distance;
Field[3].SurfacesList = {56, 59, 61, 53, 62};
Field[3].Sampling = 1000;

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[13] = Distance;
Field[13].SurfacesList = {42, 46, 49, 38};
Field[13].Sampling = 5000;
////
//Create threshold field that varrries element size near boundaries
Field[4] = Threshold;
Field[4].InField = 13;
Field[4].SizeMin = .05;
Field[4].SizeMax = 10;
Field[4].DistMin = 0.02;
Field[4].DistMax = 50;
Field[4].StopAtDistMax = 1;

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[5] = Distance;
Field[5].CurvesList = {78, 68, 84, 80};
Field[5].Sampling = 1000;
////
//Create threshold field that varrries element size near boundaries
Field[6] = Threshold;
Field[6].InField = 5;
Field[6].SizeMin = .4;
Field[6].SizeMax = 10;
Field[6].DistMin = 0.02;
Field[6].DistMax = 20;
Field[6].StopAtDistMax = 1;

// take the minimum of all defined meshing fields
Field[100] = Min;
Field[100].FieldsList = {1, 2, 4, 6};
Background Field = 100;

//Recombine Surface{3:27};
//Recombine Surface{100, 101};
//Recombine Surface{200, 202};

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
//Mesh.Algorithm = 5; // Delaunay
Mesh.Algorithm = 6; // Frontal-Delaunay
//Mesh.Algorithm = 8; // Frontal-Delaunay for quads
Mesh.Algorithm3D = 1;
Mesh.RecombinationAlgorithm = 1;
Mesh.RecombineOptimizeTopology =5;
//Mesh.RecombineAll = 0;
Mesh.RecombineAll = 1;

Mesh 2;
Mesh 3;
RecombineMesh;
Mesh.SubdivisionAlgorithm = 2; // all hexes
RefineMesh;
Mesh.MshFileVersion = 2.2;
Save "actii_3d.msh";
//+
//+
//+

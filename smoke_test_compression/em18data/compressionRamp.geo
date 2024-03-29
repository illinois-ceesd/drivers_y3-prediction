// Millimeters to meters
Mesh.ScalingFactor = 1;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {-1.280, 0, 0, 1.0};
Point(3) = {.640, .28495, 0, 1.0};
Point(4) = {-1.280, 1.0, 0, 1.0};
Point(6) = {.640, 1.0, 0, 1.0};
// Point{12} = {-.7, 0, 0, 1.0};

Line(1) = {4, 6};
Line(3) = {6, 3};
Line(4) = {3, 1};
Line(5) = {1, 2};
Line(6) = {2, 4};

Curve Loop(1) = {1, 3, 4, 5, 6};
//+d
Plane Surface(1) = {1};
//+
Physical Curve("inflow", 7) = {6};
//+
Physical Curve("outflow", 8) = {3};
//+
Physical Curve("isothermal_wall", 9) = {5, 4};
//+
Physical Curve("flow") = {1, 3, 6}; // all inflow/outflow
//+
// Physical Curve("wall_farfield") = {4, 5}; 
// 
Physical Surface("fluid", 11) = {1};



Field[1] = Distance;
Field[1].CurvesList = {
    1, //top
    5, //bottom
    4  //ramp
}; 

Field[1].Sampling = 100;

size = 1;

//Create threshold field that varies element size near boundaries
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = size / 6;
Field[2].SizeMax = size / 2.3;
Field[2].DistMin = 0.02;
Field[2].DistMax = .08;

//Field[2].StopAtDistMax = .1;

// //Create Distance Field for the corner.
// Field[3] = Distance;ermi
// Field[3].PointsList = {1};

// Field[4] = Threshold;
// Field[4].InField = 3;
// Field[4].SizeMin = size / 10;
// Field[4].SizeMax = size / 5;
// Field[4].DistMin = 0.1;
// Field[4].DistMax = .4;

// Field[100] = Distance;
// Field[100].CurvesList = {
//     1 //top
// }; 

// Field[100].Sampling = 100;

// size = 1;

// Field[100] = Threshold;
// Field[100].InField = 1;
// Field[100].SizeMin = size / 7;
// Field[100].SizeMax = size / 2.3;
// Field[100].DistMin = 0.02;
// Field[100].DistMax = .08;

// Max field size
Field[10] = Max;
Field[10].FieldsList = {2};
Background Field = 10;


Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

//Mesh.MeshSizeFactor = 0.05;

Mesh.Algorithm = 8;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 100;
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {-1280, 0, 0, 1.0};
//+
Point(3) = {640, 284.95, 0, 1.0};
//+
Point(4) = {-1280, 128, 0, 1.0};
//+
Point(5) = {0, 128, 0, 1.0};
//+
Point(6) = {640, 412.95, 0, 1.0};
//+
Line(1) = {4, 5};
//+
Line(2) = {5, 6};
//+
Line(3) = {6, 3};
//+
Line(4) = {3, 1};
//+
Line(5) = {1, 2};
//+
Line(6) = {2, 4};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, 6};
//+
Plane Surface(1) = {1};
//+
Physical Curve("inflow", 7) = {6};
//+
Physical Curve("outflow", 8) = {3};
//+
Physical Curve("isothermal_wall", 9) = {1, 5, 2, 4};
//+
Physical Curve("flow") = {3, 6}; // all inflow/outflow
//+
Physical Surface("fluid", 11) = {1};
//+

Field[1].CurvesList = {1, 5, 2, 4};
Background Field = 100;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;


// Millimeters to meters
Mesh.ScalingFactor = 1;
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {-1.280, 0, 0, 1.0};
//+
Point(3) = {.640, .28495, 0, 1.0};
//+
Point(4) = {-1.280, .512, 0, 1.0};
//+
Point(6) = {.640, .512, 0, 1.0};
//+
Line(1) = {4, 6};
//+
Line(3) = {6, 3};
//+
Line(4) = {3, 1};
//+
Line(5) = {1, 2};
//+
Line(6) = {2, 4};
//+
Curve Loop(1) = {1, 3, 4, 5, 6};
//+
Plane Surface(1) = {1};
//+
Physical Curve("inflow", 7) = {6};
//+
Physical Curve("outflow", 8) = {3};
//+
Physical Curve("isothermal_wall", 9) = {1, 5};
//+
Physical Curve("injection", 20) = {4};
//+
Physical Curve("flow") = {3, 6}; // all inflow/outflow
//+
Physical Surface("fluid", 11) = {1};
//+


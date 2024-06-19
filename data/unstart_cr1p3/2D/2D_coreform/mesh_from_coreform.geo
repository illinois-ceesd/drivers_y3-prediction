Merge "base.bdf";

// Millimeters to meters
Mesh.ScalingFactor = 0.001;

Physical Surface('fluid') = {6, 8:10, 17, 20:25, 27, 30:32, 34, 35:37};
Physical Curve("symmetry") = {2}; // inlet
Physical Curve("inflow") = {3}; // outlet
Physical Curve("outflow") = {4}; // injection
Physical Curve("isothermal_wall") = {5}; // injection

Mesh.MshFileVersion = 2.2;
Save "base.msh";

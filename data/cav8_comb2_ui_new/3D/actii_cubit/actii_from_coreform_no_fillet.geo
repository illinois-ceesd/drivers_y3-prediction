//surface_vector[] = ShapeFromFile("actii-3d.brep");
//Merge "actii-3d.brep";
Merge "actii_no_fillet.bdf";

// Millimeters to meters
Mesh.ScalingFactor = 0.001;

Physical Volume('fluid') = {3};
Physical Volume('wall_insert') = {1};
Physical Volume('wall_surround') = {2};

Physical Surface("inflow") = {4}; // inlet
Physical Surface("outflow") = {8}; // outlet
Physical Surface("injection") = {6}; // injection
Physical Surface("upstream_injection") = {5}; // injection
Physical Surface("flow") = {4:6,8}; // injection
Physical Surface('isothermal_wall') = {9};
Physical Surface('wall_farfield') = {7};

Mesh.MshFileVersion = 2.2;
Save "actii_no_fillet_3d.msh";

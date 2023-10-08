SetFactory("OpenCASCADE");
surface_vector[] = ShapeFromFile("actii-3d.brep");
//Merge "actii-3d.brep";

// Millimeters to meters
Mesh.ScalingFactor = 0.001;

If(Exists(size))
    basesize=size;
Else
    basesize=6.4;
EndIf

If(Exists(blratio))
    boundratio=blratio;
Else
    boundratio=1.0;
EndIf

If(Exists(blratioinjector))
    boundratioinjector=blratioinjector;
Else
    boundratioinjector=1.0;
EndIf

If(Exists(injectorfac))
    injector_factor=injectorfac;
Else
    injector_factor=5.0;
EndIf

If(Exists(isofac))
    iso_factor=isofac;
Else
    iso_factor=1.0;
EndIf

cavityAngle=45;
inj_h=4.;  // height of injector (bottom) from floor
inj_d=1.59; // diameter of injector
inj_l = 20; // length of injector

bigsize = basesize*4;     // the biggest mesh size 
inletsize = basesize;   // background mesh size upstream of the nozzle
isosize = basesize/iso_factor;       // background mesh size in the isolator
injectorsize = inj_d/injector_factor; // background mesh size in the injector region

Printf("basesize = %f", basesize);
Printf("inletsize = %f", inletsize);
Printf("isosize = %f", isosize);
Printf("injectorsize = %f", injectorsize);
Printf("boundratio = %f", boundratio);
Printf("boundratioinjector = %f", boundratioinjector);

Geometry.Tolerance = 1.e-3;
Coherence;

Physical Volume('fluid') = {1};

Physical Surface("inflow") = {6}; // inlet
Physical Surface("outflow") = {5}; // outlet
Physical Surface("injection") = {3}; // injection
Physical Surface("flow") = {6, 5, 3}; // injection
Physical Surface('isothermal_wall') = {
    7, // fore wall
    4, // aft wall
    8, // isolator top
    2, // isolator bottom
    1 // upstream injector
};

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[1] = Distance;
Field[1].SurfacesList = {
    7, // fore wall
    4, // aft wall
    8, // isolator top
    2 // isolator bottom
};
Field[1].Sampling = 1000;
////
//Create threshold field that varrries element size near boundaries
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = isosize/boundratio;
Field[2].SizeMax = isosize/boundratio*(2.-1./boundratio);
//Field[2].SizeMax = isosize / 1.5;
Field[2].DistMin = 0.02;
Field[2].DistMax = 6;
Field[2].StopAtDistMax = 1;

 
// Create distance field from corners for wall meshing, excludes cavity, injector
Field[3001] = Distance;
Field[3001].CurvesList = {
    15, // top fore
    9, // top aft
    4, // bottom aft
    7  // bottom fore
};
Field[3001].Sampling = 1000;
////
//Create threshold field that varrries element size near boundaries
blratiocorner = boundratio/4.;
Field[3002] = Threshold;
Field[3002].InField = 3001;
Field[3002].SizeMin = isosize/blratiocorner;
Field[3002].SizeMax = isosize/blratiocorner*(2.-1./blratiocorner);
Field[3002].DistMin = 0.02;
Field[3002].DistMax = 6.0;
Field[3002].StopAtDistMax = 1;

// Create distance field from curves, injector only
Field[13] = Distance;
Field[13].SurfacesList = {
1 // injector wall
};
Field[13].Sampling = 1000;
//
//Create threshold field that varrries element size near boundaries
Field[14] = Threshold;
Field[14].InField = 13;
Field[14].SizeMin = injectorsize / boundratioinjector;
//Field[14].SizeMax = injectorsize;
//Field[14].SizeMax = bigsize;
Field[14].SizeMax = injectorsize/boundratioinjector*(2.-1./boundratioinjector);
Field[14].DistMin = 0.001;
Field[14].DistMax = 0.5;
Field[14].StopAtDistMax = 1;

// Create distance field from corners for wall meshing, excludes cavity, injector
Field[4001] = Distance;
Field[4001].CurvesList = {
    1 // injector/isolator edge
};
Field[4001].Sampling = 1000;
////
//Create threshold field that varrries element size near boundaries
blratiocorner = boundratioinjector;
//blratiocorner = boundratioinjector;
Field[4002] = Threshold;
Field[4002].InField = 4001;
Field[4002].SizeMin = injectorsize/blratiocorner;
Field[4002].SizeMax = injectorsize/blratiocorner*(2.-1./blratiocorner);
Field[4002].DistMin = 0.02;
Field[4002].DistMax = 2.0;
Field[4002].StopAtDistMax = 1;

// a smaller region right at the corner
blratiocorner = boundratioinjector*2;
//blratiocorner = boundratioinjector;
Field[4003] = Threshold;
Field[4003].InField = 4001;
Field[4003].SizeMin = injectorsize/blratiocorner;
Field[4003].SizeMax = injectorsize/blratiocorner*(2.-1./blratiocorner);
Field[4003].DistMin = 0.02;
Field[4003].DistMax = 1.0;
Field[4003].StopAtDistMax = 1;

//  background mesh size in the isolator (downstream of the nozzle)
Field[3] = Box;
Field[3].XMin = 0.2;
Field[3].XMax = 1000.0;
Field[3].YMin = -1000.0;
Field[3].YMax = 1000.0;
Field[3].ZMin = -1000.0;
Field[3].ZMax = 1000.0;
Field[3].VIn = isosize;
Field[3].VOut = bigsize;
//
// background mesh size in the upstream injection region
Field[217] = Cylinder;
Field[217].YAxis = 17.5;
Field[217].XCenter =  533.2;
Field[217].YCenter = -22.5;
Field[217].ZCenter = 0.;
Field[217].Radius = 3;
//Field[217].VIn = isosize/4;
Field[217].VIn = injectorsize;
Field[217].VOut = bigsize;

// background mesh size between upstream injection and cavity
Field[218] = Cylinder;
Field[218].XAxis = 65;
Field[218].XCenter =  590;
Field[218].YCenter = -9;
Field[218].ZCenter = 0.;
Field[218].Radius = 6;
Field[218].VIn = isosize/blratio/2.;
//Field[218].VIn = injectorsize + 0.5*(isosize/blratio - injectorsize);
Field[218].VOut = bigsize;

// background mesh size between upstream injection and cavity
Field[219] = Cylinder;
Field[219].XAxis = 85;
Field[219].XCenter =  590;
Field[219].YCenter = -9;
Field[219].ZCenter = 0.;
Field[219].Radius = 9;
Field[219].VIn = isosize/blratio;
Field[219].VOut = bigsize;

// background mesh size between upstream injection and cavity
Field[220] = Cylinder;
Field[220].XAxis = 5;
Field[220].XCenter =  535;
Field[220].YCenter = -9;
Field[220].ZCenter = 0.;
Field[220].Radius = 5;
Field[220].VIn = isosize/blratio/3.;
//Field[218].VIn = injectorsize + 0.5*(isosize/blratio - injectorsize);
Field[220].VOut = bigsize;

// background mesh size in the isolator-injector/cavity/combustor region
shear_start_x = 520;
shear_end_x = 680;
shear_start_y = -10;
shear_end_y = 10;
shear_start_z = -1000.0;
shear_end_z = 1000.0;
Field[119] = Box;
Field[119].XMin = shear_start_x;
Field[119].XMax = shear_end_x;
Field[119].YMin = shear_start_y;
Field[119].YMax = shear_end_y;
Field[119].ZMin = shear_start_z;
Field[119].ZMax = shear_end_z;
Field[119].Thickness = 100;
Field[119].VIn = isosize/2.;
Field[119].VOut = bigsize;

// take the minimum of all defined meshing fields
Field[100] = Min;
Field[100].FieldsList = {
2 ,3002, 
4002, 4003,
14, 3, 217, 218, 219, 220, 119
    };
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;


// Delaunay, seems to respect changing mesh sizes better
// Mesh.Algorithm3D = 1;
// Frontal, makes a better looking mesh, will make bigger elements where I don't want them though
// Doesn't repsect the mesh sizing parameters ...
//Mesh.Algorithm3D = 4;
//Mesh.Algorithm = 8;
// HXT, re-implemented Delaunay in parallel
Mesh.Algorithm3D = 10;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 100;
//Mesh.Smoothing = 0;
//Mesh.OptimizeNetgen = 0;

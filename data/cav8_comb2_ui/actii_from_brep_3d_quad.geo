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

If(Exists(blratiocavity))
    boundratiocavity=blratiocavity;
Else
    boundratiocavity=1.0;
EndIf

If(Exists(blratiocorner))
    boundratiocorner=blratiocorner;
Else
    boundratiocorner=1.0;
EndIf


If(Exists(blratiocomb))
    boundratiocomb=blratiocomb;
Else
    boundratiocomb=1.0;
EndIf

If(Exists(blratioinjector))
    boundratioinjector=blratioinjector;
Else
    boundratioinjector=1.0;
EndIf

If(Exists(blrationozzle))
    boundrationozzle=blrationozzle;
Else
    boundrationozzle=1.0;
EndIf

If(Exists(blratiosample))
    boundratiosample=blratiosample;
Else
    boundratiosample=1.0;
EndIf

If(Exists(blratiosurround))
    boundratiosurround=blratiosurround;
Else
    boundratiosurround=1.0;
EndIf

If(Exists(injectorfac))
    injector_factor=injectorfac;
Else
    injector_factor=5.0;
EndIf

If(Exists(shearfac))
    shear_factor=shearfac;
Else
    shear_factor=1.0;
EndIf

If(Exists(isofac))
    iso_factor=isofac;
Else
    iso_factor=1.0;
EndIf

If(Exists(cavityfac))
    cavity_factor=cavityfac;
Else
    cavity_factor=1.0;
EndIf

If(Exists(nozzlefac))
    nozzle_factor=nozzlefac;
Else
    nozzle_factor=3.0;

EndIf

If(Exists(samplefac))
    sample_factor=samplefac;
Else
    sample_factor=2.0;
EndIf

// horizontal injection
cavityAngle=45;
inj_h=4.;  // height of injector (bottom) from floor
inj_d=1.59; // diameter of injector
inj_l = 20; // length of injector

bigsize = basesize*4;     // the biggest mesh size 
inletsize = basesize;   // background mesh size upstream of the nozzle
isosize = basesize/iso_factor;       // background mesh size in the isolator
nozzlesize = basesize/nozzle_factor;       // background mesh size in the nozzle
cavitysize = basesize/cavity_factor; // background mesh size in the cavity region
shearsize = isosize/shear_factor; // background mesh size in the shear region
samplesize = basesize/sample_factor;       // background mesh size in the sample
injectorsize = inj_d/injector_factor; // background mesh size in the injector region

Printf("basesize = %f", basesize);
Printf("inletsize = %f", inletsize);
Printf("isosize = %f", isosize);
Printf("nozzlesize = %f", nozzlesize);
Printf("cavitysize = %f", cavitysize);
Printf("shearsize = %f", shearsize);
Printf("injectorsize = %f", injectorsize);
Printf("samplesize = %f", samplesize);
Printf("boundratio = %f", boundratio);
Printf("boundratiocavity = %f", boundratiocavity);
Printf("boundratiocombustor = %f", boundratiocomb);
Printf("boundratioinjector = %f", boundratioinjector);
Printf("boundratiosample = %f", boundratiosample);
Printf("boundratiosurround = %f", boundratiosurround);

Geometry.Tolerance = 1.e-3;
Coherence;

Physical Volume('fluid') = {3};
Physical Volume('wall_insert') = {1};
Physical Volume('wall_surround') = {2};

Physical Surface("inflow") = {36}; // inlet
Physical Surface("outflow") = {29}; // outlet
Physical Surface("injection") = {38}; // injection
Physical Surface("upstream_injection") = {19}; // injection
Physical Surface("flow") = {36, 29, 38, 19}; // injection
Physical Surface('isothermal_wall') = {
    22, // fore wall
    21, // aft wall
    25, // inflow top
    26, // inflow ramp top
    27, // nozzle top
    28, // isolator top
    35, // inflow bottom
    24, // inflow ramp bottom
    20, // inflow ramp bottom
    18, // isolator bottom
    23, // cavity front
    34, // cavity bottom
    33, // cavity back (ramp)
    32, // post-cavity flat
    31, // combustor bottom
    30, // combustor flat
    17, // upstream injector
    37 // injector
};

Physical Surface('wall_farfield') = {
    11, 12, 14, 16, 15 // cavity surround
};

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[1] = Distance;
Field[1].SurfacesList = {
    22, // fore wall
    21, // aft wall
    25, // inflow top
    26, // inflow ramp top
    27, // nozzle top
    28, // isolator top
    35, // inflow bottom
    24, // inflow ramp bottom
    20, // inflow ramp bottom
    18 // isolator bottom
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
 
//Create threshold field that varrries element size near boundaries
//this is for the nozzle only
Field[2002] = Threshold;
Field[2002].InField = 1;
Field[2002].SizeMin = nozzlesize/boundrationozzle;
//Field[2002].SizeMax = isosize;
Field[2002].SizeMax = nozzlesize/boundrationozzle*(2.-1./boundrationozzle);
Field[2002].DistMin = 0.02;
Field[2002].DistMax = 5;
Field[2002].StopAtDistMax = 1;

//Create threshold field that varrries element size near boundaries
//this is for the nozzle expansion only
Field[2003] = Threshold;
Field[2003].InField = 1;
Field[2003].SizeMin = 1.5*nozzlesize / boundrationozzle;
//Field[2003].SizeMax = isosize;
Field[2002].SizeMax = 1.5*nozzlesize/boundrationozzle*(2.-1./boundrationozzle);
Field[2003].DistMin = 0.02;
Field[2003].DistMax = 5;
Field[2003].StopAtDistMax = 1;

sigma = 25;
nozzle_start = 270;
nozzle_end = 325;
nozzle_exp_end = 375;

// restrict the nozzle bl meshing to the nozzle only
Field[2010] = MathEval;
Field[2010].F = Sprintf("F2 + (F2002 - F2)*(0.5*(1.0 - tanh(%g*(x - %g))))*(0.5*(1.0 - tanh(%g*(%g - x))))", sigma, nozzle_end, sigma, nozzle_start);

// restrict the nozzle expansion bl meshing to the nozzle expansion only
Field[2011] = MathEval;
Field[2011].F = Sprintf("F2 + (F2003 - F2)*(0.5*(1.0 - tanh(%g*(x - %g))))*(0.5*(1.0 - tanh(%g*(%g - x))))", sigma, nozzle_exp_end, sigma, nozzle_end);

// Create distance field from corners for wall meshing, excludes cavity, injector
Field[3001] = Distance;
Field[3001].CurvesList = {
    63, 64, // top fore
    49, 50, // top aft
    41, 54, 32, 53, 52, // bottom aft
    42, 68, 29, 67, 66  // bottom fore
};
Field[3001].Sampling = 1000;
////
//Create threshold field that varrries element size near boundaries
cornerfac = boundratio/boundratiocorner;
Field[3002] = Threshold;
Field[3002].InField = 3001;
Field[3002].SizeMin = isosize/cornerfac;
Field[3002].SizeMax = isosize/cornerfac*(2.-1./cornerfac);
Field[3002].DistMin = 0.02;
Field[3002].DistMax = 6.0;
Field[3002].StopAtDistMax = 1;

// Create distance field from surfaces for wall meshing in the combustor
Field[101] = Distance;
Field[101].SurfacesList = {
    31, // combustor
    32, // post-cavity flat
    30 // combustor flat
};
Field[101].Sampling = 1000;
////
//Create threshold field that varrries element size near boundaries
Field[102] = Threshold;
Field[102].InField = 101;
Field[102].SizeMin = isosize/boundratiocomb;
//Field[102].SizeMax = isosize/3;
//Field[102].SizeMax = isosize;
Field[102].SizeMax = isosize/boundratiocomb*(2.-1./boundratiocomb);
Field[102].DistMin = 0.02;
Field[102].DistMax = 8;
Field[102].StopAtDistMax = 1;
//
// Create distance field from curves, cavity only
Field[11] = Distance;
Field[11].SurfacesList = {
    23, // cavity front
    34, // cavity bottom
    33 // cavity back (ramp)
};
Field[11].Sampling = 1000;

//Create threshold field that varies element size near boundaries
Field[12] = Threshold;
Field[12].InField = 11;
Field[12].SizeMin = cavitysize / boundratiocavity;
//Field[12].SizeMax = cavitysize;
Field[12].SizeMax = cavitysize/boundratiocavity*(2.-1./boundratiocavity);
Field[12].DistMin = 0.02;
Field[12].DistMax = 6;
Field[12].StopAtDistMax = 1;

// Create distance field from curves, injector only
Field[13] = Distance;
Field[13].SurfacesList = {
37, 17 // injector wall
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
Field[14].DistMax = 1.0;
Field[14].StopAtDistMax = 1;

// Create distance field from corners for wall meshing, excludes cavity, injector
Field[4001] = Distance;
Field[4001].CurvesList = {
    37, 86 // injector/isolator edge
};
Field[4001].Sampling = 1000;
////
//Create threshold field that varrries element size near boundaries
cornerfac = boundratioinjector;
//blratiocorner = boundratioinjector;
Field[4002] = Threshold;
Field[4002].InField = 4001;
Field[4002].SizeMin = injectorsize/cornerfac;
Field[4002].SizeMax = injectorsize/cornerfac*(2.-1./cornerfac);

Field[4002].DistMin = 0.02;
Field[4002].DistMax = 2.0;
Field[4002].StopAtDistMax = 1;

// a smaller region right at the corner
cornerfac = boundratioinjector;
Field[4003] = Threshold;
Field[4003].InField = 4001;
Field[4003].SizeMin = injectorsize/cornerfac;
Field[4003].SizeMax = injectorsize/cornerfac*(2.-1./cornerfac);

Field[4003].DistMin = 0.02;
Field[4003].DistMax = 1.0;
Field[4003].StopAtDistMax = 1;

// Create distance field from curves, inside wall only
Field[15] = Distance;
Field[15].SurfacesList = {
    1, 3, 7, 8, 9, 10, 6, 5, 2
};
Field[15].Sampling = 1000;

//Create threshold field that varrries element size near boundaries
Field[16] = Threshold;
Field[16].InField = 15;
Field[16].SizeMin = samplesize / boundratiosurround;
//Field[16].SizeMax = samplesize;
//Field[16].SizeMax = bigsize;
Field[16].SizeMax = samplesize/boundratiosurround*(2.-1./boundratiosurround);
Field[16].DistMin = 0.02;
Field[16].DistMax = 5;
Field[16].StopAtDistMax = 1;

// Create distance field from curves, sample/fluid interface
Field[17] = Distance;
Field[17].SurfacesList = {
    4, 13
};

Field[17].Sampling = 1000;

//Create threshold field that varies element size near boundaries
Field[18] = Threshold;
Field[18].InField = 17;
Field[18].SizeMin = samplesize / boundratiosample;
//Field[18].SizeMax = cavitysize;
//Field[18].SizeMax = bigsize;
Field[18].SizeMax = samplesize/boundratiosample*(2.-1./boundratiosample);
Field[18].DistMin = 0.02;
Field[18].DistMax = 5;
Field[18].StopAtDistMax = 1;

//  background mesh size in the isolator (downstream of the nozzle)
Field[3] = Box;
Field[3].XMin = nozzle_end;
Field[3].XMax = 1000.0;
Field[3].YMin = -1000.0;
Field[3].YMax = 1000.0;
Field[3].ZMin = -1000.0;
Field[3].ZMax = 1000.0;
Field[3].VIn = isosize;
Field[3].VOut = bigsize;
//
//// background mesh size upstream of the inlet
Field[4] = Box;
Field[4].XMin = 0.;
Field[4].XMax = nozzle_start;
Field[4].YMin = -1000.0;
Field[4].YMax = 1000.0;
Field[4].ZMin = -1000.0;
Field[4].ZMax = 1000.0;
Field[4].VIn = inletsize;
Field[4].VOut = bigsize;
//
// background mesh size in the nozzle throat
Field[5] = Box;
Field[5].XMin = nozzle_start;
Field[5].XMax = nozzle_end;
Field[5].YMin = -1000.0;
Field[5].YMax = 1000.0;
Field[5].ZMin = -1000.0;
Field[5].ZMax = 1000.0;
Field[5].Thickness = 100;    // interpolate from VIn to Vout over a distance around the box
Field[5].VIn = nozzlesize;
Field[5].VOut = bigsize;
// background mesh size in the nozzle expansion to isolator
Field[105] = Box;
Field[105].XMin = nozzle_end;
Field[105].XMax = nozzle_exp_end;
Field[105].YMin = -1000.0;
Field[105].YMax = 1000.0;
Field[105].ZMin = -1000.0;
Field[105].ZMax = 1000.0;
Field[105].Thickness = 100;    // interpolate from VIn to Vout over a distance around the box
Field[105].VIn = 1.5*nozzlesize;
Field[105].VOut = bigsize;
//
// background mesh size in the cavity region
cavity_start = 600;
cavity_end = 640;
Field[6] = Box;
Field[6].XMin = cavity_start;
Field[6].XMax = cavity_end;
Field[6].YMin = -1000.0;
Field[6].YMax = 0.;
Field[6].ZMin = -1000.0;
Field[6].ZMax = 1000.0;
Field[6].Thickness = 100;    // interpolate from VIn to Vout over a distance around the box
Field[6].VIn = cavitysize;
Field[6].VOut = bigsize;
//
// background mesh size in the injection region
injector_start_x = 615;
injector_end_x = 660;
//injector_start_y = -0.0225*1000;
injector_start_y = -12;
injector_end_y = -15;
injector_start_z = -3;
injector_end_z = 3;
Field[7] = Cylinder;
Field[7].XAxis = injector_end_x - injector_start_x;
Field[7].XCenter = injector_start_x;
Field[7].YCenter = -13.5;
Field[7].ZCenter = 0.;
Field[7].Radius = 3;
Field[7].VIn = injectorsize;
Field[7].VOut = bigsize;

// background mesh size in the upstream injection region
Field[217] = Cylinder;
Field[217].YAxis = 17.5;
Field[217].XCenter =  533.2;
Field[217].YCenter = -22.5;
Field[217].ZCenter = 0.;
Field[217].Radius = 3;
Field[217].VIn = injectorsize;
Field[217].VOut = bigsize;
//
// background mesh size between upstream injection and cavity
Field[218] = Cylinder;
Field[218].XAxis = 65;
Field[218].XCenter =  590;
Field[218].YCenter = -9;
Field[218].ZCenter = 0.;
Field[218].Radius = 5;
Field[218].VIn = isosize/boundratio/1.5;
Field[218].VOut = bigsize;

// background mesh size between upstream injection and cavity
Field[219] = Cylinder;
Field[219].XAxis = 85;
Field[219].XCenter =  590;
Field[219].YCenter = -9;
Field[219].ZCenter = 0.;
Field[219].Radius = 9;
Field[219].VIn = isosize/boundratio;
Field[219].VOut = bigsize;

// background mesh size between upstream injection and cavity
Field[220] = Cylinder;
Field[220].XAxis = 5;
Field[220].XCenter =  535;
Field[220].YCenter = -9;
Field[220].ZCenter = 0.;
Field[220].Radius = 5;
Field[220].VIn = isosize/boundratio/2.;
//Field[218].VIn = injectorsize + 0.5*(isosize/boundratio - injectorsize);
Field[220].VOut = bigsize;

// background mesh size in the sample region
//Field[8] = Constant;
//Field[8].VolumesList = {5:5};
//Field[8].VIn = samplesize;
//Field[8].VOut = bigsize;

Field[8] = Box;
Field[8].XMin = -100000;
Field[8].XMax =  100000;
Field[8].YMin = -10000;
Field[8].YMax =  100000;
Field[8].ZMin = -100000;
Field[8].ZMax =  10000;
Field[8].Thickness = 100;
Field[8].VIn = samplesize/4;
Field[8].VOut = bigsize;

// background mesh size in the shear region
shear_start_x = 600;
shear_end_x = 680;
shear_start_y = -10;
shear_end_y = -3;
shear_start_z = -1000.0;
shear_end_z = 1000.0;
Field[9] = Box;
Field[9].XMin = shear_start_x;
Field[9].XMax = shear_end_x;
Field[9].YMin = shear_start_y;
Field[9].YMax = shear_end_y;
Field[9].ZMin = shear_start_z;
Field[9].ZMax = shear_end_z;
Field[9].Thickness = 100;
Field[9].VIn = shearsize;
Field[9].VOut = bigsize;

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

// keep the injector boundary spacing in the fluid mesh only
Field[20] = Restrict;
Field[20].VolumesList = {3};
Field[20].InField = 14;

Field[21] = Restrict;
Field[21].VolumesList = {3};
Field[21].InField = 7;

Field[221] = Restrict;
Field[221].VolumesList = {3};
Field[221].InField = 218;

// keep the cavity spacing in the fluid mesh only
Field[22] = Restrict;
Field[22].VolumesList = {3};
Field[22].InField = 6;

// keep the sample spacing in the wall mesh only
Field[23] = Restrict;
Field[23].VolumesList = {1:2};
Field[23].InField = 8;

// take the minimum of all defined meshing fields
Field[100] = Min;
Field[100].FieldsList = {
    3002, 4002, 4003,
    2010, 
    2011, 
    3, 
    4, 
    5, 
    9, 
    119,
    12, 
    16, 
    18, 
    20, 
    21, 
    22, 
    23, 
    102, 
    105, 
    221 ,
    217, 218, 219, 220
    };
//Field[100].FieldsList = {3002, 2010, 2011, 3, 4, 5, 9, 12, 16, 18, 20, 21, 22, 23, 102, 105, 118, 221, 218};
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
Mesh.RecombinationAlgorithm = 2;
Mesh.RecombineAll = 1;
// MeshAdapt
Mesh.Algorithm = 1;
// Frontal-Delaunay
//Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;
// 0 - none, 1 - all quads, 2 - all hexes
Mesh.SubdivisionAlgorithm = 0;

Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 100;
//Mesh.Smoothing = 0;
//Mesh.OptimizeNetgen = 0;

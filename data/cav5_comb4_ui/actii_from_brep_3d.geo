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
    nozzle_factor=12.0;
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

Physical Volume('fluid') = {1};
Physical Volume('wall_insert') = {2,4};
Physical Volume('wall_surround') = {3,5};

Physical Surface("inflow") = {2}; // inlet
Physical Surface("outflow") = {9}; // outlet
Physical Surface("injection") = {27}; // injection
Physical Surface("flow") = {2, 9, 27}; // injection
Physical Surface('isothermal_wall') = {
    4, // fore wall
    5, // aft wall
    1, // inflow top
    3, // inflow ramp top
    7, // nozzle top
    8, // isolator top
    6, // inflow bottom
    25, // inflow ramp bottom
    24, // inflow ramp bottom
    23, // isolator bottom
    22, // cavity front
    21, // cavity bottom
    18, // cavity back (ramp)
    17, // post-cavity flat
    15, // post-cavity flat, surround
    11, // combustor bottom before sample
    13, // combustor bottom after sample
    12, // combustor bottom around sample
    10, // combustor flat
    26 // injector
};

Physical Surface('wall_farfield') = {
    34, 35, 36, 37, // cavity surround
    49, 50, 48, 47, 51 // combustor surround
};

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[1] = Distance;
Field[1].SurfacesList = {
    4, // fore wall
    5, // aft wall
    1, // inflow top
    3, // inflow ramp top
    7, // nozzle top
    8, // isolator top
    6, // inflow bottom
    25, // inflow ramp bottom
    24, // nozzle bottom
    23 // isolator bottom
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
    28, 29, // top fore
    9, 12, // top aft
    25, 24, 19, 15, 16, 17, // bottom aft
    42, 41, 35, 36, 32, 33, 34 // bottom fore
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

// Create distance field from surfaces for wall meshing in the combustor
Field[101] = Distance;
Field[101].SurfacesList = {
    17, // post-cavity flat
    11, // combustor bottom before sample
    13, // combustor bottom after sample
    10, // combustor flat
    12, 14 // combustor sample and surround
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
    22, // cavity front
    21, // cavity bottom
    18 // cavity back (ramp)
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
26 // injector wall
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

// Create distance field from curves, inside wall only
Field[15] = Distance;
Field[15].SurfacesList = {
    28:33,
    39:46
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
    16, 20, 19, 15, // cavity sample/surround
    12, 14  // combustor sample/surround
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

// Create distance field from curves, sample/fluid interface
Field[117] = Distance;
Field[117].CurvesList = {
    68, // cavity sample corner
    69, 62  // cavity surround corner
};

Field[117].Sampling = 1000;

//Create threshold field that varies element size near boundaries
Field[118] = Threshold;
Field[118].InField = 117;
Field[118].SizeMin = samplesize / boundratiosample/2.;
//Field[118].SizeMax = cavitysize;
Field[118].SizeMax = samplesize/boundratiosample*(2.-1./boundratiosample);
Field[118].DistMin = 0.02;
Field[118].DistMax = 5;
Field[118].StopAtDistMax = 1;
//
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
injector_start_y = -10;
injector_end_y = -13;
injector_start_z = -3;
injector_end_z = 3;
Field[7] = Box;
Field[7].XMin = injector_start_x;
Field[7].XMax = injector_end_x;
Field[7].YMin = injector_start_y;
Field[7].YMax = injector_end_y;
Field[7].ZMin = injector_start_z;
Field[7].ZMax = injector_end_z;
Field[7].Thickness = 100;    // interpolate from VIn to Vout over a distance around the cylinder
////Field[7] = Cylinder;
////Field[7].XAxis = 1;
////Field[7].YCenter = -0.0225295;
////Field[7].ZCenter = 0.0157;
////Field[7].Radius = 0.003;
Field[7].VIn = injectorsize;
Field[7].VOut = bigsize;

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

// keep the injector boundary spacing in the fluid mesh only
Field[20] = Restrict;
Field[20].VolumesList = {1};
Field[20].InField = 14;

Field[21] = Restrict;
Field[21].VolumesList = {1};
Field[21].InField = 7;

// keep the cavity spacing in the fluid mesh only
Field[22] = Restrict;
Field[22].VolumesList = {1};
Field[22].InField = 6;

// keep the sampel spacing in the wall mesh only
Field[23] = Restrict;
Field[23].VolumesList = {2:5};
Field[23].InField = 8;

// take the minimum of all defined meshing fields
Field[100] = Min;
//Field[100].FieldsList = {2, 3, 4, 5, 6, 7, 12, 14};
//Field[100].FieldsList = {2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 20, 21};
//Field[100].FieldsList = {2, 3, 4, 5, 6, 8, 9, 12, 16, 18, 20, 21, 22, 102, 105, 118};
//Field[100].FieldsList = {2, 3, 4, 5, 9, 12, 16, 18, 20, 21, 22, 23, 102, 105, 118};
//
//Field[100].FieldsList = {2, 3, 4, 5, 9, 12, 16, 18, 20, 21, 22, 23, 102, 105, 118};
Field[100].FieldsList = {3002, 2010, 2011, 3, 4, 5, 9, 12, 16, 18, 20, 21, 22, 23, 102, 105, 118};
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
//Mesh.OptimizeNetgen = 1;
//Mesh.Smoothing = 100;
Mesh.Smoothing = 0;
Mesh.OptimizeNetgen = 0;

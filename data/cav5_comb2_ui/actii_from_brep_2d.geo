SetFactory("OpenCASCADE");
surface_vector[] = ShapeFromFile("actii-2d.brep");

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
    boundratio=4.0;
EndIf

If(Exists(blratiocavity))
    boundratiocavity=blratiocavity;
Else
    boundratiocavity=2.0;
EndIf

If(Exists(blratioinjector))
    boundratioinjector=blratioinjector;
Else
    boundratioinjector=2.0;
EndIf

If(Exists(blratiosample))
    boundratiosample=blratiosample;
Else
    boundratiosample=8.0;
EndIf

If(Exists(blratiosurround))
    boundratiosurround=blratiosurround;
Else
    boundratiosurround=2.0;
EndIf

If(Exists(injectorfac))
    injector_factor=injectorfac;
Else
    injector_factor=5.0;
EndIf

If(Exists(samplefac))
    sample_factor=samplefac;
Else
    sample_factor=4.0;
EndIf

If(Exists(shearfac))
    shear_factor=shearfac;
Else
    shear_factor=6.0;
EndIf

If(Exists(isofac))
    iso_factor=isofac;
Else
    iso_factor=2.0;
EndIf

If(Exists(cavityfac))
    cavity_factor=cavityfac;
Else
    cavity_factor=6;
EndIf

If(Exists(nozzlefac))
    nozzle_factor=nozzlefac;
Else
    nozzle_factor=6;
EndIf

// horizontal injection
cavityAngle=45;
inj_h=4.;  // height of injector (bottom) from floor
inj_d=1.59; // diameter of injector
inj_l = 20; // length of injector

bigsize = basesize*4;     // the biggest mesh size 
inletsize = basesize*2;   // background mesh size upstream of the nozzle
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
Printf("boundratioinjector = %f", boundratioinjector);
Printf("boundratiosample = %f", boundratiosample);
Printf("boundratiosurround = %f", boundratiosurround);

Geometry.Tolerance = 1.e-3;
Coherence;

//Curve Loop(1) = {1:28};
//Curve Loop(2) = {30, 31, 16, 15};
//Curve Loop(3) = {34, 35, 17, 31, 30, 14};
//Curve Loop(4) = {40, 41, 42, 21};
//Curve Loop(5) = {44, 45, 46, 22, 42, 41, 40, 20};

Curve Loop(1) = {22:13, 39:26, 9, 1, 5};
Curve Loop(2) = {4, 3, 2, 1};
Curve Loop(3) = {9, 8, 7, 6, 5, 2, 3, 4};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

Physical Surface('fluid') = {-1};
Physical Surface('wall_insert') = {-2};
Physical Surface('wall_surround') = {-3};

Physical Curve("inflow") = {33}; // inlet
Physical Curve("outflow") = {28}; // outlet
Physical Curve("injection") = {19, 39}; // injection
Physical Curve("flow") = {33, 28, 19, 39}; // all inflow/outflow
Physical Curve("isothermal_wall") = {
26, 27, 29:32, 34:38, 22:20, 13:18
};
Physical Curve("wall_farfield") = {
6:8 //cavity wall surround exterior
};

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[1] = Distance;
Field[1].CurvesList = {
    34:37, // upstream bottom
    14, // isolator bottom
    29:32 // top
};
Field[1].Sampling = 1000;
////
//Create threshold field that varies element size near boundaries
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = isosize / boundratio;
Field[2].SizeMax = isosize;
Field[2].DistMin = 0.02;
Field[2].DistMax = 20;
Field[2].StopAtDistMax = 1;

//Create threshold field that varrries element size near boundaries
//this is for the nozzle only
Field[2002] = Threshold;
Field[2002].InField = 1;
Field[2002].SizeMin = nozzlesize / boundratio;
Field[2002].SizeMax = isosize;
Field[2002].DistMin = 0.02;
Field[2002].DistMax = 15;
Field[2002].StopAtDistMax = 1;

//Create threshold field that varrries element size near boundaries
//this is for the nozzle expansion only
Field[2003] = Threshold;
Field[2003].InField = 1;
Field[2003].SizeMin = 1.5*nozzlesize / boundratio;
Field[2003].SizeMax = isosize;
Field[2003].DistMin = 0.02;
Field[2003].DistMax = 20;
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

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[101] = Distance;
Field[101].CurvesList = {
    22, // pre sample flat
    26, // combustor before sample
    27  // combustor flat
};
Field[101].Sampling = 1000;
////
//Create threshold field that varrries element size near boundaries
Field[102] = Threshold;
Field[102].InField = 101;
Field[102].SizeMin = isosize/boundratio/1.5;
Field[102].SizeMax = isosize/3;
Field[102].DistMin = 0.02;
Field[102].DistMax = 8;
Field[102].StopAtDistMax = 1;
//
// Create distance field from curves, cavity only
Field[11] = Distance;
Field[11].CurvesList = {
    15, // cavity front
    16, // cavity bottom
    17, // cavity bottom slant
    21, // cavity top slant
    22 // post cavity flat
};
Field[11].Sampling = 1000;

//Create threshold field that varies element size near boundaries
Field[12] = Threshold;
Field[12].InField = 11;
Field[12].SizeMin = cavitysize / boundratiocavity;
Field[12].SizeMax = cavitysize;
Field[12].DistMin = 0.02;
Field[12].DistMax = 5;
Field[12].StopAtDistMax = 1;

// Create distance field from curves, injector only
Field[13] = Distance;
Field[13].CurvesList = {
    18, 20, 38, 13 // injector sides
};
Field[13].Sampling = 1000;
//
//Create threshold field that varrries element size near boundaries
Field[14] = Threshold;
Field[14].InField = 13;
Field[14].SizeMin = injectorsize / boundratioinjector;
Field[14].SizeMax = injectorsize;
Field[14].DistMin = 0.001;
Field[14].DistMax = 1.0;
Field[14].StopAtDistMax = 1;

// Create distance field from curves, inside wall only
Field[15] = Distance;
Field[15].CurvesList = {
    2, 3, 4
};
Field[15].Sampling = 1000;

//Create threshold field that varrries element size near boundaries
Field[16] = Threshold;
Field[16].InField = 15;
Field[16].SizeMin = samplesize / boundratiosurround;
Field[16].SizeMax = samplesize;
Field[16].DistMin = 0.2;
Field[16].DistMax = 5;
Field[16].StopAtDistMax = 1;

// Create distance field from curves, sample/fluid interface
Field[17] = Distance;
Field[17].CurvesList = {
    5, 1, // cavity surround
    9 // cavity sample
};
Field[17].Sampling = 1000;

//Create threshold field that varies element size near boundaries
Field[18] = Threshold;
Field[18].InField = 17;
Field[18].SizeMin = samplesize / boundratiosample;
Field[18].SizeMax = samplesize;
Field[18].DistMin = 0.2;
Field[18].DistMax = 5;
Field[18].StopAtDistMax = 1;
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
//Field[105].VIn = nozzlesize+(isosize-nozzlesize)/2;
Field[105].VIn = 2*nozzlesize;
//Field[105].VIn = nozzlesize;
Field[105].VOut = bigsize;
//
// background mesh size in the cavity region
cavity_start = 600;
cavity_end = 640;
Field[6] = Box;
Field[6].XMin = cavity_start;
Field[6].XMax = cavity_end;
Field[6].YMin = -1000.0;
Field[6].YMax = -8;
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
Field[7].VIn = injectorsize;
Field[7].VOut = bigsize;

// background mesh size in the upstream injection region
injector_start_x = 530;
injector_end_x = 610;
//injector_start_y = -0.0225*1000;
injector_start_y = -5;
injector_end_y = -8;
injector_start_z = -3;
injector_end_z = 3;
Field[117] = Box;
Field[117].XMin = injector_start_x;
Field[117].XMax = injector_end_x;
Field[117].YMin = injector_start_y;
Field[117].YMax = injector_end_y;
Field[117].ZMin = injector_start_z;
Field[117].ZMax = injector_end_z;
Field[117].Thickness = 100;    // interpolate from VIn to Vout over a distance around the cylinder
Field[117].VIn = injectorsize;
Field[117].VOut = bigsize;

// background mesh size in the sample region
Field[8] = Constant;
Field[8].SurfacesList = {1,2};
Field[8].VIn = samplesize;
Field[8].VOut = bigsize;

// background mesh size in the cavity shear region
shear_start_x = 525;
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
Field[20].SurfacesList = {1};
Field[20].InField = 14;

Field[21] = Restrict;
Field[21].SurfacesList = {1};
Field[21].InField = 7;

// take the minimum of all defined meshing fields
Field[100] = Min;
//Field[100].FieldsList = {2, 3, 4, 5, 6, 7, 12, 14};
//Field[100].FieldsList = {2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 20, 21};
//Field[100].FieldsList = {2, 3, 4, 5, 6, 8, 9, 12, 16, 18, 20, 21, 102, 105};
//
//Field[100].FieldsList = {2, 3, 4, 5, 6, 8, 9, 12, 16, 18, 20, 21, 102, 105};
//Field[100].FieldsList = {2002, 3, 4, 5, 6, 8, 9, 12, 16, 18, 20, 21, 102, 105};
Field[100].FieldsList = {2010, 2011, 3, 4, 5, 6, 8, 9, 12, 16, 18, 20, 21, 102, 105, 117};


//Field[100].FieldsList = {2, 3, 4, 5, 6, 8, 9, 12, 16, 18, 20, 21, 102};
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;


Mesh.Algorithm = 8;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 100;

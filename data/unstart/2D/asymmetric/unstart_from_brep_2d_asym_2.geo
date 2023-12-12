SetFactory("OpenCASCADE");
surface_vector[] = ShapeFromFile("asym_exhaust_2d.brep");


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

If(Exists(exhaustfac))
    exhaust_factor=exhaustfac;
Else
    exhaust_factor=0.5;
EndIf

// horizontal injection
//cavityAngle=45;
//inj_h=4.;  // height of injector (bottom) from floor
//inj_d=1.59; // diameter of injector
//inj_l = 20; // length of injector

bigsize = basesize*4;     // the biggest mesh size 
inletsize = basesize/2;   // background mesh size upstream of the nozzle
isosize = basesize/iso_factor;       // background mesh size in the isolator
nozzlesize = basesize/nozzle_factor;       // background mesh size in the nozzle
cavitysize = basesize/cavity_factor; // background mesh size in the cavity region
shearsize = isosize/shear_factor; // background mesh size in the shear region
exhaustsize = isosize/exhaust_factor; // background mesh size in the shear region
// samplesize = basesize/sample_factor;       // background mesh size in the sample
// injectorsize = inj_d/injector_factor; // background mesh size in the injector region

Printf("basesize = %f", basesize);
Printf("inletsize = %f", inletsize);
Printf("isosize = %f", isosize);
Printf("nozzlesize = %f", nozzlesize);
Printf("cavitysize = %f", cavitysize);
Printf("shearsize = %f", shearsize);
Printf("boundratio = %f", boundratio);
Printf("boundratiocavity = %f", boundratiocavity);
Printf("boundratiosurround = %f", boundratiosurround);

Geometry.Tolerance = 1.e-3;
Coherence;

Curve Loop(2) = {1:24};

Plane Surface(2) = {2};

Physical Surface('fluid') = {-1};

Physical Curve("inflow") = {20}; // inlet
Physical Curve("outflow") = {11}; // outlet
Physical Curve("flow") = {20, 11}; // all inflow/outflow
Physical Curve('isothermal_wall') = {
    1, // isolator bottom
    2, // cavity front
    3, // cavity bottom
    4, // cavity slant
    5, // combustor flat
    6, // combustor slant
    7, // fillet outlet sponge
    8, // flat outlet sponge
    9, // lower vertical exhaust wall
    10, // lower exhaust wall
    12, // upper exhaust wall
    13, // upper vertical exhaust wall
    14, // combustor/isolator top
    15, // diverging nozzle top
    16, // converging nozzle curve top
    17, // converging nozzle slant top
    18, // fillet inlet sponge top
    19, // inlet sponge top
    21, // inlet sponge bottom
    22, // fillet inlet sponge bottom
    23, // converging nozzle slant bottom
    24 // converging nozzle curve bottom
};

// Create distance field from surfaces for wall meshing, excludes cavity
Field[1] = Distance;
Field[1].CurvesList = {
    1, // isolator bottom
    5, // combustor flat
    6, // combustor slant
    7, // fillet outlet sponge
    8, // flat outlet sponge
    9, // lower vertical exhaust wall
    10, // lower exhaust wall
    12, // upper exhaust wall
    13, // upper vertical exhaust wall
    14, // combustor/isolator top
    15, // diverging nozzle top
    16, // converging nozzle curve top
    17, // converging nozzle slant top
    18, // fillet inlet sponge top
    19, // inlet sponge top
    21, // inlet sponge bottom
    22, // fillet inlet sponge bottom
    23, // converging nozzle slant bottom
    24 // converging nozzle curve bottom
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

//Create threshold field that varies element size near boundaries
//this is for the nozzle only
Field[2002] = Threshold;
Field[2002].InField = 1;
Field[2002].SizeMin = nozzlesize / boundratio;
Field[2002].SizeMax = isosize;
Field[2002].DistMin = 0.02;
Field[2002].DistMax = 15;
Field[2002].StopAtDistMax = 1;

//Create threshold field that varies element size near boundaries
//this is for the nozzle expansion only
Field[2003] = Threshold;
Field[2003].InField = 1;
Field[2003].SizeMin = 1.5*nozzlesize / boundratio;
Field[2003].SizeMax = isosize;
Field[2003].DistMin = 0.02;
Field[2003].DistMax = 20;
Field[2003].StopAtDistMax = 1;

sigma = 25;
nozzle_start = 65;
nozzle_end = 120;
nozzle_exp_end = 190;

// restrict the nozzle bl meshing to the nozzle only
Field[2010] = MathEval;
Field[2010].F = Sprintf("F2 + (F2002 - F2)*(0.5*(1.0 - tanh(%g*(x - %g))))*(0.5*(1.0 - tanh(%g*(%g - x))))", sigma, nozzle_end, sigma, nozzle_start);

// restrict the nozzle expansion bl meshing to the nozzle expansion only
Field[2011] = MathEval;
Field[2011].F = Sprintf("F2 + (F2003 - F2)*(0.5*(1.0 - tanh(%g*(x - %g))))*(0.5*(1.0 - tanh(%g*(%g - x))))", sigma, nozzle_exp_end, sigma, nozzle_end);

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[101] = Distance;
Field[101].CurvesList = {
    5, // combustor flat
    6 // combustor slant
};
Field[101].Sampling = 1000;
////
//Create threshold field that varies element size near boundaries
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
    2, // cavity front
    3, // cavity bottom
    4 // cavity slant
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
cavity_start = 450;
cavity_end = 500;
Field[6] = Box;
Field[6].XMin = cavity_start;
Field[6].XMax = cavity_end;
Field[6].YMin = -1000.0;
Field[6].YMax = 2.0;
Field[6].ZMin = -1000.0;
Field[6].ZMax = 1000.0;
Field[6].Thickness = 100;    // interpolate from VIn to Vout over a distance around the box
Field[6].VIn = cavitysize;
Field[6].VOut = bigsize;
//

// background mesh size in the cavity shear region
shear_start_x = 450;
shear_end_x = 550;
shear_start_y = -2;
shear_end_y = 5;
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

//  background mesh size in the exhaust volume
Field[33] = Box;
Field[33].XMin = 812;
Field[33].XMax = 2000.0;
Field[33].YMin = -1000.0;
Field[33].YMax = 1000.0;
Field[33].ZMin = -1000.0;
Field[33].ZMax = 1000.0;
Field[33].VIn = exhaustsize;
Field[33].VOut = bigsize;
//

// take the minimum of all defined meshing fields
Field[100] = Min;
Field[100].FieldsList = {2010, 2011, 3, 4, 5, 6, 9, 12, 102, 105, 33};

Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;


Mesh.Algorithm = 4;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 100;

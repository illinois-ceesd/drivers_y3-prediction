Merge "geometry_trans.geo_unrolled";
//
Mesh.ScalingFactor = 0.001;
//
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
    sample_factor=8.0;
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
inletsize = basesize/2;   // background mesh size upstream of the nozzle
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
//
Physical Surface('fluid') = {
    -100, -101, -3, -4, -5, 6, -7, -8, 9, 10,
    11, 12, 13, -14, -15, -16, -17, -18, -19, -20,
    -21, -22, -23, 24, -25, 26, 27};
Physical Surface('wall_insert') = {200};
Physical Surface('wall_surround') = {-202};

Physical Curve("inflow") = {1074:1076}; // inlet
Physical Curve("outflow") = {19}; // outlet
Physical Curve("injection") = {1080:1082}; // injection
Physical Curve("upstream_injection") = {1077:1079}; // injection
Physical Curve("flow") = {1074:1076, 1080:1082, 1077:1079}; // all inflow
Physical Curve("isothermal_wall") = {26, 1010, 1003, 1002, 23, 22, 21, 1105, 1104, 1107, 1106, 17, 16, 15, 14, 10, 9, 8, 6, 5, 1051, 1050, 1049, 1048, 1101, 1100, 1, 35, 33, 32, 1087, 1096, 1095, 1021, 1026, 1009, 1027, 29, 28};
Physical Curve("wall_farfield") = {37:39};
//
// upstream injector bl
//
num_bl_injector = 5;
num_injector_center = 2;
num_injector = 40;
num_injector_curve = 8;
Transfinite Curve{1077, -1079} = num_bl_injector Using Progression 1.5; // vertical segments
Transfinite Curve{302, 300} = num_bl_injector Using Progression 1.2; // vertical segments
Transfinite Curve{1078, 304} = num_injector_center; // center vertical segments
Transfinite Curve{-33, -1040, -1042, 35} = num_injector Using Progression 1.03; // axial edges
Transfinite Curve{1, 32, 1043, 1041} = num_injector_curve ; // curved bits
Transfinite Surface{25} = {35, 1, 1073, 1072}; // right
Transfinite Surface{26} = {32, 1070, 1067, 34}; // left
Transfinite Surface{27}; // center
//
// cavity injector bl
//
num_injector_ext = 10;
num_injector = 40;
num_injector_diff = 10;
Transfinite Curve{1080, -1082} = num_bl_injector Using Progression 1.3; // vertical segments
Transfinite Curve{1071, -1073} = num_bl_injector Using Progression 1.0; // vertical segments
Transfinite Curve{-1065, 1063} = num_bl_injector Using Progression 1.0; // vertical segments
Transfinite Curve{1081, 1072, 1064} = num_injector_center; // center vertical segments
Transfinite Curve{-8, 204} = num_injector Using Progression 1.03; // axial edges in injector top bl
Transfinite Curve{-1069, -6} = num_injector Using Progression 1.03; // axial edges in injector bottom bl
Transfinite Curve{1061, 1059, 1057, 1055} = num_injector_ext; // axial edges into cavity
Transfinite Surface{23}; // top
Transfinite Surface{24}; // bottom
Transfinite Surface{22};

//
// cavity bl
//
num_bl_cavity = 8;
num_cavity_front = 20;
num_cavity_bottom = 70;
num_cavity_wall_bottom = 10;
num_cavity_wall_top = 20;
Transfinite Curve {310, 1046, 1050, -1049, -1045, 313, -1061, -1108, -1067, -1055, -314} = num_bl_cavity Using Progression 1.1; // vertical segments
Transfinite Curve {1048, 1044} = num_cavity_front;
Transfinite Curve {1051, 1047} = num_cavity_bottom;
Transfinite Curve {5, 1062} = num_cavity_wall_bottom;
Transfinite Curve {9, 1066} = num_cavity_wall_top;
Transfinite Surface{14:21};
//
// top bl, isolator, combustor
//
num_bl_isolator = 8;
num_upstream = 20;
num_upstream_ramp = 30;
num_nozzle_exp = 75;
num_top = 370;
num_exhaust = 12;
num_bl_comb_exp = 88;
num_exhaust_edge = 16;
Transfinite Curve {1074, 1014, 1007, -403, -1105} = num_bl_isolator Using Progression 1.1;
Transfinite Curve {26, 48} = num_upstream;
Transfinite Curve {1010, 1012} = num_upstream_ramp Using Progression 1.03;
Transfinite Curve {-1002, -1004} = num_nozzle_exp Using Progression 1.01;
Transfinite Curve {23} = num_top;
Transfinite Curve {51} = num_top-num_bl_comb_exp+1;
Transfinite Curve {22, 52} = num_exhaust Using Progression 1.05;
Transfinite Curve {-21, 500} = num_exhaust_edge Using Progression 1.1;
Transfinite Curve { 53, 600} = num_bl_comb_exp;
Transfinite Surface {4} = {27, 1010, 1012, 56};
Transfinite Surface {6};
Transfinite Surface {7} = {1005, 24, 21, 1110};
//
// bottom bl, isolator, combustor
//
num_bl_inlet_bottom = 10;
num_bl_isolator_bottom = 14;
num_upstream_ramp1 = 20;
num_upstream_ramp2 = 16;
num_isolator = 100;
num_injector_upstream = 20;
num_injector_downstream = 100;
num_cavity_upstream = 20;
Transfinite Curve {-1076, -1031} = num_bl_inlet_bottom Using Progression 1.1;
Transfinite Curve {-1024, 1091, 1084, 1098, 311} = num_bl_isolator_bottom Using Progression 1.1;
Transfinite Curve {301, 303} = num_bl_isolator_bottom-num_bl_injector+1 Using Progression 1.1;
Transfinite Curve {28, 1016} = num_upstream;
Transfinite Curve {29, 1017} = num_upstream_ramp1;
Transfinite Curve {1027, 1029} = num_upstream_ramp2 Using Progression 1.05;
Transfinite Curve {1095, 1093} = num_nozzle_exp Using Progression 1.01;
Transfinite Curve {1096, 1094} = num_isolator;
Transfinite Curve {1087, 1089} = num_injector_upstream;
Transfinite Curve {1100, 1102} = num_injector_downstream;
Transfinite Curve {1101, 1103} = num_cavity_upstream;
Transfinite Surface {5} = {28, 1024, 1037, 1035};
Transfinite Surface {8, 9, 12};
Transfinite Surface {10} = {1100, 1101, 1032, 32};
Transfinite Surface {11} = {1, 1039, 1109, 1108};

//
// bottom bl combustor
//
num_bl_combustor = 14;
num_before_sample = 30;
num_sample_surround = 10;
num_sample = 75;
num_combustor = 300;
num_combustor_end = 30;
num_combustor_end_upper = 78;
Transfinite Curve {312, -1107} = num_bl_combustor Using Progression 1.1;
Transfinite Curve {1033, 10} = num_before_sample;
Transfinite Curve {1034, 11, 1036, 13} = num_sample_surround;
Transfinite Curve {1035, 12} = num_sample;
Transfinite Curve {14} = num_combustor Using Progression 1.005;
//Transfinite Curve {14} = num_combustor Using Progression 1.;
//Transfinite Curve {1038} = num_combustor+num_combustor_end-num_combustor_end_upper Using Progression 1.004;
Transfinite Curve {1038} = num_combustor-num_combustor_end_upper+1 Using Progression 1.005;
Transfinite Curve {600} = num_combustor_end_upper Using Progression 1.005;
Transfinite Curve {15, 601} = num_combustor_end;
Transfinite Curve {-16, -1039} = num_exhaust Using Progression 1.1;
Transfinite Curve {17, 501} = num_exhaust_edge Using Progression 1.1;
Transfinite Surface {13} = {18, 1111, 1053, 10};
//
// nozzle
//
num_nozzle_top1 = 3;
num_nozzle_top2 = 47;
num_nozzle_bottom1 = 10;
num_nozzle_bottom2 = 40;
num_nozzle_interior = 10;
Transfinite Curve {1009} = num_nozzle_top1;
Transfinite Curve {1003} = num_nozzle_top2;
Transfinite Curve {1026} = num_nozzle_bottom1;
Transfinite Curve {1021} = num_nozzle_bottom2;
Transfinite Curve {1030} = num_nozzle_interior;
Transfinite Curve {1023} = num_bl_inlet_bottom + num_nozzle_interior - num_bl_isolator_bottom; 
Transfinite Surface {3} = {1010, 1004, 1033, 1035};
//
//
nozzle_start = 270;
nozzle_end = 325;
nozzle_exp_end = 375;
//  background mesh size in the isolator (downstream of the nozzle)
Field[3] = Box;
Field[3].XMin = nozzle_end;
Field[3].XMax = 940.0;
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
injector_start_y = -12;
injector_end_y = -15;
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
Field[8].SurfacesList = {200, 202};
Field[8].VIn = samplesize;
Field[8].VOut = bigsize;

Field[1] = Constant;
Field[1].SurfacesList = {100};
Field[1].VIn = inletsize;
Field[1].VOut = bigsize;

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

// take the minimum of all defined meshing fields
Field[100] = Min;
Field[100].FieldsList = {1, 3, 4, 5, 6, 8, 9, 117};

//Field[100].FieldsList = {2, 3, 4, 5, 6, 8, 9, 12, 16, 18, 20, 21, 102};
Background Field = 100;

//Recombine Surface{3:27};
//Recombine Surface{100, 101};
//Recombine Surface{200, 202};

Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 1;
Mesh.RecombineOptimizeTopology =5;
//Mesh.RecombineAll = 0;
Mesh.RecombineAll = 1;

Mesh 2;
RecombineMesh;
Mesh.SubdivisionAlgorithm = 1;
RefineMesh;
//Mesh.SubdivisionAlgorithm = 0;
Mesh.MshFileVersion = 2.2;
Save "actii_2d.msh";

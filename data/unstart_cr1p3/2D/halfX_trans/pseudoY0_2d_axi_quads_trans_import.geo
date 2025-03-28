Merge "geometry_trans.geo_unrolled";
//
Mesh.ScalingFactor = 0.001;
//
// mesh sizing constraints
If(Exists(axialsize))
    axial_size=axialsize;
Else
    axial_size=1.;
EndIf

If(Exists(radialsize))
    radial_size=radialsize;
Else
    radial_size=2.;
EndIf

If(Exists(blsize))
    blsize=blsize;
Else
    blsize=2.0;
EndIf

// transfinite bl meshing
num_bl_nozzle = 10*blsize;

//
// nozzle bl
//
// the vertical segments
//Transfinite Curve{1247, 1245, 1249, 1252, 1255, 1258, 1261} = num_bl_nozzle Using Progression 1.1;
Transfinite Curve{1247, 1245, 1249, 1252, 1258, 1261} = num_bl_nozzle Using Progression 1.1;
Transfinite Curve{1255} = num_bl_nozzle Using Progression 1.2;
// upstream
Transfinite Curve{1246, 1244} = 14*axial_size Using Progression 1.0;
// curve upstream
Transfinite Curve{1250, 1248} = 8*axial_size Using Progression 1.0;
// curve slant upstream
Transfinite Curve{1253, 1251} = 10*axial_size Using Progression 1.0;
// curve into the throat
Transfinite Curve{1256, 1254} = 22*axial_size Using Progression 1.04;
// out of the throat
Transfinite Curve{1266, 1259, 1257} = 40*axial_size Using Progression 1.04;
// nozzle expansion
Transfinite Curve{1268, 1262, 1260} = 120*axial_size Using Progression 1.;
Transfinite Surface{300:303};
Transfinite Surface{304} = {284, 283, 286, 285};
Transfinite Surface{305} = {286, 285, 288, 287};

//Recombine Surface{300:305};

// upstream nozzle
//Transfinite Curve{1263, 1265, 1267, 1269, 1237, 1239, 1241, 1243} = 10 Using Progression 1.1;
Transfinite Curve{1263, 1267, 1269, 1237, 1239, 1241, 1243} = 10*radial_size Using Progression 1.1;
Transfinite Curve{1265} = 10*radial_size Using Progression 1.;
Transfinite Curve{1264} = (14+8+10+22)*axial_size-3 Using Progression .98; // upstream of throat
Transfinite Surface{320} = {289, 278, 290, 284};
Transfinite Surface{321} = {290, 284, 291, 286};
Transfinite Surface{322} = {291, 286, 292, 288};
//Recombine Surface{320, 321, 322};

//
// nozzle bl
//
// transfinite bl meshing
num_bl_model_inner = 8*blsize;
num_bl_model_outer = 8*blsize;

// the inner vertical segments
Transfinite Curve{1224, 1222, 1226, 1233, 1235} = num_bl_model_inner Using Progression 1.1;
// the outer vertical segments
Transfinite Curve{-1221, -1219, -1230} = num_bl_model_outer Using Progression 1.1;
Transfinite Curve{-1274} = num_bl_model_outer Using Progression 1.1;
// tip
Transfinite Curve{1223, 1220, 1218} = 10*axial_size Using Progression 1.0;
// inlet inside
Transfinite Curve{1227, 1225} = 15*axial_size Using Progression .9;
// inlet outside
Transfinite Curve{1229, 1228} = 20*axial_size Using Progression 1.1;
// model interior
Transfinite Curve{1240, 1232, 1231} = 250*axial_size Using Progression 1.0;
// outlet
Transfinite Curve{1242, 1236, 1234} = 10*axial_size Using Progression 1.0;
Transfinite Surface{200:205};

//Recombine Surface{200:205};

// model interior core vertical
num_model = 10*radial_size;
Transfinite Curve{1237, 1239, 1241, 1243} = num_model ;
// model inlet core
Transfinite Curve{1238} = (10+15)*axial_size-1 Using Progression 1.05;
Transfinite Surface{220} = {271, 272, 264, 262};
Transfinite Surface{221, 222};
//Recombine Surface{220:222};

//
// nozzle/model transition
//
num_transition = 20*axial_size;
Transfinite Curve{1270, 1271, 1273, 1280, 1282, 1283} = num_transition Using Progression 1;
// end of the nozzle bl and wall, these don't have to be the same, just add up
// to the same as the model tip mesh
Transfinite Curve{1261} = num_bl_nozzle Using Progression 1.0;
Transfinite Curve{1272} = num_bl_model_inner + num_bl_model_outer - num_bl_nozzle Using Progression 1.0;
Transfinite Surface{1000};
Transfinite Surface{1001} = {262, 257, 293, 288};
//Recombine Surface{1000:1001};

//
// the exhaust plume
//
Transfinite Curve{1275, 1277, 1215} = 40 Using Progression 1.0;
Transfinite Curve{1276} = num_model+num_bl_model_inner+num_bl_model_outer-2 Using Progression 1.0;
Transfinite Surface{1100} = {274, 294, 295, 296};
Transfinite Surface{1202};
//Recombine Surface{1100, 1202};

//
// the plume spill
//
Transfinite Curve{1281, 1284} = (20+10)*axial_size-1 Using Progression 1.0;
Transfinite Surface{1200} = {293, 257, 265, 253};
//Recombine Surface{1200};
// external bl
Transfinite Curve{1278, 1279, 1214} = 100 Using Progression 1.01;
Transfinite Surface{1101, 1201};
//Recombine Surface{1101, 1201};

// 
// external region
//
Transfinite Curve{1285, 1213} = 20 Using Progression 0.9;
Transfinite Surface{1210};
Transfinite Curve{-1212} = 20 Using Progression 1.2;
Transfinite Curve{-1216} = 20 Using Progression 1;
Transfinite Curve{1217} = 40+100+20-2 Using Progression 1.0;
Transfinite Surface{1} = {255, 256, 251, 252};
//Recombine Surface{1210, 1};

Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 1;
//Recombine Surface{1};
Mesh.RecombineAll = 1;
Physical Surface("fluid") = {
    1, -200, -201, -202, 203, 204, -205,
    220, 221, 222, 
    -300, -301, -302, -303, -304, -305, 
    320:322, 1000, 1001, 1100, 1101, 1200:1202, 1210};

Physical Surface("fluid") = {1, 200:205, 220:222, 300:305, 320:322, 1000, 1001, 1100, 1101, 1200:1202, 1210};
Physical Curve("inflow") = {1263, 1247};
Physical Curve("outflow") = {1276, 1283, 1216};
Physical Curve("isothermal_wall") = {1217, 1212, 1284, 1285, 1272, 1260, 1257, 1254, 1251, 1248, 1244, 1225, 1231, 1229, 1278, 1234};
Physical Curve("symmetry") = {1264, 1266, 1268, 1270, 1240, 1242, 1275, 1238};

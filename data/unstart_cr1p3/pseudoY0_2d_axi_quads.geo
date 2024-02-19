SetFactory("OpenCASCADE");
//Merge "pseudoY0_2d.brep";
surface_vector[] = ShapeFromFile("pseudoY0_2d.brep");

Mesh.ScalingFactor = 0.001;

// mesh sizing constraints
If(Exists(size))
    basesize=size;
Else
    basesize=10;
EndIf

If(Exists(blrationozzle))
    boundrationozzle=blrationozzle;
Else
    boundrationozzle=2.0;
EndIf

If(Exists(blratiomodel))
    boundratiomodel=blratiomodel;
Else
    boundratiomodel=3.0;
EndIf

If(Exists(nozzlefac))
    nozzle_factor=nozzlefac;
Else
    nozzle_factor=4.0;
EndIf

If(Exists(throatfac))
    throat_factor=throatfac;
Else
    throat_factor=8.0;
EndIf

If(Exists(modelfac))
    model_factor=modelfac;
Else
    model_factor=6.0;
EndIf

If(Exists(plumefac))
    plume_factor=plumefac;
Else
    plume_factor=3.0;
EndIf

If(Exists(spillfac))
    spill_factor=spillfac;
Else
    spill_factor=3.0;
EndIf

bigsize = basesize*4;     // the biggest mesh size
nozzlesize = basesize/nozzle_factor;     // background mesh size in the nozzle
throatsize = basesize/throat_factor;     // background mesh size in the nozzle
modelsize = basesize/model_factor; // background mesh size in the scramjet model region
plumesize = basesize/plume_factor; // background mesh size in the scramjet plume region
spillsize = basesize/spill_factor; // background mesh size in the gap between the nozzle and model
                                   //
Printf("bigsize = %f", bigsize);
Printf("nozzlesize = %f", nozzlesize);
Printf("modelsize = %f", modelsize);
Printf("plumesize = %f", plumesize);
Printf("spillsize = %f", spillsize);
Printf("boundrationozzle = %f", boundrationozzle);
Printf("boundratiomodel = %f", boundratiomodel);

chamber_wall_surfaces[] = {6,7,9,10};
chamber_outflow_surfaces[] = {101};

nozzle_outside_surfaces[] = {5,11};
nozzle_end_surfaces[] = {3, 4, 12, 13};
nozzle_inside_surfaces[] = {14:19, 21:24, 1:2};
nozzle_inflow_surfaces[] = {100};
nozzle_wall_surfaces[] = {
  nozzle_outside_surfaces[],
  nozzle_end_surfaces[],
  nozzle_inside_surfaces[]
};

scramjet_inlet_surfaces[] = {31, 32, 29, 25};
scramjet_inside_surfaces[] = {26,30};
scramjet_outlet_surfaces[] = {27, 34};
scramjet_outside_surfaces[] = {28,33};
scramjet_wall_surfaces[] = {
  scramjet_inlet_surfaces[],
  scramjet_inside_surfaces[],
  scramjet_outlet_surfaces[],
  scramjet_outside_surfaces[]
};

// make centerline
Point(100) = {-324., 0., 0., 1.};
Point(101) = {766., 0., 0., 1.};

// inlet/outlet/symmetry edges
Line(100) = {20, 100};
Line(101) = {9, 101};
Line(102) = {100, 101};

Curve Loop(1) = {100, 102, 101, 9:19};
Curve Loop(2) = {30:34};
Plane Surface(1) = {1, 2};

Physical Surface("fluid") = {-1};
Physical Curve("inflow") = {100 };
Physical Curve("outflow") = {101};
Physical Curve("axis") = {102};
Physical Curve("isothermal_wall") = {
  chamber_wall_surfaces[],
  nozzle_wall_surfaces[],
  scramjet_wall_surfaces[]
};



// tabulate some convenient locations
nozzle_begin = -330;
nozzle_end = -36.0;
throat_begin = -330;
throat_end = -245.0;
nozzle_radius = 33.5;
model_begin = -36;
model_end = 510.0;
model_radius = 28;
plume_radius = 75;
spill_begin = -40;
spill_end = 50.0;
spill_radius = 100;

//  background mesh size in the nozzle
Field[30] = Box;
Field[30].XMin = nozzle_begin;
Field[30].XMax = nozzle_end;
//Field[30].XMin = -1000000;
//Field[30].XMax = 1000000;
Field[30].YMin = -nozzle_radius;
Field[30].YMax = nozzle_radius;
//Field[30].YMin = -100000;
//Field[30].YMax = 100000;
Field[30].ZMin = -100000;
Field[30].ZMax = 100000;
Field[30].VIn = nozzlesize;
Field[30].VOut = bigsize;

// Inside and end surfaces of nozzle
Field[31] = Distance;
Field[31].Sampling = 1000;
Field[31].CurvesList = {
  nozzle_inside_surfaces[],
  nozzle_end_surfaces[]
};
Field[32] = Threshold;
Field[32].InField = 31;
Field[32].SizeMin = nozzlesize/boundrationozzle;
Field[32].SizeMax = bigsize;
Field[32].DistMin = 0;
Field[32].DistMax = 200;

//  background mesh size in the nozzle throat
Field[33] = Box;
Field[33].XMin = throat_begin;
Field[33].XMax = throat_end;
Field[33].YMin = -nozzle_radius;
Field[33].YMax = nozzle_radius;
Field[33].ZMin = -100000;
Field[33].ZMax = 100000;
Field[33].VIn = throatsize;
Field[33].VOut = bigsize;
Field[33].Thickness = 25;

// Inside and end surfaces of nozzle
Field[34] = Distance;
Field[34].Sampling = 1000;
Field[34].CurvesList = {15, 16, 17, 18, 19};
//Field[34].CurvesList = {16};
Field[35] = Threshold;
Field[35].InField = 34;
Field[35].SizeMin = throatsize/boundrationozzle;
Field[35].SizeMax = bigsize;
Field[35].DistMin = 0;
Field[35].DistMax = 200;

//  background mesh size in the scramjet model (downstream of the nozzle)
Field[40] = Box;
Field[40].XMin = model_begin;
Field[40].XMax = model_end;
Field[40].YMin = -model_radius;
Field[40].YMax = model_radius;
Field[40].ZMin = -100000;
Field[40].ZMax = 100000;
Field[40].VIn = modelsize;
Field[40].VOut = bigsize;
Field[40].Thickness = 50;

// scramjet model bl meshing
Field[41] = Distance;
Field[41].Sampling = 1000;
Field[41].CurvesList = {
  scramjet_inlet_surfaces[],
  scramjet_outlet_surfaces[],
  scramjet_inside_surfaces[]
 };
Field[42] = Threshold;
Field[42].InField = 41;
Field[42].SizeMin = modelsize/boundratiomodel;
Field[42].SizeMax = bigsize;
Field[42].DistMin = 0;
Field[42].DistMax = 150;

// outer scramjet model bl meshing
Field[43] = Distance;
Field[43].Sampling = 1000;
Field[43].CurvesList = {
  scramjet_outside_surfaces[]
 };
Field[44] = Threshold;
Field[44].InField = 43;
Field[44].SizeMin = basesize/4;
Field[44].SizeMax = bigsize;
Field[44].DistMin = 0;
Field[44].DistMax = 100;
Field[44].StopAtDistMax = 1;

//  background mesh size in the exhaust plume (downstream of the model)
Field[50] = Box;
Field[50].XMin = model_end;
Field[50].XMax = 10000;
Field[50].YMin = -plume_radius;
Field[50].YMax = plume_radius;
Field[50].ZMin = -100000;
Field[50].ZMax = 100000;
Field[50].VIn = plumesize;
Field[50].VOut = bigsize;
Field[50].Thickness = 100;

Field[60] = Box;
Field[60].XMin = spill_begin;
Field[60].XMax = spill_end;
Field[60].YMin = -spill_radius;
Field[60].YMax = spill_radius;
Field[60].ZMin = -100000;
Field[60].ZMax = 100000;
Field[60].VIn = spillsize;
Field[60].VOut = bigsize;
Field[60].Thickness = 125;

// Min of the two sections above
Field[100] = Min;
//Field[9].FieldsList = {2,4,6,8};
//Field[9].FieldsList = {30, 32, 40, 42};
Field[100].FieldsList = {30, 32, 33, 35, 40, 42, 44, 50, 60};

Background Field = 100;

Mesh.Algorithm = 8;
//Mesh.CharacteristicLengthMin = 1;
//Mesh.CharacteristicLengthMax = 50;
Mesh.ElementOrder = 1;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MshFileVersion = 2.2;
Mesh.RecombinationAlgorithm = 2;
Mesh.RecombineAll = 1;

//Mesh 3;
OptimizeMesh "Netgen";
Mesh.Smoothing = 100;

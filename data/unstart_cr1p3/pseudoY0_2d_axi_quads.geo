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
    boundrationozzle=8.0;
EndIf

If(Exists(blratiomodel))
    boundratiomodel=blratiomodel;
Else
    boundratiomodel=4.0;
EndIf

If(Exists(nozzlefac))
    nozzle_factor=nozzlefac;
Else
    nozzle_factor=10.0;
EndIf

If(Exists(throatfac))
    throat_factor=throatfac;
Else
    throat_factor=18.0;
EndIf

If(Exists(modelfac))
    model_factor=modelfac;
Else
    model_factor=8.0;
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

// Delete edges in the positive y plane and rotate about z so that the model so our
// symmetry axis is along y.
//
Recursive Delete{ Line{8:20, 30:34}; }

// remove the little cut at the end of the nozzle
Delete { Line{3:5}; }
Delete { Point{4}; }
Translate {1.0, 0, 0} { Point{5}; }
Line(90) = {5, 3};
Line(91) = {6, 5};

// make centerline
Point(100) = {-324., 0., 0., 1.};
Point(101) = {766., 0., 0., 1.};

// inlet/outlet/symmetry edges
Line(100) = {21, 100};
Line(101) = {8, 101};
Line(102) = {100, 101};

Curve Loop(1) = {100, 102, 101, 7, 6, 91, 90, 2, 1, 24:21};
Curve Loop(2) = {25:29};
Plane Surface(1) = {1, 2};

Rotate{{0, 0, 1.}, {0., 0., 0.}, Pi/2} { Surface{:}; }

chamber_wall_surfaces[] = {4,5};
chamber_outflow_surfaces[] = {3};

nozzle_outside_surfaces[] = {6};
nozzle_end_surfaces[] = {7};
nozzle_inside_surfaces[] = {8:13};
nozzle_inflow_surfaces[] = {1};
nozzle_wall_surfaces[] = {
  nozzle_outside_surfaces[],
  nozzle_end_surfaces[],
  nozzle_inside_surfaces[]
};

scramjet_inlet_surfaces[] = {14, 18};
scramjet_inside_surfaces[] = {15};
scramjet_outlet_surfaces[] = {16};
scramjet_outside_surfaces[] = {17};
scramjet_wall_surfaces[] = {
  scramjet_inlet_surfaces[],
  scramjet_inside_surfaces[],
  scramjet_outlet_surfaces[],
  scramjet_outside_surfaces[]
};

Physical Surface("fluid") = {1};
Physical Curve("inflow") = {1};
Physical Curve("outflow") = {3};
Physical Curve("symmetry") = {2};
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
spill_end = 150.0;
spill_radius = 100;

//  background mesh size in the nozzle
Field[30] = Box;
Field[30].YMin = nozzle_begin;
Field[30].YMax = nozzle_end;
//Field[30].XMin = -1000000;
//Field[30].XMax = 1000000;
Field[30].XMin = -nozzle_radius;
Field[30].XMax = nozzle_radius;
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
Field[32].StopAtDistMax = 1;

//  background mesh size in the nozzle throat
Field[33] = Box;
Field[33].YMin = throat_begin;
Field[33].YMax = throat_end;
Field[33].XMin = -nozzle_radius;
Field[33].XMax = nozzle_radius;
Field[33].ZMin = -100000;
Field[33].ZMax = 100000;
Field[33].VIn = throatsize;
Field[33].VOut = bigsize;
Field[33].Thickness = 25;

// Nozzle throat bl
Field[34] = Distance;
Field[34].Sampling = 1000;
Field[34].CurvesList = {9:12};
//Field[34].CurvesList = {16};
Field[35] = Threshold;
Field[35].InField = 34;
Field[35].SizeMin = throatsize/boundrationozzle;
Field[35].SizeMax = bigsize;
Field[35].DistMin = 0;
Field[35].DistMax = 200;
Field[35].StopAtDistMax = 1;

//  background mesh size in the scramjet model (downstream of the nozzle)
Field[40] = Box;
Field[40].YMin = model_begin;
Field[40].YMax = model_end;
Field[40].XMin = -model_radius;
Field[40].XMax = model_radius;
Field[40].ZMin = -100000;
Field[40].ZMax = 100000;
Field[40].VIn = modelsize;
Field[40].VOut = bigsize;
Field[40].Thickness = 50;

// scramjet model bl meshing
Field[41] = Distance;
Field[41].Sampling = 1000;
Field[41].CurvesList = {
  scramjet_inside_surfaces[]
 };
Field[42] = Threshold;
Field[42].InField = 41;
Field[42].SizeMin = modelsize/boundratiomodel;
Field[42].SizeMax = bigsize;
Field[42].DistMin = 0;
Field[42].DistMax = 200;

// scramjet model bl meshing
Field[141] = Distance;
Field[141].Sampling = 1000;
Field[141].CurvesList = {
  scramjet_outlet_surfaces[]
 };
Field[142] = Threshold;
Field[142].InField = 141;
Field[142].SizeMin = modelsize/boundratiomodel;
Field[142].SizeMax = bigsize;
Field[142].DistMin = 0;
Field[142].DistMax = 600;

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
Field[44].DistMax = 200;
Field[44].StopAtDistMax = 1;

// scramjet model tip meshing
Field[45] = Distance;
//Field[45].Sampling = 1000;
Field[45].PointsList = {15};
Field[46] = Threshold;
Field[46].InField = 45;
Field[46].SizeMin = modelsize/boundratiomodel/4;
Field[46].SizeMax = bigsize;
Field[46].DistMin = 0;
Field[46].DistMax = 300;

Field[47] = Distance;
Field[47].Sampling = 1000;
Field[47].CurvesList = {
  scramjet_inlet_surfaces[]
 };
Field[48] = Threshold;
Field[48].InField = 47;
Field[48].SizeMin = modelsize/boundratiomodel/2;
Field[48].SizeMax = bigsize;
Field[48].DistMin = 0;
Field[48].DistMax = 800;

//  background mesh size in the exhaust plume (downstream of the model)
Field[50] = Box;
Field[50].YMin = model_end;
Field[50].YMax = 10000;
Field[50].XMin = -plume_radius;
Field[50].XMax = plume_radius;
Field[50].ZMin = -100000;
Field[50].ZMax = 100000;
Field[50].VIn = plumesize;
Field[50].VOut = bigsize;
Field[50].Thickness = 100;

Field[60] = Box;
Field[60].YMin = spill_begin;
Field[60].YMax = spill_end;
Field[60].XMin = -spill_radius;
Field[60].XMax = spill_radius;
Field[60].ZMin = -100000;
Field[60].ZMax = 100000;
Field[60].VIn = spillsize;
Field[60].VOut = bigsize;
Field[60].Thickness = 125;

// Min of the two sections above
Field[100] = Min;
//Field[9].FieldsList = {2,4,6,8};
//Field[9].FieldsList = {30, 32, 40, 42};
Field[100].FieldsList = {30, 32, 33, 35, 40, 42, 142, 44, 46, 48, 50, 60};
//Field[100].FieldsList = {30, 32, 33, 35, 40, 42, 142, 44, 46, 48, 50, 60};

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

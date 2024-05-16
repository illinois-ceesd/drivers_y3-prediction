Merge "pseudoY0.brep";
Mesh.Algorithm3D = 4;
Mesh.CharacteristicLengthMin = 1;
Mesh.CharacteristicLengthMax = 50;
Mesh.ElementOrder = 1;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.Smoothing = 100;
Mesh.MshFileVersion = 2.2;

chamber_wall_surfaces[] = {1:2};
chamber_outflow_surfaces[] = {3};

nozzle_outside_surfaces[] = {4:5};
nozzle_end_surfaces[] = {6:8};
nozzle_inside_surfaces[] = {9:20};
nozzle_inflow_surfaces[] = {21};
nozzle_wall_surfaces[] = {
  nozzle_outside_surfaces[],
  nozzle_end_surfaces[],
  nozzle_inside_surfaces[]
};

scramjet_inlet_surfaces[] = {22,24};
scramjet_inside_surfaces[] = {23};
scramjet_outlet_surfaces[] = {25};
scramjet_outside_surfaces[] = {26};
scramjet_wall_surfaces[] = {
  scramjet_inlet_surfaces[],
  scramjet_inside_surfaces[],
  scramjet_outlet_surfaces[],
  scramjet_outside_surfaces[]
};

Physical Volume("Volume") = {1};
Physical Surface("Inflow") = { nozzle_inflow_surfaces[] };
Physical Surface("Outflow") = { chamber_outflow_surfaces[] };
Physical Surface("Wall") = {
  chamber_wall_surfaces[],
  nozzle_wall_surfaces[],
  scramjet_wall_surfaces[]
};

// Inside and end surfaces of nozzle
Field[1] = Distance;
Field[1].NNodesByEdge = 100;
Field[1].FacesList = { nozzle_inside_surfaces[], nozzle_end_surfaces[] };
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = 1;
Field[2].LcMax = 50;
Field[2].DistMin = 0;
Field[2].DistMax = 200;

// Edges separating nozzle surfaces with boundary layer refinement
// from those without (seems to give a smoother transition)
Field[3] = Distance;
Field[3].NNodesByEdge = 100;
nozzle_bl_transition_curves[] = CombinedBoundary { Surface {
  nozzle_inflow_surfaces[],
  nozzle_inside_surfaces[],
  nozzle_end_surfaces[]
}; };
// CombinedBoundary{} here returns the two curves we want plus two more seemingly-nonsensical
// ones (CAD issue maybe?)
nozzle_bl_transition_curves_fixed[] = {
  nozzle_bl_transition_curves[0],
  nozzle_bl_transition_curves[1]
};
Field[3].EdgesList = { nozzle_bl_transition_curves_fixed[] };
Field[4] = Threshold;
Field[4].IField = 3;
Field[4].LcMin = 1;
Field[4].LcMax = 50;
Field[4].DistMin = 0;
Field[4].DistMax = 200;

// Inside and end surfaces of scramjet
Field[5] = Distance;
Field[5].NNodesByEdge = 100;
Field[5].FacesList = {
  scramjet_inlet_surfaces[],
  scramjet_inside_surfaces[],
  scramjet_outlet_surfaces[]
};
Field[6] = Threshold;
Field[6].IField = 5;
Field[6].LcMin = 1;
Field[6].LcMax = 50;
Field[6].DistMin = 0;
Field[6].DistMax = 200;

// Edges separating scramjet surfaces with boundary layer refinement
// from those without (seems to give a smoother transition)
Field[7] = Distance;
Field[7].NNodesByEdge = 100;
Field[7].EdgesList = { CombinedBoundary { Surface {
  scramjet_inlet_surfaces[],
  scramjet_inside_surfaces[],
  scramjet_outlet_surfaces[]
}; } };
Field[8] = Threshold;
Field[8].IField = 7;
Field[8].LcMin = 1;
Field[8].LcMax = 50;
Field[8].DistMin = 0;
Field[8].DistMax = 200;

// Min of the two sections above
Field[9] = Min;
Field[9].FieldsList = {2,4,6,8};

Background Field = 9;

Mesh 3;
OptimizeMesh "Netgen";

//+
SetFactory("OpenCASCADE");
Box(1) = {-0.2, 11.8, -0.5, 1, 1, 1};
//+
//+
MeshSize {4, 8, 7, 3, 1, 5, 2, 6} = 0.01e;
//+
Field[1] = Box;
//+
Field[1].VIn = 0.1;
//+
Field[1].VOut = 10;
//+
Field[1].XMax = 100;
//+
Field[1].XMin = -100;
//+
Field[1].YMax = 100;
//+
Field[1].YMin = -100;
//+
Field[1].ZMax = 100;
//+
Field[1].ZMin = -100;
//+
Background Field = 1;
//+
Recombine Surface {4, 6, 1, 3, 2, 5};

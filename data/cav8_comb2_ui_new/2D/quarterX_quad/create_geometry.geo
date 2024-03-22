SetFactory("OpenCASCADE");
//Merge "pseudoY0_2d.brep";
surface_vector[] = ShapeFromFile("actii-2d.brep");

// make some new lines to setup transfinite meshing
iso_bl_thickness = 3.5;
comb_bl_thickness = 3.5;
cavity_bl_thickness = 1.5;
injector_bl_thickness = 0.68;
// top bl
Translate {0., -iso_bl_thickness,  0.} { Duplicata{ Line{26:23}; } }
//Dilate { {928.28, 21.674, 0.}, 1+iso_bl_thickness/10. } {Duplicata{ Line{22}; } }
Dilate { {928.28, 21.674, 0.}, 1.8} {Duplicata{ Line{22}; } }

// temporary points/lines to cut the throat section
// downstream
Point(1000) = {305., 10., 0., 1.};
Point(1001) = {305., -10., 0., 1.};
Line(1000) = {1000:1001};
// upstream
Point(1002) = {268., 10., 0., 1.};
Point(1003) = {268., -10., 0., 1.};
Line(1001) = {1002:1003};
BooleanFragments{ Line{24, 50}; Delete;}{ Line{1000}; Delete; }
BooleanFragments{ Line{25, 49}; Delete;}{ Line{1001}; Delete; }
Recursive Delete { Line{1013, 1006}; }
Recursive Delete { Line{1011, 1005}; }
//Delete { Point{1007, 1009, 1014, 1008, 1011}; }

// bottom isolator bl
Translate {0., iso_bl_thickness,  0.} { Duplicata{ Line{28:31}; } }
BooleanFragments{ Line{1019, 31}; Delete;}{ Line{1008}; Delete; }
BooleanFragments{ Line{30, 1018}; Delete;}{ Line{1015}; Delete; }
Recursive Delete { Line{1019, 1028, 1032, 1025}; }
//Delete { Point{1030, 1036, 1034, 1039}; }

// bottom isolator bl, post injection
Translate {0., iso_bl_thickness,  0.} { Duplicata{ Line{2}; } }

// bottom combustor bl
Translate {0., comb_bl_thickness,  0.} { Duplicata{ Line{10:15}; } }
//Dilate { {928.28, -25.588, 0.}, 1+comb_bl_thickness/10. } {Duplicata{ Line{16}; } }
Dilate { {928.28, -25.588, 0.}, 2. } {Duplicata{ Line{16}; } }

// upstream injector
Translate {injector_bl_thickness, 0., 0.} { Duplicata{ Line{33}; } }
//Dilate { {531.94, -8.825, 0.}, 1+2*injector_bl_thickness } {Duplicata{ Line{32}; } }
Dilate { {531.94, -8.825, 0.}, 2.35 } {Duplicata{ Line{32}; } }
Translate {-injector_bl_thickness, 0., 0.} { Duplicata{ Line{35}; } }
//Dilate { {534.53, -8.825, 0.}, 1+2*injector_bl_thickness } {Duplicata{ Line{1}; } }
Dilate { {534.53, -8.825, 0.}, 2.35} {Duplicata{ Line{1}; } }

// cavity
Translate {cavity_bl_thickness, 0., 0.} { Duplicata{ Line{3}; } }
Translate {0., cavity_bl_thickness, 0.} { Duplicata{ Line{4}; } }
BooleanFragments{ Line{1044, 1045}; Delete;}{}
BooleanFragments{ Line{3}; Delete;}{ Line{1046}; } // front wall
BooleanFragments{ Line{4}; Delete;}{ Line{1045}; } // bottom wall

// cavity injector
Translate {0., injector_bl_thickness, 0.} { Duplicata{ Line{6}; } }
Translate {0., -injector_bl_thickness, 0.} { Duplicata{ Line{8}; } }
// extend into the cavity
//Translate { -cavity_bl_thickness, 0., 0.}{Duplicata{ Point{10}; }}

// instead of using translate, just make a new point so we don't perturn the numbering
// get the coordinates and assign a non-interfering number
xyz[] = Point{10};
Point(100) = {xyz[0] - cavity_bl_thickness, xyz[1], xyz[2]};
Line(200) = {1079, 100}; // cavtiy wall bl
Translate {-10, 0., 0.} { Duplicata{ Point{9}; } }
Line(201) = {1084, 9};
Translate {0., -injector_bl_thickness, 0.} { Duplicata{ Line{201}; } }
Translate {-10, 0., 0.} { Duplicata{ Point{6}; } }
Line(202) = {1087, 6};
Translate {0., injector_bl_thickness, 0.} { Duplicata{ Line{202}; } }
BooleanFragments{ Line{201, 1054, 1055, 202}; Delete;}{ Line{200}; Delete;} 
Recursive Delete{Line{1054, 1056, 1058, 1060};}
Line(203) = {6, 9};
BooleanFragments{ Line{1057, 1052}; Delete;}{ Line{203}; Delete;} 
Delete{Line{1068, 1053};}
Delete{Point{1083, 1096};}
Line(204) = {1095, 1082};
Delete{Line{1059, 1070};}
Delete{Point{1099, 1091};}
Line(205) = {1090, 1097};
//Delete{Point{1084, 1086, 1089, 1092};}

// Connecting lines
// upstream injector
Line(300) = {32, 1070};
Line(301) = {1070, 1032};
Line(302) = {1, 1073};
Line(303) = {1073, 1039};
Line(304) = {1069, 1074};

// cavity
Line(310) = {3, 1075};
Line(311) = {3, 1040};
Line(312) = {10, 1053};
Line(313) = {5, 1079};
Line(314) = {1094, 10};

// exhaust
Line(320) = {17, 1065};
Line(321) = {22, 65};

// split boundaries that got bl meshes
BooleanFragments{ Line{27}; Delete;}{ Line{48, 1016}; Delete;} // inlet
BooleanFragments{ Line{34}; Delete;}{ Line{1040, 1042}; Delete;} // upstream injection
BooleanFragments{ Line{7}; Delete;}{ Line{204, 1069}; Delete;} // cavity injection
                                                                //
// extra edges to help refine mesh
// coming into upstream injection
Point(300) = {520., -9., 0.};
Point(301) = {520., -4., 0.};
Line(400) = {300, 301};

// coming out of nozzle 
Point(310) = {383.28, -9., 0.};
Point(311) = {383.28, -4., 0.};
Line(401) = {310, 311};
Line(403) = {1005, 24};

// coming into the cavity
Point(320) = {595, -9., 0.};
Point(321) = {595, -4., 0.};
Line(402) = {320, 321};

BooleanFragments{ Line{400}; Delete;}{Line{1022, 1020}; Delete;}
BooleanFragments{ Line{401}; Delete;}{Line{1088, 1086}; Delete;}
BooleanFragments{ Line{402}; Delete;}{Line{2, 1032}; Delete;}
Recursive Delete{Line{1099, 1097, 1085, 1083, 1092, 1090};}
//
// make sure things line up nicely
// this cleans up the connectiions between the bl fillets that were created with Dilate
// exhaust
Recursive Delete { Line{51}; }
//Point(100) = {800., 8.1755, 0.};
Point(100) = {800., 7., 0.};
Line(51) = {1005, 100};
Line(53) = {100, 64};
Recursive Delete { Line{1038, 1037}; }
//Line(1038) = {1062, 1066};
Point(101) = {800., -7, 0.};
Point(102) = {885.84, -7, 0.};
Line(1038) = {1060, 101};
Line(600) = {101, 102};
Line(601) = {102, 1066};
// upstream injector
Recursive Delete{Line{1040};}
Line(1040) = {1067, 1069};
Recursive Delete{Line{1042};}
Line(1042) = {1072, 1074};

// extend the exhaust bl to the edge of the domain
Translate {40., 0., 0.} { Duplicata{ Point{21}; } }
Translate {40., 0., 0.} { Duplicata{ Point{18}; } }
BooleanFragments{ Point{1110}; Delete;}{Line{20}; Delete;}
BooleanFragments{ Point{1111}; Delete;}{Line{18}; Delete;}
Line(500) = {65, 1110};
Line(501) = {1065, 1111};

Coherence;
//

Curve Loop(100) = {1075, 1016, 1017, -1029, -1030, 1012, -48};
Plane Surface(100) = {100};
Curve Loop(101) = {1023, 1093, 1094, 1089, -301, -1041, 304, -1043, 303, 1102, 1103, -311, 310, 1044, 1047, 1062, 1063, 1064, 1065, 1066, 314, 312, 1033, 1034, 1035, 1036, 1038, 600, 601, -1039, 501, -1106, -19, 1104, 500, -52, -53, -51, 1004};
Plane Surface(101) = {101};
Curve Loop(3) = {1030, 1031, -1026, 1021, -1024, -1023, -1007, 1003, 1009, 1014};
Plane Surface(3) = {3};
Curve Loop(4) = {1074, 48, -1012, -1014, 1010, -26};
Plane Surface(4) = {4};
Curve Loop(5) = {1076, 28, 29, -1027, -1031, 1029, -1017, -1016};
Plane Surface(5) = {5};
Curve Loop(6) = {1004, -1007, -1002, -403};
Plane Surface(6) = {6};
Curve Loop(7) = {51, 53, 52, 500, 1105, -21, -22, -23, -403};
Plane Surface(7) = {7};
Curve Loop(8) = {1095, 1091, -1093, 1024};
Plane Surface(8) = {8};
Curve Loop(9) = {1091, 1094, -1084, -1096};
Plane Surface(9) = {9};
Curve Loop(10) = {1084, 1089, -301, -300, -1087};
Plane Surface(10) = {10};
Curve Loop(11) = {303, 1102, -1098, -1100, 302};
Plane Surface(11) = {11};
Curve Loop(12) = {1098, 1103, -311, -1101};
Plane Surface(12) = {12};
Curve Loop(13) = {312, 1033, 1034, 1035, 1036, 1038, 600, 601, -1039, 501, 1107, -17, 16, -15, -14, -13, -12, -11, -10};
Plane Surface(13) = {13};
Curve Loop(14) = {1048, 1046, -1044, -310};
Plane Surface(14) = {14};
Curve Loop(15) = {1049, 1050, -1045, -1046};
Plane Surface(15) = {15};
Curve Loop(16) = {1051, 313, -1047, 1045};
Plane Surface(16) = {16};
Curve Loop(17) = {5, -1061, -1062, -313};
Plane Surface(17) = {17};
Curve Loop(18) = {1061, 1071, -1108, -1063};
Plane Surface(18) = {18};
Curve Loop(19) = {1108, 1072, -1067, -1064};
Plane Surface(19) = {19};
Curve Loop(20) = {1067, 1073, -1055, -1065};
Plane Surface(20) = {20};
Curve Loop(21) = {1055, 9, 314, -1066};
Plane Surface(21) = {21};
Curve Loop(22) = {6, 1071, -1069, 1082};
Plane Surface(22) = {22};
Curve Loop(24) = {204, -1080, 8, -1073};
Plane Surface(23) = {24};
Curve Loop(25) = {1069, 1072, 204, 1081};
Plane Surface(24) = {25};
Curve Loop(26) = {35, 1077, 1042, -1043, -302, 1};
Plane Surface(25) = {26};
Curve Loop(28) = {1079, 33, 32, 300, -1041, -1040};
Plane Surface(26) = {28};
Curve Loop(29) = {1078, 1040, 304, -1042};
Plane Surface(27) = {29};
Curve Loop(200) = {42, -41, -12, 43};
Plane Surface(200) = {200};
Curve Loop(202) = {38, -39, -13, 41, -42, -43, -11, 37};
Plane Surface(202) = {202};

Save "geometry_trans.geo_unrolled";
//+

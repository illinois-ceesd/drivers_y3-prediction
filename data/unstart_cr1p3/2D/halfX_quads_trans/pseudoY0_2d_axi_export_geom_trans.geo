SetFactory("OpenCASCADE");
//Merge "pseudoY0_2d.brep";
surface_vector[] = ShapeFromFile("pseudoY0_2d.brep");

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

// Delete the model spline and make a simple line
Delete { Line{26}; }
Line(26) = {27, 25};

// centerline
// inflow
Point(100) = {-324., 0., 0., 1.};
// outflow
Point(101) = {766., 0., 0., 1.};

// make some new lines to setup transfinite meshing
bl_thickness = 2.5;
// nozzle
Translate {0., bl_thickness,  0.} { Duplicata{ Line{1:2, 21:24}; } }
// inside model
Translate {0., bl_thickness,  0.} { Duplicata{ Line{25:27}; } }
// outside model
Translate {0., -bl_thickness,  0.} { Duplicata{ Line{29}; } }
// model tip
Translate {-5.0, 0., 0.} { Duplicata{ Point{121, 26, 124}; } }

// Extra nozzle lines
Line(300) = {3, 111};
Line(301) = {2, 110};
Line(302) = {1, 109};
Line(303) = {24, 115};
Line(304) = {23, 114};
Line(305) = {22, 112};
Line(306) = {21, 113};

// Extra model lines
Line(200) = {126, 121};
Line(201) = {127, 26};
Line(202) = {128, 124};
Line(203) = {128, 127};
Line(204) = {127, 126};
Line(205) = {124, 26};
Line(206) = {26, 121};
Line(207) = {125, 29};
Line(208) = {25, 120};
Line(209) = {27, 122};
Line(210) = {28, 123};

// make extra centerline points
// nozzle throat
Point(102) = {-289., 0., 0., 1.};
// nozzle end
Point(103) = {-36., 0., 0., 1.};
// model entrance
Point(104) = {-20.9, 0., 0., 1.};
// model flat begin
Point(105) = {-2., 0., 0., 1.};
// model flat end
Point(106) = {486., 0., 0., 1.};
// model exit
Point(107) = {506., 0., 0., 1.};
// nozzle transition
Point(108) = {-249., 0., 0., 1.};

Line(402) = {100, 102};
Line(403) = {102, 108};
Line(404) = {103, 104};
Line(405) = {104, 105};
Line(406) = {105, 106};
Line(407) = {106, 107};
Line(408) = {107, 101};
Line(409) = {110, 108};
Line(410) = {108, 103};

Line(500) = {109, 102};
Line(501) = {111, 103};
Line(502) = {126, 104};
Line(503) = {120, 105};
Line(504) = {122, 106};
Line(505) = {123, 107};

// inlet/outlet/symmetry edges
Line(400) = {113, 100};
Point(250) = {766., -60, 0, 1.};
Point(251) = {766., -120, 0, 1.};
Line(401) = {250, 101};
Line(451) = {251, 250};
Line(452) = {8, 251};

// model bl surfaces
Curve Loop(200) = {202, 203, 201, 205};
Curve Loop(201) = {201, 204, 200, 206};
Curve Loop(202) = {25, 206, 98, 208};
Curve Loop(203) = {101, 205, 29, 207};
Curve Loop(204) = {26, 208, 99, 209};
Curve Loop(205) = {27, 209, 100, 210};
Plane Surface(200) = {200};
Plane Surface(201) = {201};
Plane Surface(202) = {202};
Plane Surface(203) = {203};
Plane Surface(204) = {204};
Plane Surface(205) = {205};

Curve Loop(220) = {502, 405, -503, -98, 200};
Curve Loop(221) = {503, 406, 504, 99};
Curve Loop(222) = {504, 407, 505, 100};
Plane Surface(220) = {220};
Plane Surface(221) = {221};
Plane Surface(222) = {222};


// nozzle bl surfaces
Curve Loop(300) = {21, 306, -94, -305};
Curve Loop(301) = {22, 305, 95, 304};
Curve Loop(302) = {23, 304, 96, 303};
Curve Loop(303) = {24, 303, 97, 302};
Curve Loop(304) = {1, 302, 92, 301};
Curve Loop(305) = {2, 301, 93, 300};
Plane Surface(300) = {300};
Plane Surface(301) = {301};
Plane Surface(302) = {302};
Plane Surface(303) = {303};
Plane Surface(304) = {304};
Plane Surface(305) = {305};

Curve Loop(320) = {400, 402, -500, 97, 96, 95, 94};
Curve Loop(321) = {500, 403, -409, 92};
Curve Loop(322) = {409, 410, -501, 93};
Plane Surface(320) = {320};
Plane Surface(321) = {321};
Plane Surface(322) = {322};

// connect the nozzle outlet with model inlet
Line(1000) = {111, 126};
Line(1001) = {5, 128};

Curve Loop(1000) = {501, 404, -502, -1000};
Curve Loop(1001) = {90, 300, 1000, 204, 203, 1001};
Plane Surface(1000) = {1000};
Plane Surface(1001) = {1001};

// external bl and plume
Point(200) = {506, -60, 0., 1.0};
Line(1100) = {125, 200};
Line(1101) = {200, 28};
Line(1102) = {200, 250};

Curve Loop(1100) = {1101, 210, 505, 408, 401, 1102};
Curve Loop(1101) = {207, 28, 1101, 1100};
Plane Surface(1100) = {1100};
Plane Surface(1101) = {1101};

// model spill
Point(210) = {9.5, -90, 0., 1.0};
Point(211) = {506, -120, 0., 1.0};
Line(1200) = {5, 210};
Line(1201) = {210, 125};
Line(1202) = {210, 211};
Line(1203) = {211, 200};
Line(1204) = {211, 251};

Curve Loop(1200) = {1001, 202, 101, 1201, 1200};
Curve Loop(1201) = {1201, 1100, 1203, 1202};
Curve Loop(1202) = {1203, 1102, 451, 1204};
Plane Surface(1200) = {1200};
Plane Surface(1201) = {1201};
Plane Surface(1202) = {1202};

// Split the end by the nozzle
Delete { Line{6}; }
Point(260) = {-162.9, -130, 0., 1.};
Line(1210) = {7, 260};
Line(1211) = {260, 6};
Line(1212) = {260, 210};

Curve Loop (1210) = {1211, 91, 1200, 1212};
Plane Surface (1210) = {1210};

// everything else
Curve Loop(1) = {1210, 1212, 1202, 1204, 452, 7};
Plane Surface (1) = {1};

Rotate{{0, 0, 1.}, {0., 0., 0.}, Pi/2} { Surface{:}; }

Save "geometry_trans.geo_unrolled";

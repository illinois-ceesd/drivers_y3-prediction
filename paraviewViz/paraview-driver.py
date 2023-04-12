# Import Paraview functions
import sys
#from contour import *
from slice import SimpleSlice
import slice


def main(user_input_file, viz_path, dump_index):

    # read input file to configure plot

    # camera location (x, y, zoom)
    # camera location (bigger moves right, bigger moves up , smaller to zoom in)
    camera = [0.625, -0.0, 0.023]
    pixels = [1300, 700]
    prefix = ""

    print(f"{viz_path}")
    print(f"{dump_index}")
    print(f"{prefix}")

    slice_data = slice.SliceData(
        dataName="cv_mass",
        dataRange=[0.02, 0.1],
        camera=[0.625, 0.0, 0.023],
        colorScheme="erdc_rainbow_dark",
        logScale=0,
        invert=0,
        cbTitle="Density [kg/m^3]",
        pixels=[1300, 700],
        normal=[0, 0, 1],
        origin=[0, 0, 0]
    )

    print(slice_data)

    SimpleSlice(viz_path, dump_index, "cv_mass", [0.02, 0.1], camera,
                invert=1, logScale=1, colorScheme="erdc_rainbow_dark",
                prefix=prefix, pixels=pixels,
                cbTitle="Density [kg/m^3]")

    """
    simpleSlice(dir, dump_index, "dv_pressure", 1500.0, 10000, camera, invert=1,
                colorScheme="GREEN-WHITE_LINEAR", prefix=prefix, pixels=pixels,
                cbTitle="Pressure [Pa]")

    simpleSlice(dir, dump_index, "dv_temperature", 200.0, 1000, camera, invert=0,
                colorScheme="Black-Body Radiation", prefix=prefix, pixels=pixels,
                cbTitle="Temperature [K]")

    simpleSlice(dir, dump_index, "mach", 0.0, 3.5, camera, invert=0,
                colorScheme="Rainbow Desaturated", prefix=prefix, pixels=pixels,
                cbTitle="Mach Number")

    simpleSlice(dir, dump_index, "mu", 1.e-5, 1.e-4, camera, invert=0,
                colorScheme="Rainbow Desaturated", prefix=prefix, pixels=pixels,
                cbTitle="Viscosity [Pa-s")
                """


import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Y3 paraview visualization driver")
    parser.add_argument("-d", "--dump_index", type=int, dest="dump_index",
                        nargs="?", action="store", help="simulation viz dump index")
    parser.add_argument("-p", "--viz_path", type=ascii, dest="viz_path",
                        nargs="?", action="store", help="full path to viz data")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")

    args = parser.parse_args()

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Using user input from file: {input_file}")

    viz_path = "viz_data"
    if args.viz_path:
        viz_path = args.viz_path.replace("'", "")

    dump_index = 0
    if args.dump_index:
        dump_index = args.dump_index.replace("'", "")

    print(f"Running {sys.argv[0]}\n")

    main(user_input_file=input_file, viz_path=viz_path,
         dump_index=dump_index)

import cantera as ct
import pyrometheus as pyro


def generate_mechfile(mech_file):
    """This produces the mechanism codes."""

    sol = ct.Solution(f"{mech_file}", "gas")
    from pathlib import Path

    mech_output_file = Path(f"{mech_file}").stem
    with open(f"{mech_output_file}.py", "w") as file:
        code = pyro.codegen.python.gen_thermochem_code(sol)
        print(code, file=file)


import logging
import sys
import argparse

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Generate Pyrometheus mechanisms")
    parser.add_argument("-f", "--mech_file", type=ascii, dest="mech_file",
                        nargs="?", action="store", help="mechanism file")

    args = parser.parse_args()

    # get mechanism name from the arguments
    from mirgecom.simutil import ApplicationOptionsError
    mech_file = ""
    if args.mech_file:
        print(f"Processing Cantera mechanism file {args.mech_file}")
        mech_file = args.mech_file.replace("'", "")
    else:
        raise ApplicationOptionsError("Missing Cantera mechanism file from input")

    print(f"Running {sys.argv[0]}\n")

    generate_mechfile(mech_file)

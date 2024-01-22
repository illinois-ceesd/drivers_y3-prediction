import cantera as ct
import numpy as np
from matplotlib import pyplot as plt


def check_mechanism(mech_file):
    """Make plot for material properties for species in a mechanism."""

    gas = ct.Solution(f"{mech_file}", "gas")
    gas.TP = 300, 101325

    species_names = gas.species_names

    for species in species_names:
        gas.X = f"{species}:1"

        numpts = 101
        lnumpts = 10
        temperature_list = np.linspace(100, 5000, numpts)
        low_temp_list = np.linspace(1, 91, lnumpts)
        temperature_list = np.insert(temperature_list, 0, low_temp_list)

        numpts += lnumpts
        visc = np.zeros(numpts,)
        enthalpy = np.zeros(numpts,)
        cp = np.zeros(numpts,)
        cv = np.zeros(numpts,)
        ii = 0
        for temp in temperature_list:
            gas.TP = temp, 101325
            visc[ii] = gas.viscosity
            enthalpy[ii] = gas.enthalpy_mass
            cp[ii] = gas.cp_mass
            cv[ii] = gas.cv_mass
            ii += 1

        gamma = cp/cv


        plt.close("all")
        fig = plt.figure(1, figsize=[6.4, 4.8])
        ax1 = fig.add_subplot(111)
        ax1.set_title(f"{species}")
        ax1.set_position([0.16, 0.12, 0.75, 0.83])
        ax1.plot(temperature_list, visc, "--", label="Current")
        ax1.set_ylabel(r"$\mathbf{Viscosity \, (Pa-s)}$")
        ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
        plt.savefig(f"viscosity_{species}.png", dpi=200)
        ax1.legend()
        plt.show()

        plt.close("all")
        fig = plt.figure(1, figsize=[6.4, 4.8])
        ax1 = fig.add_subplot(111)
        ax1.set_title(f"{species}")
        ax1.set_position([0.16, 0.12, 0.75, 0.83])
        ax1.plot(temperature_list, enthalpy, "--", label="Current")
        ax1.set_ylabel(r"$\mathbf{Enthalpy \, (J/kg)}$")
        ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
        plt.savefig(f"enthalpy_{species}.png", dpi=200)
        ax1.legend()
        plt.show()

        plt.close("all")
        fig = plt.figure(1, figsize=[6.4, 4.8])
        ax1 = fig.add_subplot(111)
        ax1.set_title(f"{species}")
        ax1.set_position([0.16, 0.12, 0.75, 0.83])
        ax1.plot(temperature_list, cp, "--", label="Current")
        ax1.set_ylabel(r"$\mathbf{Heat \; Capacity (Cp)\, (J/kg-K)}$")
        ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
        plt.savefig(f"heatCapacityCp_{species}.png", dpi=200)
        ax1.legend()
        plt.show()

        plt.close("all")
        fig = plt.figure(1, figsize=[6.4, 4.8])
        ax1 = fig.add_subplot(111)
        ax1.set_title(f"{species}")
        ax1.set_position([0.16, 0.12, 0.75, 0.83])
        ax1.plot(temperature_list, cv, "--", label="Current")
        ax1.set_ylabel(r"$\mathbf{Heat \; Capacity (Cv)\, (J/kg-K)}$")
        ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
        plt.savefig(f"heatCapacityCv_{species}.png", dpi=200)
        ax1.legend()
        plt.show()
        plt.close("all")

        fig = plt.figure(1, figsize=[6.4, 4.8])
        ax1 = fig.add_subplot(111)
        ax1.set_title(f"{species}")
        ax1.set_position([0.16, 0.12, 0.75, 0.83])
        ax1.plot(temperature_list, gamma, "--", label="Current")
        ax1.set_ylabel(r"$\mathbf{gamma}$")
        ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
        plt.savefig(f"gamma_{species}.png", dpi=200)
        ax1.legend()
        plt.show()


import sys
import argparse

if __name__ == "__main__":

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

    check_mechanism(mech_file)

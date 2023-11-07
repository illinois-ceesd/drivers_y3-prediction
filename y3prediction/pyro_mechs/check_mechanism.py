import cantera as ct
import numpy as np
from matplotlib import pyplot as plt

gas = ct.Solution("my_mechanism.yaml")
gas.TP = 300, 101325
gas.X = "C2H4:1"

numpts = 43
visc = np.zeros(numpts,)
enthalpy = np.zeros(numpts,)
cp = np.zeros(numpts,)
cv = np.zeros(numpts,)
temperature_list = np.linspace(300, 2400, numpts)
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
ax1.set_position([0.16, 0.12, 0.75, 0.83])
ax1.plot(temperature_list, visc, "--", label="Current")
ax1.set_ylabel(r"$\mathbf{Viscosity \, (Pa-s)}$")
ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
plt.savefig("viscosity.png", dpi=200)
ax1.legend()
plt.show()

plt.close("all")
fig = plt.figure(1, figsize=[6.4, 4.8])
ax1 = fig.add_subplot(111)
ax1.set_position([0.16, 0.12, 0.75, 0.83])
ax1.plot(temperature_list, enthalpy, "--", label="Current")
ax1.set_ylabel(r"$\mathbf{Enthalpy \, (J/kg)}$")
ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
plt.savefig("enthalpy.png", dpi=200)
ax1.legend()
plt.show()

plt.close("all")
fig = plt.figure(1, figsize=[6.4, 4.8])
ax1 = fig.add_subplot(111)
ax1.set_position([0.16, 0.12, 0.75, 0.83])
ax1.plot(temperature_list, cp, "--", label="Current")
ax1.set_ylabel(r"$\mathbf{Heat \; Capacity (Cp)\, (J/kg-K)}$")
ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
plt.savefig("heatCapacityCp.png", dpi=200)
ax1.legend()
plt.show()

plt.close("all")
fig = plt.figure(1, figsize=[6.4, 4.8])
ax1 = fig.add_subplot(111)
ax1.set_position([0.16, 0.12, 0.75, 0.83])
ax1.plot(temperature_list, cv, "--", label="Current")
ax1.set_ylabel(r"$\mathbf{Heat \; Capacity (Cv)\, (J/kg-K)}$")
ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
plt.savefig("heatCapacityCv.png", dpi=200)
ax1.legend()
plt.show()
plt.close("all")

fig = plt.figure(1, figsize=[6.4, 4.8])
ax1 = fig.add_subplot(111)
ax1.set_position([0.16, 0.12, 0.75, 0.83])
ax1.plot(temperature_list, gamma, "--", label="Current")
ax1.set_ylabel(r"$\mathbf{gamma}$")
ax1.set_xlabel(r"$\mathbf{Temperature \, (K)}$")
plt.savefig("gamma.png", dpi=200)
ax1.legend()
plt.show()

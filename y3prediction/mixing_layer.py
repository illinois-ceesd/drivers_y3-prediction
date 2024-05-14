"""mirgecom driver initializer for the mixing layer problem."""
import numpy as np
from mirgecom.fluid import make_conserved
from functools import partial


class MixingLayerCold:
    r"""Solution initializer for flow with a discontinuity.

    This initializer creates a physics-consistent flow solution
    given an initial thermal state (pressure, temperature) and an EOS.

    The solution varies across a planar interface defined by a tanh function
    located at disc_location for pressure, temperature, velocity, and mass fraction

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, nspecies=0,
            mach_fuel, mach_air,
            temp_fuel, temp_air,
            y_fuel, y_air,
            vorticity_thickness, pressure
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        nspecies: int
            specifies the number of mixture species
        y_fuel: numpy.ndarray
            fuel stream species mass fractions
        mach_fuel: float
            fuel stream mach number
        temp_fuel: float
            fuel stream temperature
        y_air: numpy.ndarray
            air stream species mass fractions
        mach_air: float
            air stream mach number
        temp_air: float
            air stream temperature
        vorticity_thickness: float
            mixing layer thickness
        pressure: float
            ambient pressure
        """

        if y_fuel is None:
            y_fuel = np.zeros(nspecies, dtype=object)
        if y_air is None:
            y_air = np.zeros(nspecies, dtype=object)

        self._y_fuel = y_fuel
        self._y_air = y_air

        self._nspecies = nspecies
        self._dim = dim
        self._pressure = pressure
        self._mach_fuel = mach_fuel
        self._temp_fuel = temp_fuel
        self._mach_air = mach_air
        self._temp_air = temp_air
        self._vorticity_thickness = vorticity_thickness

    def __call__(self, dcoll, x_vec, eos, *, time=0.0):
        """Create the mixture state at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        ypos = x_vec[1]
        actx = ypos.array_context

        # get the species mass fractions first
        xtanh = ypos/self._vorticity_thickness
        weight = 0.5*(1. - actx.np.tanh(xtanh))
        y = self._y_air + (self._y_fuel - self._y_air)*weight

        print(f"{self._y_air=}")
        print(f"{self._y_fuel=}")

        mass_air = eos.get_density(
            self._pressure, self._temp_air,
            species_mass_fractions=self._y_air)
        mass_fuel = eos.get_density(
            self._pressure, self._temp_fuel,
            species_mass_fractions=self._y_fuel)

        vel_air = np.zeros(self._dim, dtype=object)
        vel_fuel = np.zeros(self._dim, dtype=object)

        # we need cv to get gamma
        cv_air = make_conserved(dim=self._dim,
                                mass=mass_air,
                                energy=0.,
                                momentum=mass_air*vel_air,
                                species_mass=mass_air*self._y_air)
        cv_fuel = make_conserved(dim=self._dim,
                                mass=mass_fuel,
                                energy=0.,
                                momentum=mass_fuel*vel_fuel,
                                species_mass=mass_fuel*self._y_fuel)

        gamma_air = eos.gamma(cv_air, self._temp_air)
        gamma_fuel = eos.gamma(cv_fuel, self._temp_fuel)

        c_air = np.sqrt(gamma_air*self._pressure/mass_air)
        c_fuel = np.sqrt(gamma_fuel*self._pressure/mass_fuel)

        vel_air[0] = self._mach_air*c_air
        vel_fuel[0] = self._mach_fuel*c_fuel
        velocity = vel_air + (vel_fuel - vel_air)*weight

        r = eos.gas_const(species_mass_fractions=y)

        # enthalpy
        enthalpy_air = eos.get_enthalpy(self._temp_air, self._y_air)
        enthalpy_fuel = eos.get_enthalpy(self._temp_fuel, self._y_fuel)
        enthalpy = enthalpy_air + (enthalpy_fuel - enthalpy_air)*weight

        tseed = actx.np.zeros_like(enthalpy) + 400.

        # need this for lazy for some reason
        from mirgecom.utils import force_evaluation
        enthalpy = force_evaluation(actx, enthalpy)

        temperature = eos.temperature_from_enthalpy(
            enthalpy=enthalpy, temperature_seed=400., species_mass_fractions=y)

        # compute the density from the temperature and pressure
        mass = self._pressure/r/temperature
        internal_energy = eos.get_internal_energy(temperature, y)

        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        total_energy = mass*(internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim,
                              mass=mass,
                              energy=total_energy,
                              momentum=mass*velocity,
                              species_mass=mass*y)


def get_mesh(dim, size, layer_ratio, vorticity_thickness,
             transfinite=False, use_quads=False):
    """Generate a grid using `gmsh`."""

    height = 50*vorticity_thickness
    length = 250*vorticity_thickness
    bottom_inflow = np.zeros(shape=(dim,))
    top_inflow = np.zeros(shape=(dim,))
    bottom_outflow = np.zeros(shape=(dim,))
    top_outflow = np.zeros(shape=(dim,))

    bottom_inflow_ml = np.zeros(shape=(dim,))
    top_inflow_ml = np.zeros(shape=(dim,))
    bottom_outflow_ml = np.zeros(shape=(dim,))
    top_outflow_ml = np.zeros(shape=(dim,))

    # points on the inflow
    bottom_inflow[1] = -height/2.
    bottom_inflow_ml[1] = -2*vorticity_thickness
    top_inflow[1] = height/2.
    top_inflow_ml[1] = 2*vorticity_thickness

    # points on the outflow
    bottom_outflow[0] = length
    bottom_outflow_ml[0] = length
    top_outflow[0] = length
    top_outflow_ml[0] = length

    bottom_outflow[1] = -height/2.
    bottom_outflow_ml[1] = -10*vorticity_thickness
    top_outflow[1] = height/2.
    top_outflow_ml[1] = 10*vorticity_thickness

    from meshmode.mesh.io import (
        generate_gmsh,
        ScriptSource
    )

    # for 2D, the line segments/surfaces need to be specified clockwise to
    # get the correct facing (right-handed) surface normals
    my_string = (f"""
        Point(1) = {{ {bottom_inflow[0]},  {bottom_inflow[1]}, 0, {size}}};
        Point(2) = {{ {bottom_inflow_ml[0]},  {bottom_inflow_ml[1]}, 0, {size}}};
        Point(3) = {{ {top_inflow_ml[0]},  {top_inflow_ml[1]}, 0, {size}}};
        Point(4) = {{ {top_inflow[0]},  {top_inflow[1]}, 0, {size}}};
        Point(5) = {{ {bottom_outflow[0]},  {bottom_outflow[1]}, 0, {size}}};
        Point(6) = {{ {bottom_outflow_ml[0]},  {bottom_outflow_ml[1]}, 0, {size}}};
        Point(7) = {{ {top_outflow_ml[0]},  {top_outflow_ml[1]}, 0, {size}}};
        Point(8) = {{ {top_outflow[0]},  {top_outflow[1]}, 0, {size}}};
        Line(1) = {{1, 2}};
        Line(2) = {{2, 3}};
        Line(3) = {{3, 4}};

        Line(4) = {{5, 6}};
        Line(5) = {{6, 7}};
        Line(6) = {{7, 8}};

        Line(7) = {{1, 5}};
        Line(8) = {{2, 6}};
        Line(9) = {{3, 7}};
        Line(10) = {{4, 8}};
        Line Loop(1) = {{1, 8, -4, -7}};
        Line Loop(2) = {{2, 9, -5, -8}};
        Line Loop(3) = {{3, 10, -6, -9}};
        Plane Surface(1) = {{1}};
        Plane Surface(2) = {{2}};
        Plane Surface(3) = {{3}};
        """)
    if dim == 2:
        my_string += ("""
            Physical Surface('fluid') = {1, 2, 3};
            Physical Curve('inflow') = {1,2,3};
            Physical Curve('outflow') = {4,5,6};
            Physical Curve('wall') = {7,10};
        """)

    if transfinite:
        if dim == 2:
            my_string += (f"""
                Transfinite Curve {{2, 5}} = {2*vorticity_thickness} / {size} + 1;
                Transfinite Curve {{8, 9}} = {length} / {size} + 1;
                "Transfinite Surface {{2}} Right;"
            """)
    else:
        if dim == 2:
            my_string += (f"""
                // Create distance field from curves, excludes cavity
                Field[1] = Distance;
                Field[1].CurvesList = {{8,9}};
                Field[1].NumPointsPerCurve = 100000;

                //Create threshold field that varies element size near boundaries
                Field[2] = Threshold;
                Field[2].InField = 1;
                Field[2].SizeMin = {size} / {layer_ratio};
                Field[2].SizeMax = {size};
                Field[2].DistMin = 0.0002;
                Field[2].DistMax = 0.005;
                Field[2].StopAtDistMax = 1;

                //  background mesh size
                Field[3] = Box;
                Field[3].XMin = 0.;
                Field[3].XMax = 1.0;
                Field[3].YMin = -1.0;
                Field[3].YMax = 1.0;
                Field[3].VIn = {size};

                //  mixing layer mesh size
                Field[4] = Box;
                Field[4].XMin = 0.;
                Field[4].XMax = 1.0;
                Field[4].YMin = -1.0;
                Field[4].YMax = 1.0;
                Field[4].VIn = {size} / {layer_ratio};

                Field[5] = Restrict;
                Field[5].SurfacesList = {{2}};
                Field[5].InField = 4;

                // take the minimum of all defined meshing fields
                Field[100] = Min;
                Field[100].FieldsList = {{2, 3, 5}};
                Background Field = 100;
            """)

    my_string += ("""
        Mesh.MeshSizeExtendFromBoundary = 0;
        Mesh.MeshSizeFromPoints = 0;
        Mesh.MeshSizeFromCurvature = 0;

        Mesh.Algorithm = 5;
        Mesh.OptimizeNetgen = 1;
        Mesh.Smoothing = 100;
    """)

    if use_quads:
        my_string += ("""
        // Convert the triangles back to quads
        Mesh.Algorithm = 6;
        Mesh.Algorithm3D = 1;
        Mesh.RecombinationAlgorithm = 2;
        Mesh.RecombineAll = 1;
        Recombine Surface {1, 2};
        """)

    #print(my_string)
    return partial(generate_gmsh, ScriptSource(my_string, "geo"),
                   force_ambient_dim=dim, dimensions=dim, target_unit="M",
                   return_tag_to_elements_map=True)

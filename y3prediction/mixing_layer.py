"""mirgecom driver initializer for the mixing layer problem."""
import numpy as np
from mirgecom.fluid import make_conserved

from y3prediction.utils import smooth_step
from functools import partial


class MixingLayer
    r"""Solution initializer for flow with a discontinuity.

    This initializer creates a physics-consistent flow solution
    given an initial thermal state (pressure, temperature) and an EOS.

    The solution varies across a planar interface defined by a tanh function
    located at disc_location for pressure, temperature, velocity, and mass fraction

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, normal_dir, disc_location, disc_location_species,
            nspecies=0,
            temperature_left, temperature_right,
            pressure_left, pressure_right,
            velocity_left=None, velocity_right=None,
            velocity_cross=None,
            species_mass_left=None, species_mass_right=None,
            convective_velocity=None, sigma=0.5,
            temp_sigma=0., vel_sigma=0., temp_wall=300., y_top=0.01, y_bottom=-0.01
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        normal_dir: numpy.ndarray
            specifies the direction (plane) the discontinuity is applied in
        disc_location: numpy.ndarray or Callable
            fixed location of discontinuity or optionally a function that
            returns the time-dependent location.
        disc_location_species: numpy.ndarray or Callable
            fixed location of the species discontinuity
        nspecies: int
            specifies the number of mixture species
        pressure_left: float
            pressure to the left of the discontinuity
        temperature_left: float
            temperature to the left of the discontinuity
        velocity_left: numpy.ndarray
            velocity (vector) to the left of the discontinuity
        species_mass_left: numpy.ndarray
            species mass fractions to the left of the discontinuity
        pressure_right: float
            pressure to the right of the discontinuity
        temperature_right: float
            temperaure to the right of the discontinuity
        velocity_right: numpy.ndarray
            velocity (vector) to the right of the discontinuity
        species_mass_right: numpy.ndarray
            species mass fractions to the right of the discontinuity
        sigma: float
           sharpness parameter
        velocity_cross: numpy.ndarray
            velocity (vector) tangent to the shock
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        """
        if velocity_left is None:
            velocity_left = np.zeros(shape=(dim,))
        if velocity_right is None:
            velocity_right = np.zeros(shape=(dim,))
        if velocity_cross is None:
            velocity_cross = np.zeros(shape=(dim,))

        if species_mass_left is None:
            species_mass_left = np.zeros(shape=(nspecies,))
        if species_mass_right is None:
            species_mass_right = np.zeros(shape=(nspecies,))

        self._nspecies = nspecies
        self._dim = dim
        self._disc_location = disc_location
        self._disc_location_species = disc_location_species
        self._sigma = sigma
        self._ul = velocity_left
        self._ur = velocity_right
        self._ut = velocity_cross
        self._uc = convective_velocity
        self._pl = pressure_left
        self._pr = pressure_right
        self._tl = temperature_left
        self._tr = temperature_right
        self._yl = species_mass_left
        self._yr = species_mass_right
        self._normal = normal_dir
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._temp_wall = temp_wall
        self._y_top = y_top
        self._y_bottom = y_bottom

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

        xpos = x_vec[0]*self._normal[0] + x_vec[1]*self._normal[1]
        ypos = -x_vec[0]*self._normal[1] + x_vec[1]*self._normal[0]

        actx = xpos.array_context
        #if isinstance(self._disc_location, Number):
        if callable(self._disc_location):
            x0 = self._disc_location(time)
        else:
            x0 = self._disc_location

        if callable(self._disc_location_species):
            x0_species = self._disc_location(time)
        else:
            x0_species = self._disc_location_species

        # get the species mass fractions first
        dist = np.dot(x0_species - x_vec, self._normal)
        xtanh = 1.0/self._sigma*dist
        weight = 0.5*(1.0 - actx.np.tanh(xtanh))
        y = self._yl + (self._yr - self._yl)*weight

        # now solve for T, P, velocity
        dist = np.dot(x0 - x_vec, self._normal)
        xtanh = 1.0/self._sigma*dist
        weight = 0.5*(1.0 - actx.np.tanh(xtanh))
        pressure = self._pl + (self._pr - self._pl)*weight
        temperature = self._tl + (self._tr - self._tl)*weight
        velocity = self._ul + (self._ur - self._ul)*weight + self._ut

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        y_top = self._y_top
        y_bottom = self._y_bottom
        sigma = self._temp_sigma
        if sigma > 0:
            wall_temperature = self._temp_wall
            smoothing_top = smooth_step(actx, -sigma*(ypos - y_top))
            smoothing_bottom = smooth_step(actx, sigma*(ypos - y_bottom))
            temperature = (wall_temperature +
                (temperature - wall_temperature)*smoothing_top*smoothing_bottom)

        # modify the velocity in the near wall region to match the
        # noslip boundaries
        sigma = self._vel_sigma
        if sigma > 0:
            smoothing_top = smooth_step(actx, -sigma*(ypos - y_top))
            smoothing_bottom = smooth_step(actx, sigma*(ypos - y_bottom))
            velocity = velocity*smoothing_top*smoothing_bottom

        mass = eos.get_density(pressure, temperature,
                               species_mass_fractions=y)

        specmass = mass * y
        mom = mass * velocity
        internal_energy = eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)

        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


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
    bottom_inflow[1] = height/2.
    bottom_inflow_ml[1] = -2*vorticity_thickness
    top_inflow[1] = -height/2.
    top_inflow_ml[1] = 2*vorticity_thickness

    # points on the outflow
    bottom_outflow[0] = length
    bottom_outflow_ml[0] = length
    top_outflow[0] = length
    top_outflow_ml[0] = length

    bottom_outflow[1] = height/2.
    bottom_outflow_ml[1] = -10*vorticity_thickness
    top_outflow[1] = -height/2.
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

        Line(7) = {{1, 8}};
        Line(8) = {{2, 5}};
        Line(9) = {{3, 6}};
        Line(10) = {{4, 5}};
        Line Loop(1) = {{1, 8, -4, -7}};
        Line Loop(2) = {{2, 9, -5, -8}};
        Line Loop(3) = {{3, 10, -6, -9}};
        Plane Surface(1) = {{1}};
        Plane Surface(2) = {{2}};
        Plane Surface(3) = {{3}};
        """)
    if dim == 2:
        my_string += ("""
            Physical Surface('fluid') = {1};
            Physical Curve('inflow') = {1,2,3};
            Physical Curve('outflow') = {4,5,6};
            Physical Curve('wall') = {4,8};
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
                Field[1].CurvesList = {{5,6}};
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

                // take the minimum of all defined meshing fields
                Field[100] = Min;
                Field[100].FieldsList = {{2, 3}};
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

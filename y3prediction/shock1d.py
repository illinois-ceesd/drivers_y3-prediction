"""mirgecom driver initializer for 1d shock."""
import numpy as np
from mirgecom.fluid import make_conserved

from y3prediction.utils import smooth_step
from functools import partial


class PlanarDiscontinuityMulti:
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
            temp_sigma=0., vel_sigma=0., temp_wall=300.
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

        xpos = x_vec[0]
        ypos = x_vec[1]

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
        y_top = 0.01
        y_bottom = -0.01
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
            velocity[0] = velocity[0]*smoothing_top*smoothing_bottom

        if self._nspecies:
            mass = eos.get_density(pressure, temperature,
                                   species_mass_fractions=y)
        else:
            mass = pressure/temperature/eos.gas_const()

        specmass = mass * y
        mom = mass * velocity
        internal_energy = eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)

        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


def get_mesh(dim, size, bl_ratio, interface_ratio, angle=0.,
             transfinite=False, use_wall=True, use_quads=False,
             use_gmsh=True):
    """Generate a grid using `gmsh`."""

    height = 0.02
    fluid_length = 0.1
    wall_length = 0.02
    bottom_inflow = np.zeros(shape=(dim,))
    top_inflow = np.zeros(shape=(dim,))
    bottom_interface = np.zeros(shape=(dim,))
    top_interface = np.zeros(shape=(dim,))
    bottom_wall = np.zeros(shape=(dim,))
    top_wall = np.zeros(shape=(dim,))

    # rotate the mesh around the bottom-left corner
    theta = angle/180.*np.pi/2.
    bottom_inflow[0] = 0.0
    bottom_inflow[1] = -0.01
    top_inflow[0] = bottom_inflow[0] - height*np.sin(theta)
    top_inflow[1] = bottom_inflow[1] + height*np.cos(theta)

    bottom_interface[0] = bottom_inflow[0] + fluid_length*np.cos(theta)
    bottom_interface[1] = bottom_inflow[1] + fluid_length*np.sin(theta)
    top_interface[0] = top_inflow[0] + fluid_length*np.cos(theta)
    top_interface[1] = top_inflow[1] + fluid_length*np.sin(theta)

    bottom_wall[0] = bottom_interface[0] + wall_length*np.cos(theta)
    bottom_wall[1] = bottom_interface[1] + wall_length*np.sin(theta)
    top_wall[0] = top_interface[0] + wall_length*np.cos(theta)
    top_wall[1] = top_interface[1] + wall_length*np.sin(theta)

    if use_gmsh:
        from meshmode.mesh.io import (
            generate_gmsh,
            ScriptSource
        )

        # for 2D, the line segments/surfaces need to be specified clockwise to
        # get the correct facing (right-handed) surface normals
        my_string = (f"""
            Point(1) = {{ {bottom_inflow[0]},  {bottom_inflow[1]}, 0, {size}}};
            Point(2) = {{ {bottom_interface[0]}, {bottom_interface[1]}, 0, {size}}};
            Point(3) = {{ {top_interface[0]}, {top_interface[1]}, 0, {size}}};
            Point(4) = {{ {top_inflow[0]},  {top_inflow[1]}, 0, {size}}};
            Point(5) = {{ {bottom_wall[0]},  {bottom_wall[1]}, 0, {size}}};
            Point(6) = {{ {top_wall[0]},  {top_wall[1]}, 0, {size}}};
            Line(1) = {{1, 2}};
            Line(2) = {{2, 3}};
            Line(3) = {{3, 4}};
            Line(4) = {{4, 1}};
            Line(5) = {{3, 6}};
            Line(6) = {{2, 5}};
            Line(7) = {{5, 6}};
            Line Loop(1) = {{-4, -3, -2, -1}};
            Line Loop(2) = {{2, 5, -7, -6}};
            Plane Surface(1) = {{1}};
            Plane Surface(2) = {{2}};
            Physical Surface('fluid') = {{1}};
            Physical Surface('wall_insert') = {{2}};
            Physical Curve('inflow') = {{4}};
            Physical Curve('outflow') = {{2}};
            Physical Curve('flow') = {{2, 4}};
            Physical Curve('isothermal_wall') = {{1,3}};
            Physical Curve('periodic_y_top') = {{3}};
            Physical Curve('periodic_y_bottom') = {{1}};
            Physical Curve('wall_interface') = {{2}};
            Physical Curve('wall_farfield') = {{5, 6, 7}};
            Physical Curve('solid_wall_top') = {{5}};
            Physical Curve('solid_wall_bottom') = {{6}};
            Physical Curve('solid_wall_end') = {{7}};
            """)

        if transfinite:
            my_string += (f"""
                Transfinite Curve {{1, 3}} = {0.1} / {size};
                Transfinite Curve {{5, 6}} = {0.02} / {size};
                Transfinite Curve {{-2, 4, 7}}={0.02}/{size} Using Bump 1/{bl_ratio};
                Transfinite Surface {{1, 2}} Right;

                Mesh.MeshSizeExtendFromBoundary = 0;
                Mesh.MeshSizeFromPoints = 0;
                Mesh.MeshSizeFromCurvature = 0;

                Mesh.Algorithm = 5;
                Mesh.OptimizeNetgen = 1;
                Mesh.Smoothing = 0;
            """)
        else:
            my_string += (f"""
                // Create distance field from curves, excludes cavity
                Field[1] = Distance;
                Field[1].CurvesList = {{1,3}};
                Field[1].NumPointsPerCurve = 100000;

                //Create threshold field that varrries element size near boundaries
                Field[2] = Threshold;
                Field[2].InField = 1;
                Field[2].SizeMin = {size} / {bl_ratio};
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

                // Create distance field from curves, excludes cavity
                Field[4] = Distance;
                Field[4].CurvesList = {{2}};
                Field[4].NumPointsPerCurve = 100000;

                //Create threshold field that varrries element size near boundaries
                Field[5] = Threshold;
                Field[5].InField = 4;
                Field[5].SizeMin = {size} / {interface_ratio};
                Field[5].SizeMax = {size};
                Field[5].DistMin = 0.0002;
                Field[5].DistMax = 0.005;
                Field[5].StopAtDistMax = 1;

                // take the minimum of all defined meshing fields
                Field[100] = Min;
                Field[100].FieldsList = {{2, 3, 5}};
                Background Field = 100;

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
                Mesh.Algorithm = 8;
                Mesh.RecombinationAlgorithm = 2;
                Mesh.RecombineAll = 1;
                Recombine Surface {1, 2};
            """)

        #print(my_string)
        return partial(generate_gmsh, ScriptSource(my_string, "geo"),
                                force_ambient_dim=2, dimensions=2, target_unit="M",
                                return_tag_to_elements_map=True)

    else:
        from meshmode.mesh.generation import generate_regular_rect_mesh

        # this only works for non-slanty meshes
        def get_meshmode_mesh(a, b, nelements_per_axis):

            if use_quads:
                from meshmode.mesh import TensorProductElementGroup
                group_cls = TensorProductElementGroup
            else:
                group_cls = None

            mesh = generate_regular_rect_mesh(
                a=a, b=b, nelements_per_axis=nelements_per_axis,
                group_cls=group_cls,
                boundary_tag_to_face={
                    "inflow": ["-x"],
                    "outflow": ["+x"],
                    "flow": ["-x", "+x"],
                    "isothermal_wall": ["-y", "+y"],
                    "periodic_y_top": ["+y"],
                    "periodic_y_bottom": ["-y"],
                    "wall_farfield": ["+x"],
                    "solid_wall_top": ["+y"],
                    "solid_wall_bottom": ["-y"],
                    "solid_wall_end": ["+x"]
                })

            mgrp = mesh.groups[0]
            x = mgrp.nodes[0, :, :]
            x_avg = np.sum(x, axis=1)/x.shape[1]
            tag_to_elements = {
                "fluid": np.where(x_avg < fluid_length)[0],
                "wall": np.where(x_avg > fluid_length)[0]}

            return mesh, tag_to_elements

        return partial(get_meshmode_mesh,
                       a=(bottom_inflow[0], bottom_inflow[1]),
                       b=(top_wall[0], top_wall[1]),
                       nelements_per_axis=(int((fluid_length + wall_length)/size),
                                           int(height/size)))

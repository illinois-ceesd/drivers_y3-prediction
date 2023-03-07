import numpy as np


def setup_flame(mechanism_name="uiuc", fuel_name="C2H4",
                temperature_unburned=300.0,
                stoich_ratio=None, equiv_ratio=1.0, ox_di_ratio=.21,
                oxidizer_name="O2", inert_name="N2",
                pressure_unburned=None):
    # {{{  Set up burned/unburned state using Cantera
    import cantera
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_name)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species
    species_names = cantera_soln.species_names

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    if stoich_ratio is None:
        if fuel_name == "C2H4":
            stoich_ratio = 3.0
        elif fuel_name == "H2":
            stoich_ratio = 0.5
        else:
            stoich_ratio = 1.0

    # Grab the array indices for the specific species, ethylene, oxygen, and nitrogen
    i_fu = cantera_soln.species_index(fuel_name)
    i_ox = cantera_soln.species_index(oxidizer_name)
    i_di = cantera_soln.species_index(inert_name)
    x = np.zeros(nspecies)

    # Set the species mole fractions according to our desired fuel/air mixture
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio

    one_atm = cantera.one_atm  # pylint: disable=no-member
    pres_unburned = one_atm if pressure_unburned is None else pressure_unburned

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({temperature_unburned}, {pres_unburned}, {x}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temperature_unburned, pres_unburned, x
    # Pull temperature, total density, mass fractions, and pressure from Cantera
    # We need total density, and mass fractions to initialize the fluid/gas state.
    y_unburned = np.zeros(nspecies)
    can_t, rho_unburned, y_unburned = cantera_soln.TDY

    # now find the conditions for the burned gas
    cantera_soln.equilibrate("TP")
    temp_burned, rho_burned, y_burned = cantera_soln.TDY
    pres_burned = cantera_soln.P

    print(f"Mechanism species names {species_names}")
    print(f"Unburned (T,P,Y) = ({temperature_unburned}, {pres_unburned}, "
          f"{y_unburned}")
    print(f"Burned (T,P,Y) = ({temp_burned}, {pres_burned}, {y_burned}")

    return {"unburned": (pres_unburned, temperature_unburned, y_unburned,
                         rho_unburned),
            "burned": (pres_burned, temp_burned, y_burned, rho_burned)}


class Flash1D:
    r"""Solution initializer for flow with a shock and/or flame.

    This initializer creates a physics-consistent flow solution
    given an initial thermal state (pressure, temperature) for the pre and post
    burned and shocked regions and an EOS.

    The solution varies across a planar shock/flame interfaces defined by a tanh
    function located at (shock,flame)_location for pressure, temperature, velocity,
    and mass fraction

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=3, nspecies=0,
            shock_normal_dir=None, flame_normal_dir=None,
            shock_location=None, flame_location=None,
            temperature_unshocked=None, temperature_shocked=None,
            temperature_unburned=None, temperature_burned=None,
            pressure_unshocked=None, pressure_shocked=None,
            pressure_unburned=None, pressure_burned=None,
            species_mass_fractions_shocked=None,
            species_mass_fractions_unshocked=None,
            species_mass_fractions_burned=None,
            species_mass_fractions_unburned=None,
            velocity_unshocked=None, velocity_shocked=None,
            velocity_cross=None,
            convective_velocity=None, sigma=0.5,
            temp_sigma=0., vel_sigma=0., temp_wall=300.
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        nspecies: int
            specifies the number of mixture species
        shock_normal_dir: numpy.ndarray
            specifies the orientation of the discontinuity
        shock_location: numpy.ndarray
            init/fixed location of discontinuity
        flame_normal_dir: numpy.ndarray
            specifies the orientation of the flame front
        flame_location: numpy.ndarray
            fixed location of the flame front
        pressure_shocked: float
            pressure in the shocked region
        pressure_unshocked: float
            pressure in the unshocked region
        temperature_shocked: float
            temperature in the shocked region
        temperature_unshocked: float
            temperature in the unshocked region
        velocity_shocked: numpy.ndarray
            velocity (vector) in the shocked region
        velocity_unshocked: numpy.ndarray
            velocity (vector) in the unshocked region
        species_mass_fractions_shocked: numpy.ndarray
            species mass fractions in the shocked region
        species_mass_fractions_unshocked: numpy.ndarray
            species mass fractions in the unshocked region
        pressure_burned: float
            pressure in the shocked region
        pressure_unburned: float
            pressure in the unshocked region
        temperature_burned: float
            temperature in the shocked region
        temperature_unburned: float
            temperature in the unshocked region
        species_mass_fractions_burned: numpy.ndarray
            species mass fractions in the shocked region
        species_mass_fractions_unburned: numpy.ndarray
            species mass fractions in the unshocked region
        sigma: float
           sharpness parameter for shock and flame fronts
        velocity_cross: numpy.ndarray
            velocity (vector) tangent to the shock
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        """
        self.do_shock = shock_location is not None
        self.do_flame = flame_location is not None
        if not self.do_flame and not self.do_shock:
            raise ValueError("Invalid arguments for flame and shock init.")

        if self.do_flame:
            if nspecies == 0:
                raise ValueError("nspecies must be set for flame initialization.")
            if any([pressure_burned, pressure_unburned, temperature_burned,
                   temperature_unburned, species_mass_fractions_burned,
                    species_mass_fractions_unburned, flame_normal_dir]) is None:
                raise ValueError("burned/unburned states underspecified for"
                                 " flame init.")

        if self.do_shock:
            if any([pressure_shocked, pressure_unshocked, temperature_shocked,
                    temperature_unshocked, shock_normal_dir]) is None:
                raise ValueError("shocked/unshocked states underspecified for "
                                 "flame init.")
            if not self.do_flame and nspecies > 0:
                if any([species_mass_fractions_shocked,
                        species_mass_fractions_unshocked]) is None:
                    raise ValueError("y shocked/unshocked must be given for"
                                     " multispecies shock.")

        if velocity_shocked is None:
            velocity_shocked = np.zeros(shape=(dim,))
        if velocity_unshocked is None:
            velocity_unshocked = np.zeros(shape=(dim,))
        if velocity_cross is None:
            velocity_cross = np.zeros(shape=(dim,))
        if convective_velocity is None:
            convective_velocity = np.zeros(shape=(dim,))

        self._nspecies = nspecies
        self._dim = dim
        self._sigma = sigma
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._temp_wall = temp_wall

        self._shock_loc = shock_location
        self._flame_loc = flame_location
        self._shock_normal = shock_normal_dir
        self._flame_normal = flame_normal_dir

        self._u_shocked = velocity_shocked
        self._u_unshocked = velocity_unshocked
        self._ut = velocity_cross
        self._uc = convective_velocity
        self._p_shocked = pressure_shocked
        self._p_unshocked = pressure_unshocked
        self._t_shocked = temperature_shocked
        self._t_unshocked = temperature_unshocked

        self._t_burned = temperature_burned
        self._t_unburned = temperature_unburned
        self._p_burned = pressure_burned
        self._p_unburned = pressure_unburned
        self._y_burned = species_mass_fractions_burned
        self._y_unburned = species_mass_fractions_unburned

    def __call__(self, x_vec, eos, *, time=0.0):
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
        ones = 0*xpos + 1.
        # if self._dim == 3:
        #    zpos = x_vec[2]

        actx = xpos.array_context

        # Set up flame states
        flame_rd = 1.*ones  # unburned everywhere
        if self.do_flame:
            flame_x0 = self._flame_loc
            flame_rd = np.dot(x_vec - flame_x0, self._flame_normal) / self._sigma
            dp = self._p_unburned - self._p_burned
            dy = self._y_unburned - self._y_burned
            dt = self._t_unburned - self._t_burned
            weight = 0.5*(1.0 + actx.np.tanh(flame_rd))
            y_flame = self._y_burned + dy*weight
            p_flame = self._p_burned + dp*weight
            t_flame = self._t_burned + dt*weight
            pressure = p_flame
            temperature = t_flame
            y = y_flame

        # Set up shock states
        if self.do_shock:
            shock_x0 = self._shock_loc
            shock_rd = np.dot(x_vec - shock_x0, self._shock_normal) / self._sigma
            dp = self._p_unshocked - self._p_shocked
            dt = self._t_unshocked - self._t_shocked
            weight = 0.5*(1.0 + actx.np.tanh(shock_rd))
            p_shock = self._p_shocked + dp*weight
            t_shock = self._t_shocked + dt*weight
            pressure = p_shock
            temperature = t_shock
            if self._nspecies > 0:
                dy = self._y_unshocked - self._y_shocked
                y = self._y_shocked + dy*weight
            else:
                y = None

        velocity = \
            self._u_shocked + (self._u_unshocked-self._u_shocked)*weight + self._ut

        if self.do_shock and self.do_flame:
            # Use Y from flame
            # Use pressure from:
            #  - shock (if shocked region)
            #  - flame (if unshocked region)
            # Use max temperature
            y = y_flame
            pressure = actx.np.where(actx.np.less(shock_rd, 0), p_shock, p_flame)
            temperature = actx.np.max(t_shock, t_flame)

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        y_top = 0.01
        y_bottom = -0.01
        if self._temp_sigma > 0:
            sigma = self._temp_sigma
            wall_temperature = self._temp_wall
            smoothing_top = smooth_step(actx, -sigma*(ypos - y_top))
            smoothing_bottom = smooth_step(actx, sigma*(ypos - y_bottom))
            temperature = \
                (wall_temperature
                 + (temperature - wall_temperature)*smoothing_top*smoothing_bottom)

        # modify the velocity in the near wall region to match the
        # noslip boundaries
        sigma = self._vel_sigma
        smoothing_top = smooth_step(actx, -sigma*(ypos - y_top))
        smoothing_bottom = smooth_step(actx, sigma*(ypos - y_bottom))
        velocity[0] = velocity[0]*smoothing_top*smoothing_bottom

        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
        specmass = mass * y if y is not None else None
        mom = mass * velocity
        internal_energy = eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)

        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        from mirgecom.fluid import make_conserved
        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


def get_flash_mesh(dim, size, bl_ratio, interface_ratio, angle=0.,
                   transfinite=False, use_gmsh=False,
                   left_boundary_loc=-1, right_boundary_loc=1,
                   bottom_boundary_loc=-1, top_boundary_loc=1):
    """Generate a grid using `gmsh`.

    """

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
        my_string = \
            (f"""
            Point(1) = {{ {bottom_inflow[0]},  {bottom_inflow[1]}, 0, {size}}};
            Point(2) = {{ {bottom_interface[0]}, {bottom_interface[1]},  0, {size}}};
            Point(3) = {{ {top_interface[0]}, {top_interface[1]},    0, {size}}};
            Point(4) = {{ {top_inflow[0]},  {top_inflow[1]},    0, {size}}};
            Point(5) = {{ {bottom_wall[0]},  {bottom_wall[1]},    0, {size}}};
            Point(6) = {{ {top_wall[0]},  {top_wall[1]},    0, {size}}};
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
            Physical Surface('wall') = {{2}};
            Physical Curve('fluid_inflow') = {{4}};
            Physical Curve('fluid_wall') = {{1,3}};
            Physical Curve('fluid_wall_top') = {{3}};
            Physical Curve('fluid_wall_bottom') = {{1}};
            Physical Curve('interface') = {{2}};
            Physical Curve('wall_farfield') = {{5, 6, 7}};
            Physical Curve('solid_wall_top') = {{5}};
            Physical Curve('solid_wall_bottom') = {{6}};
            Physical Curve('solid_wall_end') = {{7}};
            """)

        trans_string = \
            f"""
            Transfinite Curve {{1, 3}} = {0.1} / {size};
            Transfinite Curve {{5, 6}} = {0.02} / {size};
            Transfinite Curve {{-2, 4, 7}} = {0.02} / {size} Using Bump 1/{bl_ratio};
            Transfinite Surface {{1, 2}} Right;

            Mesh.MeshSizeExtendFromBoundary = 0;
            Mesh.MeshSizeFromPoints = 0;
            Mesh.MeshSizeFromCurvature = 0;

            Mesh.Algorithm = 5;
            Mesh.OptimizeNetgen = 1;
            Mesh.Smoothing = 0;
            """
        if transfinite:
            my_string = my_string + trans_string
        else:
            my_string = \
                my_string
            + (f"""
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

        # print(my_string)
        from functools import partial
        generate_mesh = partial(generate_gmsh, ScriptSource(my_string, "geo"),
                                force_ambient_dim=2, dimensions=2, target_unit="M",
                                return_tag_to_elements_map=True)
    else:
        char_len_x = 0.002
        char_len_y = 0.001
        box_ll = (left_boundary_loc, bottom_boundary_loc)
        box_ur = (right_boundary_loc, top_boundary_loc)
        num_elements = (int((box_ur[0]-box_ll[0])/char_len_x),
                            int((box_ur[1]-box_ll[1])/char_len_y))

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh, a=box_ll, b=box_ur,
                                n=num_elements, boundary_tag_to_face={
                                    "Inflow": ["-x"],
                                    "Outflow": ["+x"],
                                    "Wall": ["+y", "-y"]
                                })

    return generate_mesh


def get_mesh_data(dim, mesh_angle, mesh_size, bl_ratio, interface_ratio,
                  transfinite, use_gmsh=True, periodic=False):

    # from meshmode.mesh.io import read_gmsh
    # mesh, tag_to_elements = read_gmsh(
    # mesh_filename, force_ambient_dim=dim,
    # return_tag_to_elements_map=True)
    mesh, tag_to_elements = get_flash_mesh(dim=dim, angle=mesh_angle,
                                           use_gmsh=True, size=mesh_size,
                                           bl_ratio=bl_ratio,
                                           interface_ratio=interface_ratio,
                                           transfinite=transfinite)()

    volume_to_tags = {
        "fluid": ["fluid"],
        "wall": ["solid"]}

    # apply periodicity
    if periodic:
        from meshmode.mesh.processing import (
            glue_mesh_boundaries, BoundaryPairMapping)

        from meshmode import AffineMap
        bdry_pair_mappings_and_tols = []
        offset = [0., 0.02]
        bdry_pair_mappings_and_tols.append((
            BoundaryPairMapping(
                "fluid_wall_bottom",
                "fluid_wall_top",
                AffineMap(offset=offset)),
            1e-12))

        bdry_pair_mappings_and_tols.append((
            BoundaryPairMapping(
                "solid_wall_bottom",
                "solid_wall_top",
                AffineMap(offset=offset)),
            1e-12))

        mesh = glue_mesh_boundaries(mesh, bdry_pair_mappings_and_tols)

    return mesh, tag_to_elements, volume_to_tags


class InitSponge:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, x0, thickness, amplitude):
        r"""Initialize the sponge parameters.

        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """

        self._x0 = x0
        self._thickness = thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        x0 = zeros + self._x0

        return self._amplitude * actx.np.where(
            actx.np.greater(xpos, x0),
            (zeros + ((xpos - self._x0)/self._thickness)
             * ((xpos - self._x0)/self._thickness)),
            zeros + 0.0
        )


def getIsentropicPressure(mach, P0, gamma):
    pressure = (1. + (gamma - 1.)*0.5*mach**2)
    pressure = P0*pressure**(-gamma / (gamma - 1.))
    return pressure


def getIsentropicTemperature(mach, T0, gamma):
    temperature = (1. + (gamma - 1.)*0.5*mach**2)
    temperature = T0/temperature
    return temperature


def smooth_step(actx, x, epsilon=1e-12):
    # return actx.np.tanh(x)
    # return actx.np.where(
    #     actx.np.greater(x, 0),
    #     actx.np.tanh(x)**3,
    #     0*x)
    return (
        actx.np.greater(x, 0) * actx.np.less(x, 1) * (1 - actx.np.cos(np.pi*x))/2
        + actx.np.greater(x, 1))


class InitShock:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, nspecies=0,
            P0, T0, temp_wall, temp_sigma, vel_sigma, gamma_guess,
            mass_frac=None
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        P0: float
            stagnation pressure
        T0: float
            stagnation temperature
        gamma_guess: float
            guesstimate for gamma
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        """

        if mass_frac is None:
            if nspecies > 0:
                mass_frac = np.zeros(shape=(nspecies,))

        self._dim = dim
        self._nspecies = nspecies
        self._P0 = P0
        self._T0 = T0
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._gamma_guess = gamma_guess
        self._mass_frac = mass_frac

    def __call__(self, dcoll, x_vec, eos, *, time=0.0):
        """Create the solution state at locations *x_vec*.

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
        if self._dim == 3:
            zpos = x_vec[2]
        ytop = 0*x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        ones = zeros + 1.0

        mach = zeros
        ytop = zeros
        ybottom = zeros
        # theta = zeros
        gamma = self._gamma_guess

        pressure = getIsentropicPressure(
            mach=mach,
            P0=self._P0,
            gamma=gamma
        )
        temperature = getIsentropicTemperature(
            mach=mach,
            T0=self._T0,
            gamma=gamma
        )

        # save the unsmoothed temerature, so we can use it with the injector init
        # unsmoothed_temperature = temperature

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        sigma = self._temp_sigma
        # wall_temperature = self._temp_wall
        smoothing_top = smooth_step(actx, -sigma*(ypos-ytop))
        smoothing_bottom = smooth_step(
            actx, sigma*actx.np.abs(ypos-ybottom))
        smoothing_fore = ones
        smoothing_aft = ones
        z0 = 0.
        z1 = 0.035
        if self._dim == 3:
            smoothing_fore = smooth_step(actx, sigma*(zpos-z0))
            smoothing_aft = smooth_step(actx, -sigma*(zpos-z1))

        # smooth_temperature = (wall_temperature +
        #    (temperature - wall_temperature)*smoothing_top*smoothing_bottom *
        #                                     smoothing_fore*smoothing_aft)

        y = ones*self._mass_frac
        mass = eos.get_density(pressure=pressure, temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        velocity = np.zeros(self._dim, dtype=object)
        mom = mass*velocity

        from mirgecom.fluid import make_conserved
        cv = make_conserved(dim=self._dim, mass=mass, momentum=mom, energy=energy,
                            species_mass=mass*y)
        velocity[0] = mach*eos.sound_speed(cv, temperature)

        # modify the velocity in the near-wall region to have a smooth profile
        # this approximates the BL velocity profile
        sigma = self._vel_sigma
        smoothing_top = smooth_step(actx, -sigma*(ypos-ytop))
        smoothing_bottom = smooth_step(actx, sigma*(actx.np.abs(ypos-ybottom)))
        smoothing_fore = ones
        smoothing_aft = ones
        if self._dim == 3:
            smoothing_fore = smooth_step(actx, sigma*(zpos-z0))
            smoothing_aft = smooth_step(actx, -sigma*(zpos-z1))
        velocity[0] = (velocity[0]*smoothing_top*smoothing_bottom
                       * smoothing_fore*smoothing_aft)

        mom = mass*velocity
        energy = (energy + np.dot(mom, mom)/(2.0*mass))
        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )


def get_boundaries(dcoll, actx, dd_vol_fluid, dd_vol_wall, noslip, adiabatic,
                   periodic, temp_wall, gas_model, quadrature_tag,
                   target_fluid_state):

    from mirgecom.boundary import (
        PrescribedFluidBoundary,
        IsothermalWallBoundary,
        # SymmetryBoundary,
        AdiabaticSlipBoundary,
        AdiabaticNoslipWallBoundary,
        DummyBoundary
    )
    from mirgecom.diffusion import (
        DirichletDiffusionBoundary
    )
    from mirgecom.simutil import force_evaluation
    from mirgecom.gas_model import project_fluid_state

    if noslip:
        if adiabatic:
            fluid_wall = AdiabaticNoslipWallBoundary()
        else:
            fluid_wall = IsothermalWallBoundary(temp_wall)

    else:
        # new impl, following Mengaldo with modifications for slip vs no slip
        # set the flux directly, instead using viscous numerical flux func
        # fluid_wall = AdiabaticSlipWallBoundary2()

        # implementation from mirgecom
        # should be same as AdiabaticSlipBoundary2
        fluid_wall = AdiabaticSlipBoundary()

        # new impl, following Mengaldo with modifications for slip vs no slip
        # local version
        # fluid_wall = AdiabaticSlipWallBoundary()

        # Tulio's symmetry boundary
        # fluid_wall = SymmetryBoundary(dim=dim)

    wall_farfield = DirichletDiffusionBoundary(temp_wall)

    # use dummy boundaries to setup the smoothness state for the target
    target_boundaries = {
        dd_vol_fluid.trace("fluid_inflow").domain_tag: DummyBoundary(),
        # dd_vol_fluid.trace("fluid_wall").domain_tag: IsothermalWallBoundary()
        dd_vol_fluid.trace("fluid_wall").domain_tag: fluid_wall
    }

    def get_target_state_on_boundary(btag):
        return project_fluid_state(
            dcoll, dd_vol_fluid,
            dd_vol_fluid.trace(btag).with_discr_tag(quadrature_tag),
            target_fluid_state, gas_model
        )

    flow_ref_state = \
        get_target_state_on_boundary("fluid_inflow")

    flow_ref_state = force_evaluation(actx, flow_ref_state)

    def _target_flow_state_func(**kwargs):
        return flow_ref_state

    flow_boundary = PrescribedFluidBoundary(
        boundary_state_func=_target_flow_state_func)

    if periodic:
        fluid_boundaries = {
            dd_vol_fluid.trace("fluid_inflow").domain_tag: flow_boundary,
        }

        wall_boundaries = {
            dd_vol_wall.trace("solid_wall_end").domain_tag: wall_farfield
        }
    else:
        fluid_boundaries = {
            dd_vol_fluid.trace("fluid_inflow").domain_tag: flow_boundary,
            dd_vol_fluid.trace("fluid_wall").domain_tag: fluid_wall
        }

        wall_boundaries = {
            dd_vol_wall.trace("wall_farfield").domain_tag: wall_farfield
        }
    return fluid_boundaries, wall_boundaries, target_boundaries

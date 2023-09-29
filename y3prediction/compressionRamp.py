"""mirgecom driver initializer for the Y2 prediction."""
import numpy as np
from mirgecom.fluid import make_conserved


class InitCompressionRamp:
    r"""Solution initializer for flow with a discontinuity in the compression ramp case.

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
        
        #initialize each position coordinate.
        xpos = x_vec[0]
        ypos = x_vec[1]
        if self._dim == 3:
            zpos = x_vec[2]

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

        ##Smooth step helper function to adjust the velocity of near-wall region.
        def smooth_step(actx, x, epsilon=1e-12):
            return (
                actx.np.greater(x, 0) * actx.np.less(x, 1) * (1 - actx.np.cos(np.pi*x))/2
                + actx.np.greater(x, 1))
        
        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        y_top = .512
        y_bottom = 0.
        if self._temp_sigma > 0:
            sigma = self._temp_sigma
            wall_temperature = self._temp_wall
            smoothing_top = smooth_step(actx, -sigma*(ypos - y_top))
            smoothing_bottom = smooth_step(actx, sigma*(ypos - y_bottom))
            temperature = (wall_temperature +
                           (temperature -
                            wall_temperature)*smoothing_top*smoothing_bottom)
    
        # modify the velocity in the near wall region to match the
        # noslip boundaries
        sigma = self._vel_sigma
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

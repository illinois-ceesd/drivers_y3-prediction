"""mirgecom driver initializer for the Y2 prediction."""
import numpy as np
from mirgecom.fluid import make_conserved

from y3prediction.utils import (
    smooth_step
)


class InitUnstartRamp:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer initializes a pressure discontinuity in the inlet section
    based on an optionally time-dependent stagnation pressure and temperature
    and isentropic flow relations.

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: add_injection
    """

    def __init__(self, *, dim=2, nspecies=0, disc_sigma,
                 pressure_bulk, temperature_bulk, velocity_bulk,
                 mass_frac_bulk,
                 pressure_inlet, temperature_inlet, velocity_inlet,
                 mass_frac_inlet,
                 pressure_outlet, temperature_outlet, velocity_outlet,
                 mass_frac_outlet,
                 temp_wall, temp_sigma, vel_sigma,
                 inlet_pressure_func=None, outlet_pressure_func=None):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        """
        self._dim = dim
        self._nspecies = nspecies
        self._disc_sigma = disc_sigma
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma

        # bulk fluid conditions (background)
        self._temp_bulk = temperature_bulk
        self._pres_bulk = pressure_bulk
        if velocity_bulk is None:
            velocity_bulk = np.zeros(shape=(dim,))
        self._vel_bulk = velocity_bulk

        if mass_frac_bulk is None:
            mass_frac_bulk = np.zeros(shape=(nspecies,))
        self._y_bulk = mass_frac_bulk

        # inlet fluid conditions
        self._temp_inlet = temperature_inlet
        self._pres_inlet = pressure_inlet
        if velocity_inlet is None:
            velocity_inlet = np.zeros(shape=(dim,))
        self._vel_inlet = velocity_inlet

        if mass_frac_inlet is None:
            mass_frac_inlet = np.zeros(shape=(nspecies,))
        self._y_inlet = mass_frac_inlet

        self._inlet_p_fun = inlet_pressure_func

        # injection fluid conditions
        self._temp_outlet = temperature_outlet
        self._pres_outlet = pressure_outlet
        if velocity_outlet is None:
            velocity_outlet = np.zeros(shape=(dim,))
        self._vel_outlet = velocity_outlet

        if mass_frac_outlet is None:
            mass_frac_outlet = np.zeros(shape=(nspecies,))
        self._y_outlet = mass_frac_outlet

        self._outlet_p_fun = outlet_pressure_func

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
            Time at which solution is desired. The pressure and gamma are
            (optionally) dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        # initialize the entire flowfield to a uniform state
        actx = x_vec[0].array_context
        zeros = actx.np.zeros_like(x_vec[0])
        mass = eos.get_density(self._pres_bulk, self._temp_bulk,
                               self._y_bulk) + zeros
        velocity = self._vel_bulk
        mom = mass*velocity
        energy = mass*(eos.get_internal_energy(self._temp_bulk, self._y_bulk)
                      + 0.5*np.dot(velocity, velocity))

        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*self._y_bulk
        )

    def inlet_smoothing_func(self, x_vec, sigma):
        actx = x_vec[0].array_context

        if self._dim == 2:
            x0 = -0.013
            x1 = 0.013
            smth_bottom = smooth_step(actx, sigma*(x_vec[0] - x0))
            smth_top = smooth_step(actx, -sigma*(x_vec[0] - x1))
            return smth_bottom*smth_top
        else:
            r1 = 0.013
            radius = actx.np.sqrt((x_vec[1])**2 + (x_vec[2])**2)
            smth_radial = smooth_step(actx, -sigma*(radius - r1))
            return smth_radial

    def outlet_smoothing_func(self, x_vec, sigma):
        actx = x_vec[0].array_context

        if self._dim == 2:
            x0 = -.2
            x1 = .2
            smth_bottom = smooth_step(actx, sigma*(x_vec[0] - x0))
            smth_top = smooth_step(actx, -sigma*(x_vec[0] - x1))
            return smth_bottom*smth_top
        else:
            r1 = 0.2
            radius = actx.np.sqrt((x_vec[1])**2 + (x_vec[2])**2)
            smth_radial = smooth_step(actx, -sigma*(radius - r1))
            return smth_radial

    def add_inlet(self, cv, pressure, temperature, x_vec, eos, *, time=0.0):
        """Create the solution state at locations *x_vec*.

        Parameters
        ----------
        fluid_state: mirgecom.gas_modle.FluidState
            Current fluid state
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """
        actx = x_vec[0].array_context

        # get the current mesh conditions
        mass = cv.mass
        energy = cv.energy
        velocity = cv.velocity
        y = cv.species_mass_fractions

        if self._inlet_p_fun is not None:
            pres_inlet = self._inlet_p_fun(time)
        else:
            pres_inlet = self._pres_inlet

        # initial discontinuity location
        if self._dim == 2:
            y0 = -0.325
            dist = y0 - x_vec[1]
        else:
            x0 = -0.325
            dist = x0 - x_vec[0]

        # now solve for T, P, velocity
        xtanh = self._disc_sigma*dist
        weight = 0.5*(1.0 - actx.np.tanh(xtanh))
        pressure = pres_inlet + (pressure - pres_inlet)*weight
        temperature = self._temp_inlet + (temperature - self._temp_inlet)*weight
        velocity = self._vel_inlet + (velocity - self._vel_inlet)*weight
        y = self._y_inlet + (y - self._y_inlet)*weight

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        sigma = self._temp_sigma
        if sigma > 0:
            wall_temperature = self._temp_wall
            sfunc = self.inlet_smoothing_func(x_vec, sigma)
            temperature = (wall_temperature +
                (temperature - wall_temperature)*sfunc)

        # modify the velocity in the near wall region to match the
        # noslip boundaries
        sigma = self._vel_sigma
        if sigma > 0:
            sfunc = self.inlet_smoothing_func(x_vec, sigma)
            velocity = velocity*sfunc

        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
        mom = mass*velocity
        internal_energy = eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)
        kinetic_energy = 0.5*np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )

    def add_outlet(self, cv, pressure, temperature, x_vec, eos, *, time=0.0):
        """Create the solution state at locations *x_vec*.

        Parameters
        ----------
        fluid_state: mirgecom.gas_modle.FluidState
            Current fluid state
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """
        actx = x_vec[0].array_context

        # get the current mesh conditions
        mass = cv.mass
        energy = cv.energy
        velocity = cv.velocity
        y = cv.species_mass_fractions

        if self._outlet_p_fun is not None:
            pres_outlet = self._outlet_p_fun(time)
        else:
            pres_outlet = self._pres_outlet

        # initial discontinuity location
        if self._dim == 2:
            y0 = 0.825
            dist = x_vec[1] - y0
        else:
            x0 = 1.1
            dist = x_vec[0] - x0

        # now solve for T, P, velocity
        xtanh = 50*dist
        weight = 0.5*(1.0 - actx.np.tanh(xtanh))
        pressure = pres_outlet + (pressure - pres_outlet)*weight
        temperature = self._temp_outlet + (temperature - self._temp_outlet)*weight
        velocity = self._vel_outlet + (velocity - self._vel_outlet)*weight
        y = self._y_outlet + (y - self._y_outlet)*weight

        # modify the temperature in the near wall region to match the
        # isothermal boundaries

        sigma = self._temp_sigma
        if sigma > 0:
            wall_temperature = self._temp_wall
            sfunc = self.outlet_smoothing_func(x_vec, sigma)
            temperature = (wall_temperature +
                (temperature - wall_temperature)*sfunc)

        # modify the velocity in the near wall region to match the
        # noslip boundaries
        sigma = self._vel_sigma
        if sigma > 0:
            sfunc = self.outlet_smoothing_func(x_vec, sigma)
            velocity = velocity*sfunc

        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
        mom = mass*velocity
        internal_energy = eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)

        kinetic_energy = 0.5*np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)
        mom = mass*velocity
        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )

"""mirgecom driver initializer for the Y2 prediction."""
import numpy as np
from pytools.obj_array import make_obj_array
from mirgecom.fluid import make_conserved
from mirgecom.gas_model import make_fluid_state


def getIsentropicPressure(mach, P0, gamma):
    pressure = (1. + (gamma - 1.)*0.5*mach**2)
    pressure = P0*pressure**(-gamma / (gamma - 1.))
    return pressure


def getIsentropicTemperature(mach, T0, gamma):
    temperature = (1. + (gamma - 1.)*0.5*mach**2)
    temperature = T0/temperature
    return temperature


def getMachFromAreaRatio(area_ratio, gamma, mach_guess=0.01):
    error = 1.0e-8
    nextError = 1.0e8
    g = gamma
    M0 = mach_guess
    while nextError > error:
        R = (((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))/M0
            - area_ratio)
        dRdM = (2*((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               / (2*g - 2)*(g - 1)/(2/(g + 1) + ((g - 1)/(g + 1)*M0*M0)) -
               ((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               * M0**(-2))
        M1 = M0 - R/dRdM
        nextError = abs(R)
        M0 = M1

    return M1


def get_y_from_x(x, data):
    """
    Return the linearly interpolated the value of y
    from the value in data(x,y) at x
    """

    if x <= data[0][0]:
        y = data[0][1]
    elif x >= data[-1][0]:
        y = data[-1][1]
    else:
        ileft = 0
        iright = data.shape[0]-1

        # find the bracketing points, simple subdivision search
        while iright - ileft > 1:
            ind = int(ileft+(iright - ileft)/2)
            if x < data[ind][0]:
                iright = ind
            else:
                ileft = ind

        leftx = data[ileft][0]
        rightx = data[iright][0]
        lefty = data[ileft][1]
        righty = data[iright][1]

        dx = rightx - leftx
        dy = righty - lefty
        y = lefty + (x - leftx)*dy/dx
    return y


def get_theta_from_data(data):
    """
    Calculate theta = arctan(dy/dx)
    Where data[][0] = x and data[][1] = y
    """

    theta = data.copy()
    for index in range(1, theta.shape[0]-1):
        #print(f"index {index}")
        theta[index][1] = np.arctan((data[index+1][1]-data[index-1][1]) /
                          (data[index+1][0]-data[index-1][0]))
    theta[0][1] = np.arctan(data[1][1]-data[0][1])/(data[1][0]-data[0][0])
    theta[-1][1] = np.arctan(data[-1][1]-data[-2][1])/(data[-1][0]-data[-2][0])
    return (theta)


def smooth_step(actx, x, epsilon=1e-12):
    # return actx.np.tanh(x)
    # return actx.np.where(
    #     actx.np.greater(x, 0),
    #     actx.np.tanh(x)**3,
    #     0*x)
    return (
        actx.np.greater(x, 0) * actx.np.less(x, 1) * (1 - actx.np.cos(np.pi*x))/2
        + actx.np.greater(x, 1))


class HeatSource:

    r"""Deposit energy from an ignition source."

    Internal energy is deposited as a gaussian of the form:

    .. math::

        e &= e + e_{a}\exp^{(1-r^{2})}\\

    Density if modified to keep pressure constant, according to the eos

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, dim, center=None, width=1.0,
                 amplitude=0., amplitude_func=None):
        r"""Initialize the spark parameters.

        Parameters
        ----------
        center: numpy.ndarray
            center of source
        amplitude: float
            source strength modifier
        amplitude_fun: function
            variation of amplitude with time
        """
        if center is None:
            center = np.zeros(shape=(dim,))
        self._center = center
        self._dim = dim
        self._amplitude = amplitude
        self._width = width
        self._amplitude_func = amplitude_func

    def __call__(self, x_vec, state, eos, time, **kwargs):
        """
        Create the energy deposition at time *t* and location *x_vec*.

        the source at time *t* is created by evaluting the gaussian
        with time-dependent amplitude at *t*.

        If modify density is true, only adjust the temperature. Pressure
        is maintained by adjusting the density.

        Parameters
        ----------
        cv: :class:`mirgecom.fluid.ConservedVars`
            Fluid conserved quantities
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        """
        t = time
        if self._amplitude_func is not None:
            amplitude = self._amplitude*self._amplitude_func(t)
        else:
            amplitude = self._amplitude

        loc = self._center

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))
        expterm = amplitude * actx.np.exp(-(r**2)/(2*self._width*self._width))

        # elevate the local temperature
        # if it's below some threshold
        #temperature = state.temperature + expterm
        temp_max = 10000.0
        temperature = actx.np.where(
            actx.np.greater(state.temperature, temp_max),
            state.temperature,
            state.temperature + expterm)
        pressure = state.pressure
        y = state.species_mass_fractions

        # density of this new state
        new_mass = eos.get_density(pressure=pressure, temperature=temperature,
                                   species_mass_fractions=y)

        # change the density so the pressure stays constant
        mass_source = new_mass - state.mass_density

        # keep the velocity constant
        momentum_source = state.velocity*mass_source

        # keep the mass fractions constant
        species_mass_source = state.species_mass_fractions*mass_source

        # the source term that keeps the energy constant having changed the mass
        energy_source = 0.5*np.dot(state.velocity, state.velocity)*mass_source

        return make_conserved(dim=self._dim, mass=mass_source,
                              energy=energy_source,
                              momentum=momentum_source,
                              species_mass=species_mass_source)


class SparkSource:
    r"""Energy deposition from a ignition source"

    Internal energy is deposited as a gaussian  of the form:

    .. math::

        e &= e + e_{a}\exp^{(1-r^{2})}\\

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, dim, center=None, width=1.0,
                 amplitude=0., amplitude_func=None):
        r"""Initialize the spark parameters.

        Parameters
        ----------
        center: numpy.ndarray
            center of source
        amplitude: float
            source strength modifier
        amplitude_fun: function
            variation of amplitude with time
        """

        if center is None:
            center = np.zeros(shape=(dim,))
        self._center = center
        self._dim = dim
        self._amplitude = amplitude
        self._width = width
        self._amplitude_func = amplitude_func

    def __call__(self, x_vec, cv, time, **kwargs):
        """
        Create the energy deposition at time *t* and location *x_vec*.

        the source at time *t* is created by evaluting the gaussian
        with time-dependent amplitude at *t*.

        Parameters
        ----------
        cv: :class:`mirgecom.gas_model.FluidState`
            Fluid state object with the conserved and thermal state.
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        """

        t = time
        if self._amplitude_func is not None:
            amplitude = self._amplitude*self._amplitude_func(t)
        else:
            amplitude = self._amplitude

        #print(f"{time=} {amplitude=}")

        loc = self._center

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))
        expterm = amplitude * actx.np.exp(-(r**2)/(2*self._width*self._width))

        mass = 0*cv.mass
        momentum = 0*cv.momentum
        species_mass = 0*cv.species_mass

        energy = cv.energy + cv.mass*expterm

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=momentum, species_mass=species_mass)


class StateSource:
    r"""State variable deposition from a source"

    Density, momentum, energy, and species mass fraction
    are deposited as a gaussian  of the form:

    .. math::

        e &= e + e_{a}\exp^{(1-r^{2})}\\

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, dim, nspecies,
                 center=None, width=1.0,
                 mass_amplitude,
                 mom_amplitude,
                 energy_amplitude,
                 y_amplitude,
                 amplitude_func=None):
        r"""Initialize the source parameters.

        Parameters
        ----------
        center: numpy.ndarray
            center of source
        amplitude: float
            source strength modifier
        amplitude_fun: function
            variation of amplitude with time
        """

        if center is None:
            center = np.zeros(shape=(dim,))
        self._center = center
        self._dim = dim
        self._nspecies = nspecies
        self._mass_amplitude = mass_amplitude
        self._mom_amplitude = mom_amplitude
        self._energy_amplitude = energy_amplitude
        self._y_amplitude = y_amplitude
        self._width = width
        self._amplitude_func = amplitude_func

    def __call__(self, x_vec, cv, time, **kwargs):
        """
        Create the energy deposition at time *t* and location *x_vec*.

        the source at time *t* is created by evaluting the gaussian
        with time-dependent amplitude at *t*.

        Parameters
        ----------
        cv: :class:`mirgecom.gas_model.FluidState`
            Fluid state object with the conserved and thermal state.
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        """

        t = time
        if self._amplitude_func is not None:
            time_amplitude = self._amplitude_func(t)
        else:
            time_amplitude = 1.0

        #print(f"{time=} {amplitude=}")

        loc = self._center

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))
        expterm = time_amplitude*actx.np.exp(-(r**2)/(2*self._width*self._width))

        mass = actx.np.zeros_like(cv.mass) + self._mass_amplitude*expterm
        momentum = actx.np.zeros_like(cv.momentum)
        for i in range(self._dim):
            momentum[i] = self._mom_amplitude[i]*expterm

        species_mass = actx.np.zeros_like(cv.species_mass)
        for i in range(self._nspecies):
            species_mass[i] = mass*self._y_amplitude[i]

        kinetic_energy = actx.np.where(
            actx.np.greater(mass, 0.), 0.5*np.dot(momentum, momentum)/mass, 0.)

        energy = actx.np.zeros_like(cv.energy) + \
            self._energy_amplitude*expterm + kinetic_energy

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=momentum, species_mass=species_mass)


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
    def __init__(self, *, x0, thickness, amplitude, direction=1.,
                 xmin=-1000., xmax=1000., ymin=-1000., ymax=1000.,
                 zmin=-1000., zmax=1000.):
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
        self._direction = direction
        self._xmin = xmin
        self._xmax = xmax
        self._ymax = ymax
        self._ymin = ymin
        self._zmin = zmin
        self._zmax = zmax

    def __call__(self, sponge_field, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        x_vec: numpy.ndarray
            Current sponge field
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zeros = actx.np.zeros_like(xpos)
        x0 = zeros + self._x0

        coords = xpos
        if abs(self._direction) == 2:
            coords = ypos

        if self._direction > 0:

            new_sponge_field = self._amplitude * actx.np.where(
                actx.np.greater(coords, x0),
                (zeros + ((coords - self._x0)/self._thickness) *
                ((coords - self._x0)/self._thickness)),
                zeros + 0.0)
        else:
            new_sponge_field = self._amplitude * actx.np.where(
                actx.np.less(coords, x0),
                (zeros + ((coords - self._x0)/self._thickness) *
                ((coords - self._x0)/self._thickness)),
                zeros + 0.0)

        left_edge = actx.np.greater(xpos, self._xmin)
        right_edge = actx.np.less(xpos, self._xmax)
        bottom_edge = actx.np.greater(ypos, self._ymin)
        top_edge = actx.np.less(ypos, self._ymax)

        inside_block = left_edge*right_edge*top_edge*bottom_edge

        sponge_field = actx.np.where(inside_block,
                                     sponge_field + new_sponge_field,
                                     sponge_field)

        return sponge_field


class IsentropicInflow:
    r"""Fluid state initializer for isentropic inflow""

    Creates a flow solution from mach, total pressure and total temperature
    Optionally smooths the solution near walls to account for noslip, isothermal
    boundary conditions.

    Optionally takes a pressure function that allows the total pressure
    to vary with time

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, dim, T0, P0, mass_frac, mach, gamma,
                 temp_wall, temp_sigma=0., vel_sigma=0.,
                 smooth_x0=-1000., smooth_x1=1000.,
                 smooth_y0=-1000., smooth_y1=1000.,
                 smooth_z0=-1000., smooth_z1=1000.,
                 smooth_r0=None, smooth_r1=1000.,
                 nspecies=0, normal_dir=None, p_fun=None):

        self._P0 = P0
        self._T0 = T0
        self._dim = dim
        self._mach = mach
        self._gamma = gamma
        self._nspecies = nspecies

        # wall smoothing parameters
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma

        # wall edges for smoothing
        self._x0 = smooth_x0
        self._y0 = smooth_y0
        self._z0 = smooth_z0
        self._r0 = smooth_r0
        self._x1 = smooth_x1
        self._y1 = smooth_y1
        self._z1 = smooth_z1
        self._r1 = smooth_r1

        if smooth_r0 is None:
            self._r0 = np.zeros(shape=(dim,))
            self._r0[0] = 1

        if normal_dir is None:
            self._normal_dir = np.zeros(shape=(dim,))
            self._normal_dir[0] = 1
        else:
            self._normal_dir = normal_dir

        if self._normal_dir.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional normal_dir")

        if self._r0.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional r0")

        if mass_frac is None:
            if nspecies > 0:
                mass_frac = np.zeros(shape=(nspecies,))

        if p_fun is not None:
            self._p_fun = p_fun

        self._mass_frac = mass_frac

    def __call__(self,  x_vec, gas_model, *, time=0, **kwargs):

        actx = x_vec[0].array_context
        zeros = 0*x_vec[0]
        ones = zeros + 1.0

        if self._p_fun is not None:
            P0 = self._p_fun(time)
        else:
            P0 = self._P0
        T0 = self._T0

        pressure = getIsentropicPressure(
            mach=self._mach,
            P0=P0,
            gamma=self._gamma
        )
        temperature = getIsentropicTemperature(
            mach=self._mach,
            T0=T0,
            gamma=self._gamma
        )

        pressure = pressure*ones
        temperature = temperature*ones

        def smoothing_func(dim, x_vec, sigma):
            actx = x_vec[0].array_context
            radial_pos = actx.np.sqrt(
                np.dot(x_vec - self._r0, x_vec - self._r0))

            smoothing_left = smooth_step(actx, sigma*(x_vec[0]-self._x0))
            smoothing_right = smooth_step(actx, -sigma*(x_vec[0]-self._x1))
            smoothing_bottom = smooth_step(actx, sigma*(x_vec[1]-self._y0))
            smoothing_top = smooth_step(actx, -sigma*(x_vec[1]-self._y1))
            smoothing_radius = smooth_step(actx, sigma*(
                actx.np.abs(radial_pos - self._r1)))
            if self._dim == 3:
                smoothing_fore = smooth_step(actx, sigma*(x_vec[2]-self._z0))
                smoothing_aft = smooth_step(actx, -sigma*(x_vec[2]-self._z1))
            else:
                smoothing_fore = ones
                smoothing_aft = ones

            return (smoothing_left*smoothing_right*smoothing_bottom *
                    smoothing_top*smoothing_aft*smoothing_fore *
                    smoothing_radius)

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        wall_temperature = self._temp_wall
        if self._temp_sigma > 0:
            sigma = self._temp_sigma

            sfunc = smoothing_func(self._dim, x_vec, sigma)
            temperature = (wall_temperature +
                (temperature - wall_temperature)*sfunc)

        y = np.zeros(self._nspecies, dtype=object)
        for i in range(self._nspecies):
            y[i] = self._mass_frac[i]

        mass = gas_model.eos.get_density(pressure=pressure,
                                         temperature=temperature,
                                         species_mass_fractions=y)
        energy = mass*gas_model.eos.get_internal_energy(temperature=temperature,
                                                        species_mass_fractions=y)

        velocity = np.zeros(self._dim, dtype=float)
        mom = mass*velocity
        cv = make_conserved(dim=self._dim, mass=mass, momentum=mom,
                            energy=energy, species_mass=mass*y)

        vmag = self._mach*gas_model.eos.sound_speed(cv, temperature)

        # modify the velocity in the near-wall region to have a smooth profile
        # this approximates the BL velocity profile
        if self._vel_sigma > 0:
            sigma = self._vel_sigma
            sfunc = smoothing_func(self._dim, x_vec, sigma)
            vmag = (vmag*sfunc)

        velocity = vmag*self._normal_dir
        mom = mass*velocity
        energy = (energy + np.dot(mom, mom)/(2.0*mass))

        cv = make_conserved(dim=self._dim, mass=mass, momentum=mom,
                            energy=energy, species_mass=mass*y)

        av_smu = actx.np.zeros_like(cv.mass)
        av_sbeta = actx.np.zeros_like(cv.mass)
        av_skappa = actx.np.zeros_like(cv.mass)
        av_sd = actx.np.zeros_like(cv.mass)
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temperature,
                                smoothness_mu=av_smu,
                                smoothness_beta=av_sbeta,
                                smoothness_kappa=av_skappa,
                                smoothness_d=av_sd,
                                limiter_func=None,
                                limiter_dd=None)

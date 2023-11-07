"""mirgecom driver initializer for the Y2 prediction."""
import numpy as np
from pytools.obj_array import make_obj_array
from mirgecom.fluid import make_conserved


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

        if abs(self._direction) == 1:
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


class MomentumSource:
    r"""Add x - momentum to the flow.

    Momentum is added to the system as a gaussian of the form:

    .. math::

        e &= e + e_{a}\exp^{(1-r^{2})}\\

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, dim, center=None, width=1.0,
                 amplitude=0., amplitude_func=None):
        r"""Initialize the momentum parameters.

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
        self.expterm = 0

    def __call__(self, x_vec, state, eos, time, **kwargs):
        """
        Create the momentum source at time *t* and location *x_vec*.

        the source at time *t* is created by evaluating the gaussian
        with time-dependent amplitude at *t*.

        Parameters
        ----------
        cv: :class:`mirgecom.fluid.ConservedVars`
            Fluid conserved quantities
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        """
        from pytools.obj_array import make_obj_array
        from mirgecom.fluid import make_conserved
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
        self.expterm = expterm

        velocity_source = np.zeros(self._dim, dtype=object)
        velocity_source[0] = velocity_source[0] + expterm
        momentum_source = velocity_source * state.mass_density

        energy_source = 0 * state.energy_density
        mass = 0 * state.mass_density
        species_mass = 0 * state.species_mass_fractions

        return make_conserved(dim=self._dim, mass=mass,
                              energy=energy_source,
                              momentum=momentum_source,
                              species_mass=species_mass)


class NormalShockSource:
    r"""Add an artificial normal shock into the flow.

    Momentum is added to the system as a gaussian of the form:

    .. math::

        e &= e + e_{a}\exp^{(1-r^{2})}\\

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, dim, center=None, width=1.0,
                 amplitude_func=None,
                 T1=87, T2=100, P1=1000, P2=10000):
        r"""Initialize the momentum parameters.

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
        #self._amplitude = amplitude
        self._width = width
        self._tot_width = self._width * 6.0697
        self._amplitude_func = amplitude_func
        self.expterm = 0
        self.T2 = T2
        self.T1 = T1
        self.P2 = P2
        self.P1 = P1
        self._pi = 3.14159

    def __call__(self, x_vec, state, eos, time, **kwargs):
        """
        Create the momentum source at time *t* and location *x_vec*.

        the source at time *t* is created by evaluating the gaussian
        with time-dependent amplitude at *t*.

        Parameters
        ----------
        cv: :class:`mirgecom.fluid.ConservedVars`
            Fluid conserved quantities
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        """
        from pytools.obj_array import make_obj_array
        from mirgecom.fluid import make_conserved
        t = time

        loc = self._center

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))

        # set the target state for this time step depending on the amplitude function
        # varies between X1 and X2
        if self._amplitude_func is not None:
            P_targ = (self.P2 - self.P1) * self._amplitude_func(t) + self.P1
            T_targ = (self.T2 - self.T1) * self._amplitude_func(t) + self.T1
        else:
            P_targ = self.P2
            T_targ = self.T2

        # set the necessary change to reach the target temperature
        delta_P = P_targ - state.pressure
        delta_T = T_targ - state.temperature

        expterm_pressure = (delta_P * actx.np.exp(-(r ** 2) /
                                                (2 * self._width * self._width)))
        expterm_temperature = (delta_T * actx.np.exp(-(r ** 2) /
                                                    (2 * self._width * self._width)))

        # increase temperature
        #T_targ = 0
        temperature = actx.np.where(
            actx.np.greater(state.temperature, T_targ),
            state.temperature,
            state.temperature + expterm_temperature)

        # increase pressure
        pressure = actx.np.where(
            actx.np.greater(state.pressure, P_targ),
            state.pressure,
            state.pressure + expterm_pressure)

        y = state.species_mass_fractions

        # get density with new pressure and temperature
        new_mass = eos.get_density(pressure=pressure, temperature=temperature,
                                   species_mass_fractions=y)

        # calculate source required to match new density
        mass_source = new_mass - state.mass_density

        # no momentum change
        momentum_source = 0 * state.velocity

        # add energy for the increase in temperature
        energy_source = (eos.heat_capacity_cv(state, state.temperature)
                            * (temperature - state.temperature))

        # no mass fraction changes
        species_mass = 0 * state.species_mass_fractions

        return make_conserved(dim=self._dim, mass=mass_source,
                              energy=energy_source,
                              momentum=momentum_source,
                              species_mass=species_mass)

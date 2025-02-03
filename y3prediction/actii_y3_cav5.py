"""mirgecom driver initializer for the Y2 prediction."""
import numpy as np
from mirgecom.fluid import make_conserved

from y3prediction.utils import (
    getIsentropicPressure,
    getIsentropicTemperature,
    getMachFromAreaRatio,
    get_theta_from_data,
    smooth_step
)


class InitACTII:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: add_injection
    """

    def __init__(
            self, *, dim=2, nspecies=0, geom_top, geom_bottom,
            P0, T0, temp_wall, temp_sigma, vel_sigma, gamma_guess,
            mass_frac=None,
            inj_pres, inj_temp, inj_vel,
            inj_pres_u, inj_temp_u, inj_vel_u,
            inj_mass_frac=None,
            inj_gamma_guess,
            inj_temp_sigma, inj_vel_sigma,
            inj_ytop, inj_ybottom,
            inj_mach, injection=True
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
        geom_top: numpy.ndarray
            coordinates for the top wall
        geom_bottom: numpy.ndarray
            coordinates for the bottom wall
        """

        if mass_frac is None:
            if nspecies > 0:
                mass_frac = np.zeros(shape=(nspecies,))

        if inj_mass_frac is None:
            if nspecies > 0:
                inj_mass_frac = np.zeros(shape=(nspecies,))

        if inj_vel is None:
            inj_vel = np.zeros(shape=(dim,))

        if inj_vel_u is None:
            inj_vel_u = np.zeros(shape=(dim,))

        self._dim = dim
        self._nspecies = nspecies
        self._P0 = P0
        self._T0 = T0
        self._geom_top = geom_top
        self._geom_bottom = geom_bottom
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._gamma_guess = gamma_guess
        # TODO, calculate these from the geometry files
        self._throat_height = 3.61909e-3
        self._x_throat = 0.283718298
        self._mass_frac = mass_frac

        self._inj_P0 = inj_pres
        self._inj_T0 = inj_temp
        self._inj_vel = inj_vel
        self._inju_P0 = inj_pres_u
        self._inju_T0 = inj_temp_u
        self._inju_vel = inj_vel_u
        self._inj_gamma_guess = inj_gamma_guess

        self._temp_sigma_injection = inj_temp_sigma
        self._vel_sigma_injection = inj_vel_sigma
        self._inj_mass_frac = inj_mass_frac
        self._inj_ytop = inj_ytop
        self._inj_ybottom = inj_ybottom
        self._inj_mach = inj_mach
        self._injection = injection

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
        gamma = zeros + self._gamma_guess
        ytop = zeros
        ybottom = zeros
        theta = zeros
        gamma_guess = self._gamma_guess

        theta_geom_top = get_theta_from_data(self._geom_top)
        theta_geom_bottom = get_theta_from_data(self._geom_bottom)

        # process the mesh piecemeal, one interval at a time
        # linearly interpolate between the data points
        area_ratio = ((self._geom_top[0][1] - self._geom_bottom[0][1]) /
                      self._throat_height)
        if self._geom_top[0][0] < self._x_throat:
            mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                             gamma=gamma_guess,
                                             mach_guess=0.01)
        elif self._geom_top[0][0] > self._x_throat:
            mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                             gamma=gamma_guess,
                                             mach_guess=1.01)
        else:
            mach_left = 1.0

        x_left = self._geom_top[0][0]
        ytop_left = self._geom_top[0][1]
        ybottom_left = self._geom_bottom[0][1]
        theta_top_left = theta_geom_top[0][1]
        theta_bottom_left = theta_geom_bottom[0][1]

        pres_left = getIsentropicPressure(mach=mach_left,
                                          P0=self._P0,
                                          gamma=gamma_guess)
        temp_left = getIsentropicTemperature(mach=mach_left,
                                             T0=self._T0,
                                             gamma=gamma_guess)

        # iterate over gamma to get a better initial condition
        y = np.zeros(self._nspecies, dtype=object)
        for i in range(self._nspecies):
            y[i] = self._mass_frac[i]
        mass = actx.to_numpy(eos.get_density(pressure=pres_left, temperature=temp_left,
                               species_mass_fractions=y))
        energy = mass*actx.to_numpy(eos.get_internal_energy(temperature=temp_left,
                                              species_mass_fractions=y))

        velocity = np.zeros(self._dim, dtype=object)
        mom = mass*velocity

        cv = make_conserved(dim=self._dim, mass=mass, momentum=mom,
                            energy=energy, species_mass=mass*y)
        gamma_left = eos.gamma(cv, temp_left)

        gamma_error = (gamma_guess - gamma_left)
        gamma_iter = gamma_left
        toler = 1.e-6
        # iterate over the gamma/mach since gamma = gamma(T)
        while gamma_error > toler:
            if self._geom_top[0][0] < self._x_throat:
                mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma_iter,
                                                 mach_guess=0.01)
            elif self._geom_top[0][0] > self._x_throat:
                mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma_iter,
                                                 mach_guess=1.01)
            else:
                mach_left = 1.0

            pres_left = getIsentropicPressure(mach=mach_left,
                                                 P0=self._P0,
                                                 gamma=gamma_iter)
            temp_left = getIsentropicTemperature(mach=mach_left,
                                                    T0=self._T0,
                                                    gamma=gamma_iter)
            mass = eos.get_density(pressure=pres_left, temperature=temp_left,
                                   species_mass_fractions=y)
            energy = mass*eos.get_internal_energy(temperature=temp_left,
                                                  species_mass_fractions=y)

            velocity = np.zeros(self._dim, dtype=object)
            mom = mass*velocity
            cv = make_conserved(dim=self._dim, mass=mass, momentum=mom,
                                energy=energy, species_mass=mass*y)
            gamma_left = eos.gamma(cv, temp_left)
            gamma_error = (gamma_iter - gamma_left)
            gamma_iter = gamma_left

        for ind in range(1, self._geom_top.shape[0]):
            area_ratio = ((self._geom_top[ind][1] - self._geom_bottom[ind][1]) /
                          self._throat_height)
            if self._geom_top[ind][0] < self._x_throat:
                mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma_left,
                                                 mach_guess=0.01)
            elif self._geom_top[ind][0] > self._x_throat:
                mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma_left,
                                                 mach_guess=1.01)
            else:
                mach_right = 1.0

            ytop_right = self._geom_top[ind][1]
            ybottom_right = self._geom_bottom[ind][1]
            theta_top_right = theta_geom_top[ind][1]
            theta_bottom_right = theta_geom_bottom[ind][1]

            pres_right = getIsentropicPressure(mach=mach_right,
                                               P0=self._P0,
                                               gamma=gamma_left)
            temp_right = getIsentropicTemperature(mach=mach_right,
                                                  T0=self._T0,
                                                  gamma=gamma_left)

            # iterate over gamma to get a better initial condition
            y = np.zeros(self._nspecies, dtype=object)
            for i in range(self._nspecies):
                y[i] = self._mass_frac[i]
            mass = actx.to_numpy(eos.get_density(pressure=pres_right, temperature=temp_right,
                                   species_mass_fractions=y))
            energy = mass*actx.to_numpy(eos.get_internal_energy(temperature=temp_right,
                                                  species_mass_fractions=y))

            velocity = np.zeros(self._dim, dtype=object)
            mom = mass*velocity
            cv = make_conserved(dim=self._dim, mass=mass, momentum=mom,
                                energy=energy, species_mass=mass*y)
            gamma_right = eos.gamma(cv, temp_right)

            gamma_error = (gamma_left - gamma_right)
            gamma_iter = gamma_right
            toler = 1.e-6
            # iterate over the gamma/mach since gamma = gamma(T)
            while gamma_error > toler:
                if self._geom_top[ind][0] < self._x_throat:
                    mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                     gamma=gamma_iter,
                                                     mach_guess=0.01)
                elif self._geom_top[ind][0] > self._x_throat:
                    mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                     gamma=gamma_iter,
                                                     mach_guess=1.01)
                else:
                    mach_right = 1.0

                pres_right = getIsentropicPressure(mach=mach_right,
                                                     P0=self._P0,
                                                     gamma=gamma_iter)
                temp_right = getIsentropicTemperature(mach=mach_right,
                                                        T0=self._T0,
                                                        gamma=gamma_iter)
                mass = eos.get_density(pressure=pres_right,
                                       temperature=temp_right,
                                       species_mass_fractions=y)
                energy = mass*eos.get_internal_energy(temperature=temp_right,
                                                      species_mass_fractions=y)

                velocity = np.zeros(self._dim, dtype=object)
                mom = mass*velocity
                cv = make_conserved(dim=self._dim, mass=mass, momentum=mom,
                                    energy=energy, species_mass=mass*y)
                gamma_right = eos.gamma(cv, temp_right)
                gamma_error = (gamma_iter - gamma_right)
                gamma_iter = gamma_right

            # interpolate our data
            x_right = self._geom_top[ind][0]

            dx = x_right - x_left
            dm = mach_right - mach_left
            dg = gamma_right - gamma_left
            dyt = ytop_right - ytop_left
            dyb = ybottom_right - ybottom_left
            dtb = theta_bottom_right - theta_bottom_left
            dtt = theta_top_right - theta_top_left

            local_mach = mach_left + (xpos - x_left)*dm/dx
            local_gamma = gamma_left + (xpos - x_left)*dg/dx
            local_ytop = ytop_left + (xpos - x_left)*dyt/dx
            local_ybottom = ybottom_left + (xpos - x_left)*dyb/dx
            local_theta_bottom = theta_bottom_left + (xpos - x_left)*dtb/dx
            local_theta_top = theta_top_left + (xpos - x_left)*dtt/dx

            local_theta = (local_theta_bottom +
                           (local_theta_top - local_theta_bottom) /
                           (local_ytop - local_ybottom)*(ypos - local_ybottom))

            # extend just a a little bit to catch the edges
            left_edge = actx.np.greater(xpos, x_left - 1.e-6)
            right_edge = actx.np.less(xpos, x_right + 1.e-6)
            inside_block = left_edge*right_edge

            mach = actx.np.where(inside_block, local_mach, mach)
            gamma = actx.np.where(inside_block, local_gamma, gamma)
            ytop = actx.np.where(inside_block, local_ytop, ytop)
            ybottom = actx.np.where(inside_block, local_ybottom, ybottom)
            theta = actx.np.where(inside_block, local_theta, theta)

            mach_left = mach_right
            gamma_left = gamma_right
            ytop_left = ytop_right
            ybottom_left = ybottom_right
            theta_bottom_left = theta_bottom_right
            theta_top_left = theta_top_right
            x_left = x_right

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
        #unsmoothed_temperature = temperature

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        wall_temperature = self._temp_wall
        if self._temp_sigma > 0:
            sigma = self._temp_sigma
            smoothing_top = smooth_step(actx, -sigma*(ypos-ytop))
            smoothing_bottom = smooth_step(
                actx, sigma*actx.np.abs(ypos-ybottom))
            smoothing_fore = ones
            smoothing_aft = ones
            z0 = -0.0175
            z1 = 0.0175
            if self._dim == 3:
                smoothing_fore = smooth_step(actx, sigma*(zpos-z0))
                smoothing_aft = smooth_step(actx, -sigma*(zpos-z1))

            smooth_temperature = (wall_temperature +
                (temperature - wall_temperature)*smoothing_top*smoothing_bottom *
                                                 smoothing_fore*smoothing_aft)
        else:
            smooth_temperature = temperature

        # make a little region along the top of the cavity where we don't want
        # the temperature smoothed
        #xc_left = zeros + 0.65163 + 0.0004
        #xc_right = zeros + 0.72163 - 0.0004
        #xc_left = zeros + 0.60628 + 0.0010
        #xc_right = zeros + 0.63578 - 0.0015
        xc_left = zeros + 0.60628
        xc_right = zeros + 0.63578
        yc_top = zeros - 0.006
        yc_bottom = zeros - 0.01
        zc_fore = 0.0175 - 0.001
        zc_aft = -0.0175 + 0.001

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        fore_edge = ones
        aft_edge = ones
        if self._dim == 3:
            fore_edge = actx.np.less(zpos, zc_fore)
            aft_edge = actx.np.greater(zpos, zc_aft)
        inside_block = left_edge*right_edge*top_edge*bottom_edge*fore_edge*aft_edge
        smooth_temperature = actx.np.where(inside_block, temperature,
                                           smooth_temperature)

        # smooth the temperature in the cavity region, this helps along the wall
        # initially in pressure/temperature equilibrium with the exterior flow
        xc_left = zeros + 0.60627
        xc_right = zeros + 0.65088
        #xc_left = zeros + 0.65163 - 0.000001
        #xc_right = zeros + 0.742 + 0.000001
        #xc_left = zeros + 0.60628 + 0.0004
        #xc_right = zeros + 0.63578 - 0.0004
        #yc_top = zeros - 0.0083245 + 0.0006
        yc_top = zeros - 0.0083245
        if self._temp_sigma <= 0:
            yc_top = zeros - 0.0099
        #yc_bottom = zeros - 0.0283245
        #xc_bottom = zeros + 0.70163
        yc_bottom = zeros - 0.0133245
        xc_bottom = zeros + 0.63078
        wall_theta = np.sqrt(2)/2.

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        inside_cavity = left_edge*right_edge*top_edge

        # smooth the temperature at the cavity walls
        wall_dist = (wall_theta*(ypos - yc_bottom) -
                     wall_theta*(xpos - xc_bottom))
        if self._temp_sigma > 0.:
            sigma = self._temp_sigma
            smoothing_front = smooth_step(actx, sigma*abs(xpos-xc_left))
            smoothing_bottom = smooth_step(actx, sigma*(ypos-yc_bottom))
            smoothing_slant = smooth_step(actx, sigma*wall_dist)
            cavity_temperature = (wall_temperature +
                (temperature - wall_temperature) *
                 #smoothing_front*smoothing_bottom*smoothing_slant)
                 fore_edge*aft_edge*smoothing_front*smoothing_bottom*smoothing_slant)
        else:
            sigma = 1500
            smoothing_slant = smooth_step(actx, sigma*wall_dist)
            cavity_temperature = (wall_temperature +
                (temperature - wall_temperature)*smoothing_slant)
        smooth_temperature = actx.np.where(inside_cavity,
                                           cavity_temperature,
                                           #cavity_temperature)
                                           #smooth_temperature,
                                           smooth_temperature)

        # smooth the temperature at the upstream corner
        xc_left = zeros + 0.60627
        xc_right = xc_left + 0.0015
        yc_bottom = zeros - 0.0083245
        yc_top = yc_bottom + 0.0015
        zc_aft = zeros - 0.0175 + 0.001
        zc_fore = zeros + 0.0175 - 0.001

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        aft_edge = ones
        fore_edge = ones
        if self._dim == 3:
            aft_edge = actx.np.greater(zpos, zc_aft)
            fore_edge = actx.np.less(zpos, zc_fore)

        inside_corner = left_edge*right_edge*top_edge*bottom_edge*aft_edge*fore_edge

        # smooth the temperature at the cavity walls
        corner_dist = actx.np.sqrt((ypos - yc_bottom)*(ypos - yc_bottom) +
                              (xpos - xc_left)*(xpos - xc_left))
        if self._temp_sigma > 0.:
            sigma = self._temp_sigma
            smoothing_corner = smooth_step(actx, sigma*corner_dist)
            corner_temperature = (wall_temperature +
                (temperature - wall_temperature)*smoothing_corner)
        else:
            sigma = 1500
            smoothing_corner = smooth_step(actx, sigma*corner_dist)
            corner_temperature = (wall_temperature +
                (temperature - wall_temperature)*smoothing_corner)
        smooth_temperature = actx.np.where(inside_corner,
                                           corner_temperature,
                                           smooth_temperature)

        # smooth the temperature at the downstream corner
        xc_right = zeros + 0.63578
        xc_left = xc_right - 0.0015
        yc_bottom = zeros - 0.0083245
        yc_top = yc_bottom + 0.0015
        zc_aft = zeros - 0.0175 + 0.001
        zc_fore = zeros + 0.0175 - 0.001

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        aft_edge = ones
        fore_edge = ones
        if self._dim == 3:
            aft_edge = actx.np.greater(zpos, zc_aft)
            fore_edge = actx.np.less(zpos, zc_fore)

        inside_corner = left_edge*right_edge*top_edge*bottom_edge*aft_edge*fore_edge

        # smooth the temperature at the cavity walls
        corner_dist = actx.np.sqrt((ypos - yc_bottom)*(ypos - yc_bottom) +
                              (xpos - xc_right)*(xpos - xc_right))
        wall_dist = (wall_theta*(ypos - yc_bottom) -
                     wall_theta*(xpos - xc_right))
        if self._temp_sigma > 0.:
            sigma = self._temp_sigma
            smoothing_corner = smooth_step(actx, sigma*corner_dist)
            smoothing_slant = smooth_step(actx, sigma*wall_dist)
            smoothing_top = smooth_step(actx, sigma*(ypos - yc_bottom))
            smoothing_func = smoothing_corner
            #smoothing_func = (smoothing_top + smoothing_slant)/2.
            #smoothing_func = (smoothing_top*smoothing_slant*smoothing_corner)
            corner_temperature = (wall_temperature +
                (temperature - wall_temperature)*smoothing_func)
        else:
            sigma = 1500
            smoothing_corner = smooth_step(actx, sigma*corner_dist)
            corner_temperature = (wall_temperature +
                (temperature - wall_temperature)*smoothing_corner)
        smooth_temperature = actx.np.where(inside_corner,
                                           corner_temperature,
                                           #corner_temperature)
                                           smooth_temperature)

        temperature = smooth_temperature

        y = ones*self._mass_frac
        mass = eos.get_density(pressure=pressure, temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        velocity = np.zeros(self._dim, dtype=object)
        mom = mass*velocity
        cv = make_conserved(dim=self._dim, mass=mass, momentum=mom, energy=energy,
                            species_mass=mass*y)
        velocity[0] = mach*eos.sound_speed(cv, temperature)

        # modify the velocity in the near-wall region to have a smooth profile
        # this approximates the BL velocity profile
        if self._vel_sigma > 0:
            sigma = self._vel_sigma
            smoothing_top = smooth_step(actx, -sigma*(ypos-ytop))
            smoothing_bottom = smooth_step(actx, sigma*(actx.np.abs(ypos-ybottom)))
            smoothing_fore = ones
            smoothing_aft = ones
            if self._dim == 3:
                smoothing_fore = smooth_step(actx, sigma*(zpos-z0))
                smoothing_aft = smooth_step(actx, -sigma*(zpos-z1))
            velocity[0] = (velocity[0]*smoothing_top*smoothing_bottom *
                           smoothing_fore*smoothing_aft)

        # split into x and y components
        velocity[1] = velocity[0]*actx.np.sin(theta)
        velocity[0] = velocity[0]*actx.np.cos(theta)
        if self._dim == 3:
            velocity[2] = 0.*velocity[2]

        # zero out the velocity in the cavity region, let the flow develop naturally
        #xc_left = zeros + 0.60627
        xc_left = zeros + 0.50
        xc_right = zeros + 0.65088
        yc_top = zeros - 0.0083245
        if self._vel_sigma <= 0:
            yc_top = zeros - 0.0099
        yc_bottom = zeros - 0.0133245
        xc_bottom = zeros + 0.63078

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        inside_cavity = left_edge*right_edge*top_edge

        # zero of the velocity
        for i in range(self._dim):
            velocity[i] = actx.np.where(inside_cavity, zeros, velocity[i])

        # modify the velocity above the cavity to have a smooth profile
        # this approximates the BL velocity profile
        if self._vel_sigma <= 0:

            xc_left = zeros + 0.606 - 0.000001
            xc_right = zeros + 0.636 + 0.000001
            yc_top = zeros
            #yc_bottom = zeros - 0.0083246
            yc_bottom = zeros - 0.0099

            left_edge = actx.np.greater(xpos, xc_left)
            right_edge = actx.np.less(xpos, xc_right)
            top_edge = actx.np.less(ypos, yc_top)
            bottom_edge = actx.np.greater(ypos, yc_bottom)
            above_cavity = left_edge*right_edge*top_edge*bottom_edge

            # cavity slip surface
            sigma = 500
            smoothing = smooth_step(actx, sigma*(actx.np.abs(ypos-yc_bottom)))

            for i in range(self._dim):
                velocity[i] = actx.np.where(above_cavity,
                                            velocity[i]*smoothing,
                                            velocity[i])

            xc_left = zeros + 0.606 - 0.000001
            xc_right = zeros + 0.636 + 0.000001
            yc_top = zeros - 0.0083246
            yc_bottom = zeros - 0.0099

            left_edge = actx.np.greater(xpos, xc_left)
            right_edge = actx.np.less(xpos, xc_right)
            top_edge = actx.np.less(ypos, yc_top)
            bottom_edge = actx.np.greater(ypos, yc_bottom)
            below_slip = left_edge*right_edge*top_edge*bottom_edge

            # upstream corner
            sigma = 250
            smoothing = smooth_step(actx, sigma*(actx.np.abs(xpos-xc_left)))

            for i in range(self._dim):
                velocity[i] = actx.np.where(below_slip,
                                            velocity[i]*smoothing,
                                            velocity[i])

            # downstream corner
            sigma = 1000
            smoothing = smooth_step(actx, sigma*(actx.np.abs(xpos-xc_right)))

            for i in range(self._dim):
                velocity[i] = actx.np.where(below_slip,
                                            velocity[i]*smoothing,
                                            velocity[i])

        mom = mass*velocity
        energy = (energy + np.dot(mom, mom)/(2.0*mass))
        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )

    def add_injection(self, cv, pressure, temperature, x_vec, eos, *, time=0.0):
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

        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zpos = actx.np.zeros_like(xpos)
        if self._dim == 3:
            zpos = x_vec[2]

        zeros = actx.np.zeros_like(xpos)
        ones = zeros + 1.0

        # get the current mesh conditions
        mass = cv.mass
        energy = cv.energy
        velocity = cv.velocity
        y = cv.species_mass_fractions

        # fuel stream initialization
        # initially in pressure/temperature equilibrium with the cavity
        #inj_left = 0.71
        # even with the bottom corner
        #inj_left = 0.632
        # even with the top corner
        inj_left = 0.6337
        inj_right = 0.651
        inj_top = -0.0105
        inj_bottom = -0.01213
        inj_fore = 1.59e-3
        inj_aft = -1.59e-3
        xc_left = zeros + inj_left
        xc_right = zeros + inj_right
        yc_top = zeros + inj_top
        yc_bottom = zeros + inj_bottom
        zc_fore = zeros + inj_fore
        zc_aft = zeros + inj_aft

        yc_center = zeros - 0.01212 + 1.59e-3/2.
        zc_center = zeros
        inj_radius = 1.59e-3/2.

        if self._dim == 2:
            radius = actx.np.sqrt((ypos - yc_center)**2)
        else:
            radius = actx.np.sqrt((ypos - yc_center)**2 + (zpos - zc_center)**2)

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        top_edge = actx.np.less(ypos, yc_top)
        aft_edge = ones
        fore_edge = ones
        if self._dim == 3:
            aft_edge = actx.np.greater(zpos, zc_aft)
            fore_edge = actx.np.less(zpos, zc_fore)
        inside_injector = (left_edge*right_edge*top_edge*bottom_edge *
                           aft_edge*fore_edge)

        inj_y = ones*self._inj_mass_frac

        inj_velocity = mass*np.zeros(self._dim, dtype=object)
        inj_velocity[0] = self._inj_vel[0]

        inj_mach = self._inj_mach*ones

        # smooth out the injection profile
        # relax to the cavity temperature/pressure/velocity
        inj_x0 = 0.6375
        inj_fuel_x0 = 0.6425
        inj_sigma = 1500

        # left extent
        inj_tanh = inj_sigma*(inj_fuel_x0 - xpos)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        for i in range(self._nspecies):
            inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

        # transition the mach number from 0 (cavitiy) to 1 (injection)
        inj_tanh = inj_sigma*(inj_x0 - xpos)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        inj_mach = inj_weight*inj_mach

        # assume a smooth transition in gamma, could calculate it
        inj_gamma = (self._gamma_guess +
            (self._inj_gamma_guess - self._gamma_guess)*inj_weight)

        inj_pressure = getIsentropicPressure(
            mach=inj_mach,
            P0=self._inj_P0,
            gamma=inj_gamma
        )
        inj_temperature = getIsentropicTemperature(
            mach=inj_mach,
            T0=self._inj_T0,
            gamma=inj_gamma
        )

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(
            temperature=inj_temperature, species_mass_fractions=inj_y)

        inj_velocity = mass*np.zeros(self._dim, dtype=object)
        inj_mom = inj_mass*inj_velocity

        # the velocity magnitude
        inj_cv = make_conserved(dim=self._dim, mass=inj_mass, momentum=inj_mom,
                                energy=inj_energy, species_mass=inj_mass*inj_y)

        inj_velocity[0] = -inj_mach*eos.sound_speed(inj_cv, inj_temperature)

        # relax the velocity, temperature, and pressure at the injector interface
        for i in range(self._dim):
            inj_velocity[i] = velocity[i] + \
                (inj_velocity[i] - velocity[i])*inj_weight
        inj_pressure = pressure + (inj_pressure - pressure)*inj_weight
        inj_temperature = (temperature +
            (inj_temperature - temperature)*inj_weight)

        # we need to calculate the velocity from a prescribed mass flow rate
        # this will need to take into account the velocity relaxation at the
        # injector walls
        #inj_velocity[0] = (velocity[0] +
        #                   (self._inj_vel[0] - velocity[0])*inj_weight)

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        if self._temp_sigma_injection > 0.:
            sigma = self._temp_sigma_injection
            wall_temperature = self._temp_wall
            smoothing_radius = smooth_step(
                actx, sigma*(actx.np.abs(radius - inj_radius)))
            inj_temperature = (wall_temperature +
                (inj_temperature - wall_temperature)*smoothing_radius)

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(
            temperature=inj_temperature, species_mass_fractions=inj_y)

        # modify the velocity in the near-wall region to have a smooth profile
        # this approximates the BL velocity profile
        if self._vel_sigma_injection > 0.:
            sigma = self._vel_sigma_injection
            smoothing_radius = smooth_step(
                actx, sigma*(actx.np.abs(radius - inj_radius)))
            inj_velocity[0] = inj_velocity[0]*smoothing_radius

        # use the species field with fuel added everywhere
        for i in range(self._nspecies):
            y[i] = actx.np.where(inside_injector, inj_y[i], y[i])

        # recompute the mass and energy (outside the injector) to account for
        # the change in mass fraction
        mass = eos.get_density(pressure=pressure,
                               temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        mass = actx.np.where(inside_injector, inj_mass, mass)
        velocity[0] = actx.np.where(inside_injector,
                                    inj_velocity[0],
                                    velocity[0])
        energy = actx.np.where(inside_injector, inj_energy, energy)

        mom = mass*velocity
        energy = (energy + np.dot(mom, mom)/(2.0*mass))
        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )

    def add_injection_upstream(self, cv, pressure, temperature,
                               x_vec, eos, *, time=0.0):
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

        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zpos = actx.np.zeros_like(xpos)
        if self._dim == 3:
            zpos = x_vec[2]

        zeros = actx.np.zeros_like(xpos)
        ones = zeros + 1.0

        # get the current mesh conditions
        mass = cv.mass
        energy = cv.energy
        velocity = cv.velocity
        y = cv.species_mass_fractions

        # fuel stream initialization
        # initially in pressure/temperature equilibrium with the cavity
        # even with the bottom corner
        inj_left = 0.53243
        # even with the top corner
        inj_right = 0.53404
        inj_top = -0.0083245
        inj_bottom = -0.02253
        inj_fore = 1.59e-3
        inj_aft = -1.59e-3
        xc_left = zeros + inj_left
        xc_right = zeros + inj_right
        yc_top = zeros + inj_top
        yc_bottom = zeros + inj_bottom
        zc_fore = zeros + inj_fore
        zc_aft = zeros + inj_aft

        inj_radius = 1.59e-3/2.
        xc_center = zeros + inj_left + inj_radius
        zc_center = zeros

        if self._dim == 2:
            radius = actx.np.sqrt((xpos - xc_center)**2)
        else:
            radius = actx.np.sqrt((xpos - xc_center)**2 + (zpos - zc_center)**2)

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        top_edge = actx.np.less(ypos, yc_top)
        aft_edge = ones
        fore_edge = ones
        if self._dim == 3:
            aft_edge = actx.np.greater(zpos, zc_aft)
            fore_edge = actx.np.less(zpos, zc_fore)
        inside_injector = (left_edge*right_edge*top_edge*bottom_edge *
                           aft_edge*fore_edge)

        inj_y = ones*self._inj_mass_frac

        inj_velocity = mass*np.zeros(self._dim, dtype=object)
        inj_velocity = self._inju_vel

        inj_mach = self._inj_mach*ones

        # smooth out the injection profile
        # relax to the cavity temperature/pressure/velocity
        inj_y0 = -0.012
        inj_fuel_y0 = -0.015
        inj_sigma = 1500

        inj_tanh = inj_sigma*(ypos - inj_fuel_y0)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        for i in range(self._nspecies):
            inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

        # transition the mach number from 0 (cavitiy) to 1 (injection)
        inj_tanh = inj_sigma*(ypos - inj_y0)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        inj_mach = inj_weight*inj_mach

        # assume a smooth transition in gamma, could calculate it
        inj_gamma = (self._gamma_guess +
            (self._inj_gamma_guess - self._gamma_guess)*inj_weight)

        inj_pressure = getIsentropicPressure(
            mach=inj_mach,
            P0=self._inju_P0,
            gamma=inj_gamma
        )
        inj_temperature = getIsentropicTemperature(
            mach=inj_mach,
            T0=self._inju_T0,
            gamma=inj_gamma
        )

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(
            temperature=inj_temperature, species_mass_fractions=inj_y)

        inj_velocity = mass*np.zeros(self._dim, dtype=object)
        inj_mom = inj_mass*inj_velocity

        # the velocity magnitude
        inj_cv = make_conserved(dim=self._dim, mass=inj_mass, momentum=inj_mom,
                                energy=inj_energy, species_mass=inj_mass*inj_y)

        inj_velocity[1] = inj_mach*eos.sound_speed(inj_cv, inj_temperature)

        # relax the velocity, temperature, and pressure at the injector interface
        for i in range(self._dim):
            inj_velocity[i] = velocity[i] + \
                (inj_velocity[i] - velocity[i])*inj_weight
        inj_pressure = pressure + (inj_pressure - pressure)*inj_weight
        inj_temperature = (temperature +
            (inj_temperature - temperature)*inj_weight)

        # we need to calculate the velocity from a prescribed mass flow rate
        # this will need to take into account the velocity relaxation at the
        # injector walls
        #inj_velocity[0] = (velocity[0] +
        #                   (self._inj_vel[0] - velocity[0])*inj_weight)

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        if self._temp_sigma_injection > 0.:
            sigma = self._temp_sigma_injection
            wall_temperature = self._temp_wall
            smoothing_radius = smooth_step(
                actx, sigma*(actx.np.abs(radius - inj_radius)))
            inj_temperature = (wall_temperature +
                (inj_temperature - wall_temperature)*smoothing_radius)

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(
            temperature=inj_temperature, species_mass_fractions=inj_y)

        # modify the velocity in the near-wall region to have a smooth profile
        # this approximates the BL velocity profile
        if self._vel_sigma_injection > 0.:
            sigma = self._vel_sigma_injection
            smoothing_radius = smooth_step(
                actx, sigma*(actx.np.abs(radius - inj_radius)))
            inj_velocity[1] = inj_velocity[1]*smoothing_radius

        # use the species field with fuel added everywhere
        for i in range(self._nspecies):
            y[i] = actx.np.where(inside_injector, inj_y[i], y[i])

        # recompute the mass and energy (outside the injector) to account for
        # the change in mass fraction
        mass = eos.get_density(pressure=pressure,
                               temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        mass = actx.np.where(inside_injector, inj_mass, mass)
        velocity[1] = actx.np.where(inside_injector,
                                    inj_velocity[1],
                                    velocity[1])
        energy = actx.np.where(inside_injector, inj_energy, energy)

        mom = mass*velocity
        energy = (energy + np.dot(mom, mom)/(2.0*mass))
        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )

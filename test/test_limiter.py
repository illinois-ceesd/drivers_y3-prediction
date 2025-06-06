__copyright__ = """Copyright (C) 2020 University of Illinois Board of Trustees"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
    [PytestPyOpenCLArrayContextFactory])

from arraycontext import (  # noqa
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from y3prediction.prediction import limit_fluid_state_lv
from mirgecom.discretization import create_discretization_collection

from mirgecom.mechanisms import get_mechanism_input
import cantera
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.gas_model import (  # noqa
    GasModel, make_fluid_state, make_operator_fluid_states
)
from mirgecom.fluid import make_conserved
from pytools.obj_array import make_obj_array

import pytest


class MulticomponentLump:
    r"""Solution initializer for multi-component N-dimensional Gaussian lump of mass.

    The Gaussian lump is defined by:

    .. math::

         \rho &= 1.0\\
         {\rho}\mathbf{V} &= {\rho}\mathbf{V}_0\\
         {\rho}E &= \frac{p_0}{(\gamma - 1)} + \frac{1}{2}\rho{|V_0|}^{2}\\
         {\rho~Y_\alpha} &= {\rho~Y_\alpha}_{0}
         + {a_\alpha}{e}^{({c_\alpha}-{r_\alpha})^2},

    where $\mathbf{V}_0$ is the fixed velocity specified by the user at init time,
    and $\gamma$ is taken from the equation-of-state object (eos).

    The user-specified vector of initial values (${{Y}_\alpha}_0$)
    for the mass fraction of each species, *spec_y0s*, and $a_\alpha$ is the
    user-specified vector of amplitudes for each species, *spec_amplitudes*, and
    $c_\alpha$ is the user-specified origin for each species, *spec_centers*.

    A call to this object after creation/init creates the lump solution at a given
    time (*t*) relative to the configured origin (*center*) and background flow
    velocity (*velocity*).

    This object also supplies the exact expected RHS terms from the analytic
    expression via :meth:`exact_rhs`.

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: exact_rhs
    """

    def __init__(
            self, *, dim=1, nspecies=0, rho0=1.0, rho_amp=0., p0=1.0, p_amp=0.,
            center=None, velocity=None,
            spec_y0s=None, spec_amplitudes=None, sigma=1.0):
        r"""Initialize MulticomponentLump parameters.

        Parameters
        ----------
        dim: int
            specify the number of dimensions for the lump
        rho0: float
            specifies the value of $\rho_0$
        p0: float
            specifies the value of $p_0$
        center: numpy.ndarray
            center of lump, shape ``(dim,)``
        velocity: numpy.ndarray
            fixed flow velocity used for exact solution at t != 0,
            shape ``(dim,)``
        sigma: float
            std deviation of the gaussian
        """
        if center is None:
            center = np.zeros(shape=(dim,))
        if velocity is None:
            velocity = np.zeros(shape=(dim,))
        if center.shape != (dim,) or velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional vector inputs.")

        if spec_y0s is None:
            spec_y0s = np.ones(shape=(nspecies,))
        if spec_amplitudes is None:
            spec_amplitudes = np.zeros(shape=(nspecies,))

        if len(spec_y0s) != nspecies or len(spec_amplitudes) != nspecies:
            raise ValueError(f"Expected nspecies={nspecies} inputs.")

        self._nspecies = nspecies
        self._dim = dim
        self._velocity = velocity
        self._center = center
        self._p0 = p0
        self._p_amp = p_amp
        self._rho0 = rho0
        self._rho_amp = rho_amp
        self._spec_y0s = spec_y0s
        self._spec_amplitudes = spec_amplitudes
        self._sigma = sigma

    def __call__(self, x_vec, *, eos=None, time=0, **kwargs):
        """
        Create a multi-component lump solution at time *t* and locations *x_vec*.

        The solution at time *t* is created by advecting the component mass lump
        at the user-specified constant, uniform velocity
        (``MulticomponentLump._velocity``).

        Parameters
        ----------
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """
        t = time
        if x_vec.shape != (self._dim,):
            print(f"len(x_vec) = {len(x_vec)}")
            print(f"self._dim = {self._dim}")
            raise ValueError(f"Expected {self._dim}-dimensional inputs.")

        actx = x_vec[0].array_context

        # coordinates relative to lump center
        lump_loc = self._center + t * self._velocity
        rel_center = make_obj_array(
            [x_vec[i] - lump_loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r2 = np.dot(rel_center, rel_center)/(self._sigma**2)
        expterm = actx.np.exp(-0.5*r2)

        # gaussian in density
        mass = self._rho_amp*expterm + self._rho0
        mom = self._velocity * mass

        # gaussian in species mass fraction
        # process the species components independently
        species_mass = np.empty((self._nspecies,), dtype=object)
        for i in range(self._nspecies):
            species_mass[i] = mass * (self._spec_y0s[i] +
                                      expterm * self._spec_amplitudes[i])

        # gaussian in pressure
        pressure = self._p0 + self._p_amp*expterm
        r = eos.gas_const(species_mass_fractions=species_mass/mass)
        temperature = pressure/mass/r
        energy = mass*(
            eos.get_internal_energy(
                temperature=temperature, species_mass_fractions=species_mass/mass) +
            0.5*np.dot(self._velocity, self._velocity))

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=species_mass)


@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("rho_amp", [0.001, 0., -0.002, -0.005])
@pytest.mark.parametrize("p_amp", [50, 0., -150, -400])
@pytest.mark.parametrize("vmag", [0., 1])
#@pytest.mark.parametrize("order", [1])
#@pytest.mark.parametrize("dim", [2])
#@pytest.mark.parametrize("rho_amp", [0.001])
#@pytest.mark.parametrize("rho_amp", [0.001, 0., -0.002, -0.005])
#@pytest.mark.parametrize("p_amp", [0.])
#@pytest.mark.parametrize("vmag", [0.])
def test_positivity_preserving_limiter_single(actx_factory, order, dim,
                                       rho_amp, p_amp, vmag):
    """Testing positivity-preserving limiter."""
    actx = actx_factory()

    nel_1d = 2

    print(f"{order=}")
    print(f"{dim=}")
    print(f"{rho_amp=}")
    print(f"{p_amp=}")
    print(f"{vmag=}")
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-1.0,) * dim, b=(1.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    dcoll = create_discretization_collection(actx, mesh, order=order)

    # create cells with negative values "eps"
    nodes = actx.thaw(actx.freeze(dcoll.nodes()))
    eos = IdealSingleGas(gas_const=1000)
    print(f"{eos.gamma()=}")
    gas_model = GasModel(eos=eos)
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,)) + vmag
    # Gaussian with a negative peak for rho and pressure

    initializer = MulticomponentLump(dim=dim, p0=100.0, p_amp=p_amp,
                                     rho0=0.001, rho_amp=rho_amp,
                                     center=center, velocity=velocity,
                                     sigma=0.4)

    # think of this as the last known good temperature
    tseed = 100.0
    fluid_cv = initializer(nodes, eos=eos)
    print(f"{actx.to_numpy(fluid_cv.mass)=}")
    temperature = gas_model.eos.temperature(
         cv=fluid_cv, temperature_seed=tseed)
    pressure = gas_model.eos.pressure(
         cv=fluid_cv, temperature=tseed)
    print(f"{actx.to_numpy(pressure)=}")
    print(f"{actx.to_numpy(temperature)=}")
    entropy = actx.np.log(pressure/fluid_cv.mass**1.4)
    print(f"{actx.to_numpy(entropy)=}")

    # apply positivity-preserving limiter
    #
    # test range of density, pressure, test if tseed is None?
    #
    from grudge.dof_desc import DD_VOLUME_ALL
    smin = 11
    entropy_min = actx.np.zeros_like(fluid_cv.mass) + smin
    limited_cv = limit_fluid_state_lv(
        dcoll=dcoll, cv=fluid_cv, temperature_seed=tseed,
        gas_model=gas_model, dd=DD_VOLUME_ALL, entropy_min=entropy_min)
    limited_mass = limited_cv.mass
    print(f"{actx.to_numpy(limited_mass)=}")
    assert actx.to_numpy(actx.np.min(limited_mass)) >= 0.0

    temperature_limited = gas_model.eos.temperature(
         cv=limited_cv, temperature_seed=tseed)
    pressure_limited = gas_model.eos.pressure(
         cv=limited_cv, temperature=tseed)
    entropy_limited = actx.np.log(pressure_limited/limited_cv.mass**1.4)
    print(f"{actx.to_numpy(entropy_limited)=}")
    print(f"{actx.to_numpy(pressure_limited)=}")
    print(f"{actx.to_numpy(temperature_limited)=}")
    assert actx.to_numpy(actx.np.min(entropy_limited)) >= smin - 1.e-4


@pytest.mark.parametrize("order", [1, 4])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("rho_amp", [0.001, 0., -0.002, -0.005])
@pytest.mark.parametrize("p_amp", [50, 0., -150, -400])
@pytest.mark.parametrize("y_amp", [0., .3, 2.])
@pytest.mark.parametrize("vmag", [0., 1])
@pytest.mark.parametrize("nspecies", [2, 7])
#@pytest.mark.parametrize("order", [1])
#@pytest.mark.parametrize("dim", [2])
#@pytest.mark.parametrize("rho_amp", [-0.005])
#@pytest.mark.parametrize("p_amp", [50])
#@pytest.mark.parametrize("y_amp", [0.0])
#@pytest.mark.parametrize("vmag", [0.])
#@pytest.mark.parametrize("nspecies", [2])
def test_positivity_preserving_limiter_multi(actx_factory, order, dim, nspecies,
                                             rho_amp, p_amp, y_amp, vmag):
    """Testing positivity-preserving limiter."""
    actx = actx_factory()

    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-1.0,) * dim, b=(1.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    dcoll = create_discretization_collection(actx, mesh, order=order)

    # create cells with negative values "eps"
    nodes = actx.thaw(actx.freeze(dcoll.nodes()))
    if nspecies == 2:
        eos = IdealSingleGas(gas_const=1000)
    elif nspecies == 7:
        # Pyrometheus initialization
        mech_input = get_mechanism_input("uiuc_7sp")
        cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
        pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
            cantera_soln, temperature_niter=3)(actx.np)
        eos = PyrometheusMixture(pyro_obj, temperature_guess=100)

    gas_model = GasModel(eos=eos)
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,)) + vmag
    spec_y = np.zeros(shape=(nspecies,))
    spec_amp = np.zeros(shape=(nspecies,))
    if nspecies == 2:
        spec_y[0] = .8
        spec_y[1] = .2
        spec_amp[0] = y_amp
        spec_amp[1] = -y_amp
    elif nspecies == 7:
        spec_y[0] = .1
        spec_y[1] = .2
        spec_y[6] = .7
        spec_amp[0] = y_amp
        spec_amp[1] = -y_amp
    # Gaussian with a negative peak for rho and pressure

    # use an unperturbed state to get a temperature seed
    tseed = 100.*(1 + actx.np.zeros_like(nodes[0]))
    if nspecies == 7:
        initializer_unperturbed = MulticomponentLump(dim=dim, nspecies=nspecies,
                                         p0=100.0, rho0=0.001, spec_y0s=spec_y,
                                         center=center, velocity=velocity, sigma=0.4)
        fluid_cv_unp = initializer_unperturbed(nodes, eos=eos)
        temperature_unp = gas_model.eos.temperature(
             cv=fluid_cv_unp, temperature_seed=tseed)
        pressure_unp = gas_model.eos.pressure(
             cv=fluid_cv_unp, temperature=tseed)
        print(f"{fluid_cv_unp=}")
        print(f"{pressure_unp=}")
        print(f"{temperature_unp=}")
        tseed = temperature_unp

    # gets a negative pressure and pres_avg is effed
    initializer = MulticomponentLump(dim=dim, nspecies=nspecies,
                                     p0=100.0, p_amp=p_amp,
                                     rho0=0.001, rho_amp=rho_amp,
                                     spec_y0s=spec_y, spec_amplitudes=spec_amp,
                                     center=center, velocity=velocity, sigma=0.4)

    # think of this as the last known good temperature
    fluid_cv = initializer(nodes, eos=eos)
    nelem = 8
    ndof = 3
    #group = dcoll.discr_from_dd(dd_vol_fluid).mesh.groups
    from grudge.dof_desc import DD_VOLUME_ALL
    discr = dcoll.discr_from_dd(DD_VOLUME_ALL)
    nelem = discr.groups[0].nelements
    ndof = discr.groups[0].nunit_dofs

    from meshmode.dof_array import DOFArray
    el_indicies = DOFArray(actx, data=(actx.from_numpy(np.outer(
        np.indices((nelem,)), np.ones(ndof))),))
    print(f"{actx.to_numpy(el_indicies)=}")
    print(f"{actx.to_numpy(nodes)=}")
    print(f"{actx.to_numpy(fluid_cv.mass)=}")
    print(f"{actx.to_numpy(fluid_cv.species_mass_fractions)=}")
    temperature = gas_model.eos.temperature(
         cv=fluid_cv, temperature_seed=tseed)
    pressure = gas_model.eos.pressure(
         cv=fluid_cv, temperature=tseed)
    print(f"{actx.to_numpy(pressure)=}")
    print(f"{actx.to_numpy(temperature)=}")
    entropy = actx.np.log(pressure/fluid_cv.mass**1.4)
    print(f"{actx.to_numpy(entropy)=}")

    # apply positivity-preserving limiter
    #
    # test range of density, pressure, test if tseed is None?
    #
    from grudge.dof_desc import DD_VOLUME_ALL
    smin = 11
    entropy_min = actx.np.zeros_like(fluid_cv.mass) + smin
    limited_cv = limit_fluid_state_lv(
        dcoll=dcoll, cv=fluid_cv, temperature_seed=tseed,
        gas_model=gas_model, dd=DD_VOLUME_ALL, entropy_min=entropy_min)
    limited_mass = limited_cv.mass
    print(f"{actx.to_numpy(limited_mass)=}")
    assert actx.to_numpy(actx.np.min(limited_mass)) >= 0.0

    temperature_limited = gas_model.eos.temperature(
         cv=limited_cv, temperature_seed=tseed)
    pressure_limited = gas_model.eos.pressure(
         cv=limited_cv, temperature=tseed)
    entropy_limited = actx.np.log(pressure_limited/limited_cv.mass**1.4)
    print(f"{actx.to_numpy(entropy_limited)=}")
    print(f"{actx.to_numpy(pressure_limited)=}")
    print(f"{actx.to_numpy(temperature_limited)=}")
    assert actx.to_numpy(actx.np.min(entropy_limited)) >= smin - 1.e-4

    limited_mass_frac = limited_cv.species_mass_fractions

    print(f"{actx.to_numpy(limited_mass_frac)=}")
    # check minimum and maximum
    spec_tol_min = 1e-16
    spec_tol_max = 2e-16
    for i in range(nspecies):
        assert actx.to_numpy(actx.np.min(limited_mass_frac[i])) > -spec_tol_min
        assert actx.to_numpy(actx.np.max(limited_mass_frac[i])) < 1.0 + spec_tol_max

    # check y sums to 1
    spec_sum_tol = 1e-15
    y_sum = actx.np.zeros_like(limited_cv.mass)
    for i in range(nspecies):
        y_sum = y_sum + limited_mass_frac[i]
    y_sum_m1 = actx.np.abs(y_sum - 1.0)
    assert actx.to_numpy(actx.np.max(y_sum_m1)) < spec_sum_tol

    y_ones = 0.*y_sum + 1.0
    y_sum_mone = y_sum - y_ones
    print(f"{actx.to_numpy(y_sum)=}")
    assert actx.to_numpy(actx.np.max(actx.np.abs(y_sum_mone))) < 1e-11

    # check pressure positivity
    assert actx.to_numpy(actx.np.min(pressure_limited)) > 0.

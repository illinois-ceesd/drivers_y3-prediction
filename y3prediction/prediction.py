"""mirgecom driver for the Y2 prediction."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

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
import logging
import sys
import pickle
import os
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
import math
from dataclasses import replace
import grudge.op as op
from pytools.obj_array import make_obj_array
from functools import partial
from mirgecom.discretization import create_discretization_collection

from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    VolumeDomainTag,
    BoundaryDomainTag,
    DOFDesc,
    DISCR_TAG_BASE,
    DD_VOLUME_ALL
)
from grudge.op import nodal_max, nodal_min
from grudge.trace_pair import inter_volume_trace_pairs
from grudge.discretization import filter_part_boundaries
from grudge.trace_pair import TracePair
from grudge.geometry.metrics import normal as normal_vector
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage,
    logmgr_add_mempool_usage,
)

from mirgecom.simutil import (
    SimulationConfigurationError,
    check_step,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
)
from mirgecom.utils import force_evaluation
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step, ssprk43_step)
from mirgecom.inviscid import (inviscid_facial_flux_rusanov,
                               inviscid_facial_flux_hll)
from mirgecom.viscous import (viscous_facial_flux_central,
                              viscous_facial_flux_harmonic)
from grudge.shortcuts import compiled_lsrk45_step

from mirgecom.fluid import (
    make_conserved,
    velocity_gradient,
    species_mass_fraction_gradient
)
from mirgecom.limiter import (bound_preserving_limiter)
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    IsothermalWallBoundary,
    IsothermalSlipWallBoundary,
    AdiabaticSlipBoundary,
    AdiabaticNoslipWallBoundary,
    PressureOutflowBoundary,
    DummyBoundary
)
from mirgecom.diffusion import (
    diffusion_operator,
    grad_operator as wall_grad_t_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary
)
from mirgecom.initializers import Uniform, MulticomponentLump
from mirgecom.eos import (
    IdealSingleGas, PyrometheusMixture,
    MixtureDependentVars, GasDependentVars
)
from mirgecom.transport import (SimpleTransport,
                                PowerLawTransport,
                                ArtificialViscosityTransportDiv,
                                ArtificialViscosityTransportDiv2,
                                ArtificialViscosityTransportDiv3)
from mirgecom.gas_model import (
    GasModel,
    make_fluid_state,
    replace_fluid_state,
    make_operator_fluid_states,
    project_fluid_state
)
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    add_interface_boundaries_no_grad,
    add_interface_boundaries
)
from mirgecom.navierstokes import (
    grad_cv_operator,
    grad_t_operator as fluid_grad_t_operator,
    ns_operator as general_ns_operator
)
from mirgecom.artificial_viscosity import smoothness_indicator
# driver specific utilties
from y3prediction.utils import (
    IsentropicInflow,
    getIsentropicPressure,
    getIsentropicTemperature,
    getMachFromAreaRatio
)
from y3prediction.wall import (
    mask_from_elements,
    WallVars,
    WallModel,
)
from y3prediction.shock1d import PlanarDiscontinuityMulti

from dataclasses import dataclass
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic
)
from mirgecom.fluid import ConservedVars
from meshmode.dof_array import DOFArray  # noqa
from grudge.dof_desc import DISCR_TAG_MODAL
from meshmode.transform_metadata import FirstAxisIsElementsTag
from arraycontext import outer
from grudge.trace_pair import interior_trace_pairs, tracepair_with_discr_tag
from meshmode.discretization.connection import FACE_RESTR_ALL
from mirgecom.flux import num_flux_central


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class StepperState:
    r"""Store quantities to advance in time.

    Store the quanitites that should be evolved in time by an advancer
    """

    cv: ConservedVars
    tseed: DOFArray
    av_smu: DOFArray
    av_sbeta: DOFArray
    av_skappa: DOFArray
    av_sd: DOFArray
    smin: DOFArray

    __array_ufunc__ = None

    def replace(self, **kwargs):
        """Return a copy of *self* with the attributes in *kwargs* replaced."""
        return replace(self, **kwargs)

    def get_obj_array(self):
        """Return an object array containing all the stored quantitines."""
        return make_obj_array([self.cv, self.tseed,
                               self.av_smu, self.av_sbeta,
                               self.av_skappa, self.av_sd,
                               self.smin])


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallStepperState(StepperState):
    r"""Store quantities to advance in time.

    Store the quanitites that should be evolved in time by an advancer
    Adding WallVars
    """

    wv: WallVars

    __array_ufunc__ = None

    def get_obj_array(self):
        """Return an object array containing all the stored quantitines."""
        return make_obj_array([self.cv, self.tseed,
                               self.av_smu, self.av_sbeta,
                               self.av_skappa, self.av_sd,
                               self.smin,
                               self.wv])


def make_stepper_state(cv, tseed, av_smu, av_sbeta, av_skappa, av_sd, smin, wv=None):
    if wv is not None:
        return WallStepperState(cv=cv, tseed=tseed, av_smu=av_smu,
                                av_sbeta=av_sbeta, av_skappa=av_skappa,
                                av_sd=av_sd, smin=smin, wv=wv)
    else:
        return StepperState(cv=cv, tseed=tseed, av_smu=av_smu,
                            av_sbeta=av_sbeta, av_skappa=av_skappa,
                            av_sd=av_sd, smin=smin)


def make_stepper_state_obj(ary):
    if ary.size > 7:
        return WallStepperState(cv=ary[0], tseed=ary[1], av_smu=ary[2],
                                av_sbeta=ary[3], av_skappa=ary[4],
                                av_sd=ary[5], smin=ary[6], wv=ary[7])
    else:
        return StepperState(cv=ary[0], tseed=ary[1], av_smu=ary[2],
                            av_sbeta=ary[3], av_skappa=ary[4],
                            av_sd=ary[5], smin=ary[6])


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class _FluidAvgCVTag:
    pass


class _InitCommTag:
    pass


class _SmoothnessCVGradCommTag:
    pass


class _OxCommTag:
    pass


class _FluidOxDiffCommTag:
    pass


class _WallOxDiffCommTag:
    pass


class _SmoothDiffCommTag:
    pass


class _SmoothCharDiffCommTag:
    pass


class _SmoothCharDiffFluidCommTag:
    pass


class _SmoothCharDiffWallCommTag:
    pass


class _BetaDiffCommTag:
    pass


class _BetaDiffWallCommTag:
    pass


class _BetaDiffFluidCommTag:
    pass


class _KappaDiffCommTag:
    pass


class _KappaDiffWallCommTag:
    pass


class _KappaDiffFluidCommTag:
    pass


class _MuDiffCommTag:
    pass


class _DDiffFluidCommTag:
    pass


class _MuDiffWallCommTag:
    pass


class _MuDiffFluidCommTag:
    pass


class _WallOperatorCommTag:
    pass


class _FluidOperatorCommTag:
    pass


class _UpdateCoupledBoundariesCommTag:
    pass


class _FluidOpStatesCommTag:
    pass


class _MyGradTag1:
    pass


class _MyGradTag2:
    pass


class _MyGradTag3:
    pass


class _MyGradTag4:
    pass


class _MyGradTag5:
    pass


class _MyGradTag6:
    pass


def my_derivative_function(
        dcoll, field, field_bounds, quadrature_tag, dd_vol,
        bnd_cond, comm_tag):

    actx = field.array_context
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    interp_to_surf_quad = partial(
        tracepair_with_discr_tag, dcoll, quadrature_tag)

    def interior_flux(field_tpair):
        dd_trace_quad = field_tpair.dd.with_discr_tag(quadrature_tag)
        #normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
        normal_quad = normal_vector(actx, dcoll, dd_trace_quad)
        bnd_tpair_quad = interp_to_surf_quad(field_tpair)
        flux_int = outer(
            num_flux_central(bnd_tpair_quad.int, bnd_tpair_quad.ext),
            normal_quad)

        return op.project(dcoll, dd_trace_quad, dd_allfaces_quad, flux_int)

    def boundary_flux(bdtag, bdry):
        if isinstance(bdtag, DOFDesc):
            bdtag = bdtag.domain_tag
        dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
        normal_quad = normal_vector(actx, dcoll, dd_bdry_quad)
        int_soln_quad = op.project(dcoll, dd_vol, dd_bdry_quad, field)

        # MJA, not sure about this
        if bnd_cond == "symmetry" and bdtag == "symmetry":
            ext_soln_quad = 0.0*int_soln_quad
        else:
            ext_soln_quad = 1.0*int_soln_quad

        bnd_tpair = TracePair(bdtag, interior=int_soln_quad,
                              exterior=ext_soln_quad)
        flux_bnd = outer(
            num_flux_central(bnd_tpair.int, bnd_tpair.ext), normal_quad)

        return op.project(dcoll, dd_bdry_quad, dd_allfaces_quad, flux_bnd)

    field_quad = op.project(dcoll, dd_vol, dd_vol_quad, field)

    return -1.0*op.inverse_mass(
        dcoll, dd_vol_quad,
        op.weak_local_grad(dcoll, dd_vol_quad, field_quad)
        -  # noqa: W504
        op.face_mass(
            dcoll, dd_allfaces_quad,
            sum(
                interior_flux(u_tpair) for u_tpair in interior_trace_pairs(
                    dcoll, field, volume_dd=dd_vol, comm_tag=comm_tag))
            + sum(
                 boundary_flux(bdtag, bdry)
                 for bdtag, bdry in field_bounds.items())
        )
    )


def axisym_source_fluid(dcoll, fluid_state, fluid_nodes, gas_model,
                        quadrature_tag, dd_vol_fluid, boundaries, grad_cv, grad_t):
    cv = fluid_state.cv
    dv = fluid_state.dv
    actx = cv.array_context

    mu = fluid_state.tv.viscosity
    beta = gas_model.transport.volume_viscosity(cv, dv, gas_model.eos)
    kappa = fluid_state.tv.thermal_conductivity
    d_ij = fluid_state.tv.species_diffusivity

    grad_v = velocity_gradient(cv, grad_cv)
    grad_y = species_mass_fraction_gradient(cv, grad_cv)

    u = cv.velocity[0]
    v = cv.velocity[1]

    dudr = grad_v[0][0]
    dudy = grad_v[0][1]
    dvdr = grad_v[1][0]
    dvdy = grad_v[1][1]

    drhoudr = (grad_cv.momentum[0])[0]

    #d2udr2 = my_derivative_function(dcoll,  dudr, boundaries, dd_vol_fluid,
    #                                "replicate", comm_tag=_MyGradTag1)[0]
    d2vdr2 = my_derivative_function(dcoll, dvdr, boundaries, quadrature_tag,
                                    dd_vol_fluid, "replicate",
                                    comm_tag=_MyGradTag2)[0]
    d2udrdy = my_derivative_function(dcoll, dudy, boundaries, quadrature_tag,
                                     dd_vol_fluid, "replicate",
                                     comm_tag=_MyGradTag3)[0]
    dmudr = my_derivative_function(dcoll, mu, boundaries, quadrature_tag,
                                   dd_vol_fluid, "replicate",
                                   comm_tag=_MyGradTag4)[0]
    dbetadr = my_derivative_function(dcoll, beta, boundaries, quadrature_tag,
                                     dd_vol_fluid, "replicate",
                                     comm_tag=_MyGradTag5)[0]
    dbetady = my_derivative_function(dcoll, beta, boundaries, quadrature_tag,
                                     dd_vol_fluid, "replicate",
                                     comm_tag=_MyGradTag6)[1]

    qr = -(kappa*grad_t)[0]
    dqrdr = 0.0
    off_axis_x = 1.e-7
    fluid_nodes_are_off_axis = actx.np.greater(fluid_nodes[0], off_axis_x)

    dyidr = grad_y[:, 0]
    #dyi2dr2 = my_derivative_function(dcoll, dyidr, 'replicate')[:,0]

    tau_ry = 1.0*mu*(dudy + dvdr)
    tau_rr = 2.0*mu*dudr + beta*(dudr + dvdy)
    #tau_yy = 2.0*mu*dvdy + beta*(dudr + dvdy)
    tau_tt = beta*(dudr + dvdy) + 2.0*mu*actx.np.where(
        fluid_nodes_are_off_axis, u/fluid_nodes[0], dudr)

    dtaurydr = dmudr*dudy + mu*d2udrdy + dmudr*dvdr + mu*d2vdr2

    source_mass_dom = - cv.momentum[0]

    source_rhoU_dom = - cv.momentum[0]*u \
                      + tau_rr - tau_tt \
                      + u*dbetadr + beta*dudr \
                      + beta*actx.np.where(
                          fluid_nodes_are_off_axis, -u/fluid_nodes[0], -dudr)

    source_rhoV_dom = - cv.momentum[0]*v \
                      + tau_ry \
                      + u*dbetady + beta*dudy

    # FIXME add species diffusion term
    source_rhoE_dom = -((cv.energy+dv.pressure)*u + qr) \
                      + u*tau_rr + v*tau_ry \
                      + u**2*dbetadr + beta*2.0*u*dudr \
                      + u*v*dbetady + u*beta*dvdy + v*beta*dudy

    source_spec_dom = - cv.species_mass*u + cv.mass*d_ij*dyidr

    source_mass_sng = - drhoudr
    source_rhoU_sng = 0.0
    source_rhoV_sng = - v*drhoudr + dtaurydr + beta*d2udrdy + dudr*dbetady
    source_rhoE_sng = -((cv.energy + dv.pressure)*dudr + dqrdr) \
                            + tau_rr*dudr + v*dtaurydr \
                            + 2.0*beta*dudr**2 \
                            + beta*dudr*dvdy \
                            + v*dudr*dbetady \
                            + v*beta*d2udrdy
    #source_spec_sng = - cv.species_mass*dudr + d_ij*dyidr
    source_spec_sng = - cv.species_mass*dudr

    source_mass = actx.np.where(
        fluid_nodes_are_off_axis, source_mass_dom/fluid_nodes[0],
        source_mass_sng)
    source_rhoU = actx.np.where(
        fluid_nodes_are_off_axis, source_rhoU_dom/fluid_nodes[0],
        source_rhoU_sng)
    source_rhoV = actx.np.where(
        fluid_nodes_are_off_axis, source_rhoV_dom/fluid_nodes[0],
        source_rhoV_sng)
    source_rhoE = actx.np.where(
        fluid_nodes_are_off_axis, source_rhoE_dom/fluid_nodes[0],
        source_rhoE_sng)

    source_spec = make_obj_array([
                  actx.np.where(
                      fluid_nodes_are_off_axis,
                      source_spec_dom[i]/fluid_nodes[0],
                      source_spec_sng[i])
                  for i in range(cv.nspecies)])

    return make_conserved(dim=2, mass=source_mass, energy=source_rhoE,
                   momentum=make_obj_array([source_rhoU, source_rhoV]),
                   species_mass=source_spec)


def axisym_source_wall(dcoll, wall_nodes, wall_temperature, thermal_conductivity,
                       boundaries, grad_t):
    #dkappadr = 0.0*wall_nodes[0]

    actx = wall_temperature.array_context
    off_axis_x = 1.e-7
    wall_nodes_are_off_axis = actx.np.greater(wall_nodes[0], off_axis_x)
    kappa = thermal_conductivity
    qr = - (kappa*grad_t)[0]
    #d2Tdr2  = my_derivative_function(dcoll, grad_t[0], boundaries,
    #                                 dd_vol_wall, "symmetry")[0]
    #dqrdr = - (dkappadr*grad_t[0] + kappa*d2Tdr2)

    source_rhoE_dom = - qr
    source_rhoE_sng = 0.0
    source_rhoE = actx.np.where(
        wall_nodes_are_off_axis, source_rhoE_dom/wall_nodes[0], source_rhoE_sng)

    return source_rhoE


def make_coupled_operator_fluid_states(
        dcoll, fluid_state, gas_model, boundaries, fluid_dd, wall_dd,
        quadrature_tag=DISCR_TAG_BASE, comm_tag=None, limiter_func=None,
        entropy_min=None):
    """Prepare gas model-consistent fluid states for use in coupled operators."""

    all_boundaries = {}
    all_boundaries.update(boundaries)
    all_boundaries.update({
        dd_bdry.domain_tag: None  # Don't need the full boundaries, just the tags
        for dd_bdry in filter_part_boundaries(
            dcoll, volume_dd=fluid_dd, neighbor_volume_dd=wall_dd)})

    return make_operator_fluid_states(
        dcoll, fluid_state, gas_model, all_boundaries, quadrature_tag,
        dd=fluid_dd, comm_tag=comm_tag, limiter_func=limiter_func,
        entropy_min=entropy_min)


def coupled_grad_operator(
        dcoll,
        gas_model,
        fluid_dd, wall_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_kappa, wall_temperature,
        *,
        time=0.,
        interface_noslip=True,
        quadrature_tag=DISCR_TAG_BASE,
        fluid_operator_states_quad=None,
        limiter_func=None,
        entropy_min=None,
        comm_tag=None):
    """Compute the gradients for the coupled fluid/wall."""

    # Insert the interface boundaries for computing the gradient
    fluid_all_boundaries_no_grad, wall_all_boundaries_no_grad = \
        add_interface_boundaries_no_grad(
            dcoll=dcoll,
            gas_model=gas_model,
            fluid_dd=fluid_dd,
            wall_dd=wall_dd,
            fluid_state=fluid_state,
            wall_kappa=wall_kappa,
            wall_temperature=wall_temperature,
            fluid_boundaries=fluid_boundaries,
            wall_boundaries=wall_boundaries,
            interface_noslip=interface_noslip,
            #interface_radiation,
            quadrature_tag=quadrature_tag,
            comm_tag=comm_tag)

    # Get the operator fluid states
    # Note: Don't need to use the make_coupled_* version here because we're passing
    # in the augmented boundaries
    if fluid_operator_states_quad is None:
        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, fluid_all_boundaries_no_grad,
            quadrature_tag, dd=fluid_dd, limiter_func=limiter_func,
            entropy_min=entropy_min,
            comm_tag=(comm_tag, _FluidOpStatesCommTag))

    # Compute the gradient operators for both subdomains
    fluid_grad_cv = grad_cv_operator(
        dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
        dd=fluid_dd, time=time, quadrature_tag=quadrature_tag,
        operator_states_quad=fluid_operator_states_quad,
        entropy_min=entropy_min,
        comm_tag=comm_tag)

    fluid_grad_temperature = fluid_grad_t_operator(
        dcoll, gas_model, fluid_all_boundaries_no_grad, fluid_state,
        time=time, quadrature_tag=quadrature_tag,
        limiter_func=limiter_func, entropy_min=entropy_min,
        dd=fluid_dd, operator_states_quad=fluid_operator_states_quad)

    wall_grad_temperature = wall_grad_t_operator(
        dcoll, wall_kappa, wall_all_boundaries_no_grad, wall_temperature,
        quadrature_tag=quadrature_tag, dd=wall_dd)

    return fluid_grad_cv, fluid_grad_temperature, wall_grad_temperature


def coupled_ns_heat_operator(
        dcoll,
        gas_model,
        fluid_boundaries,
        wall_boundaries,
        fluid_dd, wall_dd,
        fluid_state,
        wall_kappa, wall_temperature,
        fluid_grad_cv,
        fluid_grad_temperature,
        wall_grad_temperature,
        *,
        time=0.,
        interface_noslip=True,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE,
        fluid_operator_states_quad=None,
        limiter_func=None,
        ns_operator=general_ns_operator,
        entropy_min=None,
        comm_tag=None,
        axisymmetric=False,
        fluid_nodes=None,
        wall_nodes=None):
    """Compute the NS and heat operators for the coupled fluid/wall."""

    # Insert the interface boundaries
    fluid_all_boundaries, wall_all_boundaries = \
        add_interface_boundaries(
            dcoll=dcoll,
            gas_model=gas_model,
            fluid_dd=fluid_dd, wall_dd=wall_dd,
            fluid_state=fluid_state,
            wall_kappa=wall_kappa,
            wall_temperature=wall_temperature,
            fluid_grad_temperature=fluid_grad_temperature,
            wall_grad_temperature=wall_grad_temperature,
            fluid_boundaries=fluid_boundaries,
            wall_boundaries=wall_boundaries,
            interface_noslip=interface_noslip,
            wall_penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag,
            comm_tag=comm_tag)

    # Get the operator fluid states with the updated boundaries
    if fluid_operator_states_quad is None:
        fluid_operator_states_quad = make_operator_fluid_states(
            dcoll, fluid_state, gas_model, fluid_all_boundaries,
            quadrature_tag, dd=fluid_dd, limiter_func=limiter_func,
            entropy_min=entropy_min,
            comm_tag=(comm_tag, _FluidOpStatesCommTag))

    ns_result = ns_operator(
        dcoll=dcoll,
        gas_model=gas_model,
        dd=fluid_dd,
        operator_states_quad=fluid_operator_states_quad,
        limiter_func=limiter_func, entropy_min=entropy_min,
        grad_cv=fluid_grad_cv,
        grad_t=fluid_grad_temperature,
        boundaries=fluid_all_boundaries,
        state=fluid_state,
        time=time,
        quadrature_tag=quadrature_tag,
        comm_tag=(comm_tag, _FluidOperatorCommTag))

    diff_result = diffusion_operator(
        dcoll=dcoll,
        kappa=wall_kappa,
        boundaries=wall_all_boundaries,
        u=wall_temperature,
        penalty_amount=wall_penalty_amount,
        quadrature_tag=quadrature_tag,
        dd=wall_dd,
        grad_u=wall_grad_temperature,
        comm_tag=(comm_tag, _WallOperatorCommTag))

    if axisymmetric is True:
        ns_result = ns_result + \
            axisym_source_fluid(dcoll=dcoll,
                                fluid_nodes=fluid_nodes,
                                fluid_state=fluid_state,
                                dd_vol_fluid=fluid_dd,
                                gas_model=gas_model,
                                quadrature_tag=quadrature_tag,
                                boundaries=fluid_all_boundaries,
                                grad_cv=fluid_grad_cv, grad_t=fluid_grad_temperature)

    if axisymmetric is True:
        diff_result = diff_result + \
            axisym_source_wall(dcoll=dcoll,
                               wall_nodes=wall_nodes,
                               wall_temperature=wall_temperature,
                               thermal_conductivity=wall_kappa,
                               boundaries=wall_all_boundaries,
                               grad_t=wall_grad_temperature)

    return ns_result, diff_result


def limit_fluid_state(dcoll, cv, temperature_seed, gas_model, dd):

    actx = cv.array_context
    nspecies = cv.nspecies
    dim = cv.dim

    temperature = gas_model.eos.temperature(
        cv=cv, temperature_seed=temperature_seed)
    pressure = gas_model.eos.pressure(cv=cv, temperature=temperature)

    spec_lim = make_obj_array([
        bound_preserving_limiter(dcoll=dcoll, dd=dd,
                                 field=cv.species_mass_fractions[i],
                                 mmin=0.0, mmax=1.0, modify_average=True)
        for i in range(nspecies)
    ])

    # limit the sum to 1.0
    aux = actx.np.zeros_like(cv.mass)
    for i in range(0, nspecies):
        aux = aux + spec_lim[i]
    spec_lim = spec_lim/aux

    kin_energy = 0.5*np.dot(cv.velocity, cv.velocity)

    mass_lim = gas_model.eos.get_density(
        pressure=pressure, temperature=temperature,
        species_mass_fractions=cv.species_mass_fractions)
    energy_lim = mass_lim*(
        gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
        + kin_energy
    )
    mom_lim = mass_lim*cv.velocity

    cv_lim = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                            momentum=mom_lim,
                            species_mass=mass_lim*spec_lim)

    return make_obj_array([cv_lim, pressure, temperature])


def element_average(dcoll, dd, field, volumes=None):
    # Compute cell averages of the state

    actx = field.array_context
    cell_avgs = op.elementwise_integral(dcoll, dd, field)
    if volumes is None:
        volumes = abs(op.elementwise_integral(
            dcoll, dd, actx.np.zeros_like(field) + 1.0))

    return cell_avgs/volumes


def _element_average_cv(dcoll, dd, cv, volumes=None):

    nspecies = cv.nspecies
    dim = cv.dim

    density = element_average(dcoll, dd, cv.mass, volumes)
    momentum = make_obj_array([
        element_average(dcoll, dd, cv.momentum[i], volumes)
        for i in range(dim)])
    energy = element_average(dcoll, dd, cv.energy, volumes)

    species_mass = None
    if nspecies > 0:
        species_mass = make_obj_array([
            element_average(dcoll, dd, cv.species_mass[i], volumes)
            for i in range(0, nspecies)])

    # make a new CV with the limited variables
    return make_conserved(dim=dim, mass=density, energy=energy,
                          momentum=momentum, species_mass=species_mass)


def limit_fluid_state_liu(dcoll, cv, temperature_seed, gas_model, dd):
    r"""lement average positivity preserving limiter

    Follows loosely the implementation outline in Liu, et. al.
    Limits the density and mass fractions based on global minima
    then computes an average fluid state and uses the averge pressure
    to limit the entire cv in regions with very small pressures
    """

    actx = cv.array_context
    nspecies = cv.nspecies
    dim = cv.dim

    rho_lim = 1.e-10
    pres_lim = 1.0

    elem_avg_cv = _element_average_cv(dcoll, cv, dd)

    # 1.0 limit the density
    theta_rho = actx.np.abs((elem_avg_cv.mass - rho_lim) /
                            (elem_avg_cv.mass - cv.mass + 1.e-13))

    # only apply limiting when theta < 1
    mass_lim = actx.np.where(actx.np.less(theta_rho, 1.0),
        theta_rho*cv.mass + (1 - theta_rho)*elem_avg_cv.mass, cv.mass)

    # 2.0 limit the species mass fractions
    spec_mass_lim = cv.species_mass
    theta_rhoY = actx.np.zeros_like(cv.species_mass)
    for i in range(0, nspecies):
        theta_rhoY[i] = actx.np.abs(
            (elem_avg_cv.species_mass) /
            (elem_avg_cv.species_mass - cv.species_mass + 1.e-13))

        # only apply limiting when theta < 1
        spec_mass_lim = actx.np.where(
            actx.np.less(theta_rhoY, 1.0),
            theta_rho*cv.species_mass + (1 - theta_rho)*elem_avg_cv.mass,
            cv.species_mass)

    # 3.0 reconstruct cv and find the average element cv and pressure
    #
    # Question:
    # we don't update the energy or the momentum here
    # so if the density is reduced it results in a
    #    net decrease in the pressure and increase in velocity?
    cv_updated = make_conserved(dim=dim, mass=mass_lim, energy=cv.energy,
                                momentum=cv.momentum,
                                species_mass=spec_mass_lim)
    temperature_updated = gas_model.eos.temperature(
        cv=cv_updated, temperature_seed=temperature_seed)
    pressure_updated = gas_model.eos.pressure(cv=cv_updated,
                                              temperature=temperature_updated)

    elem_avg_temp = gas_model.eos.temperature(
        cv=elem_avg_cv, temperature_seed=temperature_updated)
    elem_avg_pres = gas_model.eos.pressure(
        cv=elem_avg_cv, temperature=elem_avg_temp)

    mmin_i = op.elementwise_min(dcoll, dd, pressure_updated)
    mmin = pres_lim

    _theta = actx.np.minimum(
        1.0, actx.np.where(actx.np.less(mmin_i, mmin),
        abs((mmin-elem_avg_pres)/(mmin_i-elem_avg_pres+1e-13)), 1.0)
    )

    # 4.0 limit cv where the pressure is negative
    #     this is turn keeps the pressure positive

    mass_lim = (_theta*(cv_updated.mass - elem_avg_cv.mass)
        + elem_avg_cv.mass)
    mom_lim = make_obj_array([
        (_theta*(cv_updated.momentum[i] - elem_avg_cv.momentum[i])
         + elem_avg_cv.momentum[i])
        for i in range(dim)
    ])
    energy_lim = (_theta*(cv_updated.energy - elem_avg_cv.energy)
        + elem_avg_cv.energy)
    spec_lim = make_obj_array([
        (_theta*(cv_updated.species_mass[i] - elem_avg_cv.species_mass[i])
         + elem_avg_cv.species_mass[i])
        for i in range(0, nspecies)
    ])

    cv_lim = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                            momentum=mom_lim,
                            species_mass=spec_lim)

    return cv_lim


def hammer_species(cv):
    # limit the species mass fraction (Y) to [0, 1.0] and
    # sum(Y) = 1.0
    actx = cv.array_context
    aux = actx.np.zeros_like(cv.mass)
    zeros_spec = actx.np.zeros_like(cv.mass)
    ones_spec = zeros_spec + 1.0
    spec_lim = cv.species_mass_fractions
    for i in range(cv.nspecies):
        spec_lim[i] = actx.np.where(actx.np.less(spec_lim[i], zeros_spec),
                                    zeros_spec, spec_lim[i])
        spec_lim[i] = actx.np.where(actx.np.greater(spec_lim[i], ones_spec),
                                    ones_spec, spec_lim[i])
        aux = aux + spec_lim[i]

    spec_lim = spec_lim / aux
    # return cv.replace(species_mass=cv.mass*spec_lim)
    return make_conserved(dim=cv.dim, mass=cv.mass,
                          momentum=cv.momentum,
                          energy=cv.energy,
                          species_mass=cv.mass*spec_lim)


def limit_fluid_state_lv(dcoll, cv, temperature_seed, entropy_min,
                         gas_model, dd, viz_theta=False):
    r"""Entropy-based positivity preserving limiter

    Follows loosely the implementation outline in Lv, et. al.
    Limits the density, entropy, and mass fractions locally
    based on average states.
    """

    actx = cv.array_context
    nspecies = cv.nspecies
    dim = cv.dim
    do_limit_toler = 1.e-9
    min_allowed_density = 1.e-7
    min_allowed_pressure = 1.e-12

    ones = 1. + actx.np.zeros_like(cv.mass)
    element_vols = abs(op.elementwise_integral(dcoll, dd,
                                               actx.np.zeros_like(cv.mass) + 1.0))

    rank = 0
    print_stuff = False

    if print_stuff is True:
        print("Start of limiting")
        print(f"{dd=}")
        print(f"{dd.domain_tag=}")
        print(f"{dd.domain_tag.tag=}")
        print(f"{type(dd.domain_tag.tag)=}")
        print(f"{cv.mass[0].shape=} elements in discretization")

        my_rank = dcoll.mpi_communicator.Get_rank()

    print_all_nodes = False
    index = [0]
    from meshmode.discretization.connection.face import FACE_RESTR_INTERIOR
    if print_stuff is True and isinstance(dd.domain_tag, VolumeDomainTag):
        index = [6000]
        print(f"volume limiter {rank=}, element index {index}")
    elif print_stuff is True and (isinstance(dd.domain_tag, BoundaryDomainTag) and
          dd.domain_tag.tag == "noslip_wall"):
        index = [0]
        print(f"isothermal_wall limiter {rank=}, element index {index}")
        print_all_nodes = False
    elif print_stuff is True and (isinstance(dd.domain_tag, BoundaryDomainTag) and
          #isinstance(dd.domain_tag.tag, FACE_RESTR_INTERIOR)):
          dd.domain_tag.tag is FACE_RESTR_INTERIOR):
        index = [76755, 5940, 80880]
        for ind in index:
            print(f"interior volume faces {rank=}, element index {ind}")
        print_all_nodes = False
    else:
        print_stuff = False

    if print_stuff is True:
        print("bbbb")
        np.set_printoptions(threshold=sys.maxsize, precision=16)
        print(f"{dd.domain_tag=}")
        print(f"{dd.domain_tag.tag=}")
        #print(f"{cv.mass=}")
        #data = actx.to_numpy(cv.mass)
        #print(f"cv.mass \n {data[0]}")

        # get the location of the indicies, useful for comparing different dd's
        limiter_fluid_nodes = force_evaluation(actx, actx.thaw(dcoll.nodes(dd)))
        nodes_x = actx.to_numpy(limiter_fluid_nodes)[0]
        nodes_y = actx.to_numpy(limiter_fluid_nodes)[1]
        if print_all_nodes is True:
            print(f"element location x {nodes_x[0]=}")
            print(f"element location y {nodes_y[0]=}")
        else:
            for ind in index:
                print(f"element {ind} location x {nodes_x[0][ind]=}")
                print(f"element {ind} location y {nodes_y[0][ind]=}")
        #if (isinstance(dd.domain_tag, BoundaryDomainTag) and
              #dd.domain_tag.tag == "isothermal_wall"):
            #print(f"element location x {nodes_x[0]=}")
            #print(f"element location y {nodes_y[0]=}")
        print("eeee")

    if print_stuff is True:
        temperature_initial = gas_model.eos.temperature(
            cv=cv, temperature_seed=temperature_seed)
        pressure_initial = gas_model.eos.pressure(
            cv=cv, temperature=temperature_initial)
        # initial state
        for ind in index:
            print(f"element {ind}")
            data = actx.to_numpy(cv.mass)
            print(f"cv.mass \n {data[0][ind]}")
            data = actx.to_numpy(temperature_initial)
            print(f"temperature_initial \n {data[0][ind]}")
            data = actx.to_numpy(pressure_initial)
            print(f"pressure_initial \n {data[0][ind]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(cv.species_mass_fractions)
                print(f"Y_initial[{i}] \n {data[i][0][ind]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(cv.species_mass)
                print(f"rhoY_initial[{i}] \n {data[i][0][ind]}")

    #print(f"inside limiter")

    ##################
    # 1.0 limit the density to be above 0.
    ##################

    cell_avgs = element_average(dcoll, dd, cv.mass, volumes=element_vols)
    #print(f"rho_avg {cell_avgs}")

    mmin_i = op.elementwise_min(dcoll, dd, cv.mass)
    cell_avgs = actx.np.where(
        actx.np.greater(cell_avgs, min_allowed_density),
        cell_avgs,
        min_allowed_density)
    mmin = 0.1*cell_avgs
    #mmin = 1.e-4

    #print(f"modified rho_avg {cell_avgs}")

    theta_rho = ones*actx.np.maximum(0.,
        actx.np.where(actx.np.less(mmin_i + do_limit_toler, mmin),
                      (mmin-mmin_i)/(cell_avgs - mmin_i),
                      0.)
    )
    #print(f"{theta_rho=}")


    """
    spec_lim = make_obj_array([cv.species_mass[i] +
                               theta_rho*(elem_avg_cv_safe.species_mass[i] -
                                               cv_updated.species_mass[i])
                               for i in range(0, nspecies)
                               ])
    """

    mass_lim = actx.np.zeros_like(cv.mass)
    if nspecies > 0:
        spec_lim = actx.np.zeros_like(cv.species_mass)
        # find theta for all the species
        for i in range(0, nspecies):
            cell_avgs = element_average(
                dcoll, dd, cv.species_mass[i],
                volumes=element_vols)
            mmin = 0.1*cell_avgs
            cell_avgs = actx.np.where(actx.np.greater(cell_avgs, mmin), cell_avgs,
                                      mmin)
            spec_lim[i] = (cv.species_mass[i] +
                           theta_rho*(
                               cell_avgs - cv.species_mass[i]))
            mass_lim = mass_lim + spec_lim[i]
    else:
        mass_lim = (cv.mass + theta_rho*(cell_avgs - cv.mass))
        #spec_lim = mass_lim*cv.species_mass_fractions

    mom_lim = mass_lim*cv.velocity
    kin_energy = 0.5*np.dot(cv.velocity, cv.velocity)
    int_energy = cv.energy - cv.mass*kin_energy
    energy_lim = (int_energy/cv.mass + kin_energy)*mass_lim

    """
    # keep rho balanced with rhoY
    if nspecies > 0:
        mass_lim = 0.
        for i in range(0, nspecies):
            mass_lim = mass_lim + spec_lim[i]
    #else:
        #mass_lim = cv_updated.mass + theta_pressure*(
            #elem_avg_cv_safe.mass - cv_updated.mass)
    """

    cv_update_rho = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                                   momentum=mom_lim,
                                   species_mass=spec_lim)
    #cv_update_rho = cv

    if print_stuff is True:
        np.set_printoptions(threshold=sys.maxsize, precision=16)
        # initial state
        #print(f"{theta_rho=}")
        #print(f"{theta_pressure=}")
        print("After rho limit")
        for ind in index:
            print(f"element {ind}")
            data = actx.to_numpy(theta_rho)
            print(f"theta_rho \n {data[0][ind]}")
            data = actx.to_numpy(cell_avgs)
            print(f"cell_avgs \n {data[0][ind]}")
            temperature_rho = gas_model.eos.temperature(
                cv=cv_update_rho, temperature_seed=temperature_seed)
            pressure_rho = gas_model.eos.pressure(
                cv=cv_update_rho, temperature=temperature_rho)
            # initial state
            data = actx.to_numpy(cv_update_rho.mass)
            print(f"cv.mass \n {data[0][ind]}")
            data = actx.to_numpy(temperature_rho)
            print(f"temperature_rho \n {data[0][ind]}")
            data = actx.to_numpy(pressure_rho)
            print(f"pressure_rho \n {data[0][ind]}")
            #data = actx.to_numpy(temperature_seed)
            #print(f"temperature_seed \n {data[0][ind]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(cv_update_rho.species_mass_fractions)
                print(f"Y_[{i}] \n {data[i][0][ind]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(cv_update_rho.species_mass)
                print(f"rhoY_[{i}] \n {data[i][0][ind]}")

    ##################
    # 2.0 limit the species mass fractions
    ##################
    theta_spec = actx.np.zeros_like(cv.species_mass_fractions)
    balance_spec = actx.np.zeros_like(cv.mass)
    #if 0:
    if nspecies > 0:
        # find theta for all the species
        for i in range(0, nspecies):
            mmin = 0.
            mmin_i = op.elementwise_min(dcoll, dd,
                                        cv_update_rho.species_mass_fractions[i])

            cell_avgs = element_average(
                dcoll, dd, cv_update_rho.species_mass_fractions[i],
                volumes=element_vols)
            cell_avgs = actx.np.where(actx.np.greater(cell_avgs, mmin), cell_avgs,
                                      mmin)

            _theta = actx.np.maximum(0.,
                actx.np.where(actx.np.less(mmin_i + do_limit_toler, mmin),
                              (mmin-mmin_i)/(cell_avgs - mmin_i),
                              0.)
            )

            mmax_i = op.elementwise_max(dcoll, dd,
                                        cv_update_rho.species_mass_fractions[i])
            mmax = 1.0
            cell_avgs = actx.np.where(actx.np.less(cell_avgs, mmax), cell_avgs, mmax)
            _theta = actx.np.maximum(_theta,
                actx.np.where(actx.np.greater(mmax_i - do_limit_toler, mmax),
                              (mmax_i - mmax)/(mmax_i - cell_avgs),
                              0.)
            )

            theta_spec[i] = _theta*ones
            balance_spec = actx.np.where(actx.np.greater(theta_spec[i], 1.e-15),
                                      1.0, balance_spec)

            #print(f"species {i}, {_theta=}")

            # apply the limiting to all species equally
            spec_lim[i] = (cv_update_rho.species_mass_fractions[i] +
                           theta_spec[i]*(
                               cell_avgs - cv_update_rho.species_mass_fractions[i]))

            if print_stuff is True:
                for ind in index:
                    print(f"element {ind}")
                    np.set_printoptions(threshold=sys.maxsize, precision=16)
                    data = actx.to_numpy(spec_lim[i])
                    print(f"spec_lim[i] \n {data[0][ind]}")

        # limit the species mass fraction sum to 1.0
        aux = actx.np.zeros_like(cv_update_rho.mass)
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        for i in range(nspecies):
            # only rebalance where species limiting actually occured
            spec_lim[i] = actx.np.where(actx.np.greater(balance_spec, 0.),
                                        spec_lim[i]/aux, spec_lim[i])

        # convert back to rhoY
        spec_lim_mass = spec_lim*cv_update_rho.mass

        mass_lim = 0.
        for i in range(0, nspecies):
            mass_lim = mass_lim + spec_lim_mass[i]
        mom_lim = mass_lim*cv_update_rho.velocity
        kin_energy = 0.5*np.dot(cv_update_rho.velocity, cv_update_rho.velocity)
        int_energy = cv_update_rho.energy - mass_lim*kin_energy
        energy_lim = (int_energy/mass_lim + kin_energy)*mass_lim

        # modify Temperature (energy) maintain pressure equilibrium
        temperature_update_rho = gas_model.eos.temperature(
            cv=cv_update_rho, temperature_seed=temperature_seed)
        pressure_update_rho = gas_model.eos.pressure(
            cv=cv_update_rho, temperature=temperature_update_rho)

        kin_energy = 0.5*np.dot(cv_update_rho.velocity, cv_update_rho.velocity)
        positive_pressure = actx.np.greater(pressure_update_rho,
                                            min_allowed_pressure)

        update_dv = positive_pressure

        r = gas_model.eos.gas_const(species_mass_fractions=spec_lim)
        temperature_update_y = pressure_update_rho/r/mass_lim

        energy_lim = actx.np.where(
            update_dv,
            mass_lim*(gas_model.eos.get_internal_energy(temperature_update_y,
                      species_mass_fractions=spec_lim)
                      + kin_energy),
            cv_update_rho.energy
        )
        cv_update_y = make_conserved(dim=dim,
                                     mass=mass_lim,
                                     energy=energy_lim,
                                     momentum=mom_lim,
                                     species_mass=spec_lim_mass)
    else:
        cv_update_y = cv_update_rho

    if print_stuff is True:
        np.set_printoptions(threshold=sys.maxsize, precision=16)

        print("After mass fraction limiting")
        temperature_update_y = gas_model.eos.temperature(
            cv=cv_update_y, temperature_seed=temperature_seed)
        pressure_update_y = gas_model.eos.pressure(
            cv=cv_update_y, temperature=temperature_update_y)
        for ind in index:
            print(f"element {ind}")
            data = actx.to_numpy(temperature_update_y)
            print(f"temperature_update_y \n {data[0][ind]}")
            data = actx.to_numpy(pressure_update_y)
            print(f"pressure_update_y \n {data[0][ind]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(theta_spec)
                print(f"theta_spec[{i}] \n {data[i][0][ind]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(cv_update_y.species_mass_fractions)
                print(f"Y_update_y[{i}] \n {data[i][0][ind]}")

    ##################
    # 3.0 find the average element cv and pressure
    ##################
    cv_updated = cv_update_y
    temperature_updated = gas_model.eos.temperature(
        cv=cv_updated, temperature_seed=temperature_seed)
    pressure_updated = gas_model.eos.pressure(
        cv=cv_updated, temperature=temperature_updated)

    elem_avg_cv = _element_average_cv(dcoll, dd, cv_updated, volumes=element_vols)
    elem_avg_temp = gas_model.eos.temperature(
        cv=elem_avg_cv, temperature_seed=temperature_seed)
    elem_avg_pres = gas_model.eos.pressure(
        cv=elem_avg_cv, temperature=elem_avg_temp)

    # use an entropy function to keep pressure positive and entropy
    # above some minimum value
    # not sure which gamma to use here? the average state?
    gamma = gas_model.eos.gamma(cv_updated, temperature_updated)
    gamma_avg = gas_model.eos.gamma(elem_avg_cv, elem_avg_temp)
    mmin = 1.e-12
    theta_smin = (pressure_updated -
                  actx.np.exp(entropy_min)*cv_updated.mass**gamma)
    theta_smin_i = op.elementwise_min(dcoll, dd, theta_smin)

    # in some cases, the average pressure is too small, we need to satisfy
    # elem_avg_pres > math.exp(entropy_min)*elem_avg_cv.mass**gamma
    # compute a safe pressure such that this is always true and use it to compute
    # a safe average energy
    #safe_pressure = actx.np.exp(entropy_min)*elem_avg_cv.mass**gamma_avg - toler
    safe_pressure = actx.np.exp(entropy_min)*elem_avg_cv.mass**gamma_avg
    r_avg = gas_model.eos.gas_const(
        species_mass_fractions=elem_avg_cv.species_mass_fractions)
    safe_temperature = safe_pressure/elem_avg_cv.mass/r_avg
    kin_energy = 0.5*np.dot(elem_avg_cv.velocity, elem_avg_cv.velocity)
    safe_energy = elem_avg_cv.mass*(
        gas_model.eos.get_internal_energy(
            safe_temperature,
            species_mass_fractions=elem_avg_cv.species_mass_fractions)
        + kin_energy)

    safe_energy = actx.np.where(
        actx.np.less(actx.np.exp(entropy_min)*elem_avg_cv.mass**gamma_avg,
                     elem_avg_pres),
        elem_avg_cv.energy, safe_energy)

    theta_savg = actx.np.maximum(
        elem_avg_pres, safe_pressure) -\
        actx.np.exp(entropy_min)*elem_avg_cv.mass**gamma_avg
    elem_avg_cv_safe = elem_avg_cv.replace(energy=safe_energy)

    """
    theta_pressure = ones*actx.np.maximum(0.,
        actx.np.where(actx.np.greater(actx.np.abs(theta_smin_i - theta_savg),
                                      actx.np.abs(1.e-6*theta_savg)),
                      (mmin-theta_smin_i)/(theta_savg - theta_smin_i),
                      0.))
    """
    theta_pressure = actx.np.maximum(0.,
        actx.np.where(actx.np.less(theta_smin_i + do_limit_toler, mmin),
                      (mmin-theta_smin_i)/(theta_savg - theta_smin_i),
                      0.)
    )

    ##################
    # 4.0 limit cv where the entropy minimum function is violated
    #     this in turn keeps the pressure positive
    ##################
    mom_lim = make_obj_array([cv_updated.momentum[i] +
        theta_pressure*(elem_avg_cv_safe.momentum[i] - cv_updated.momentum[i])
        for i in range(dim)
    ])
    energy_lim = (cv_updated.energy +
                  theta_pressure*(elem_avg_cv_safe.energy - cv_updated.energy))
    spec_lim = make_obj_array([cv_updated.species_mass[i] +
                               theta_pressure*(elem_avg_cv_safe.species_mass[i] -
                                               cv_updated.species_mass[i])
                               for i in range(0, nspecies)
                               ])

    # keep rho balanced with rhoY
    if nspecies > 0:
        mass_lim = 0.
        for i in range(0, nspecies):
            mass_lim = mass_lim + spec_lim[i]
    else:
        mass_lim = cv_updated.mass + theta_pressure*(
            elem_avg_cv_safe.mass - cv_updated.mass)

    cv_lim = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                            momentum=mom_lim,
                            species_mass=spec_lim)

    if print_stuff is True:
        np.set_printoptions(threshold=sys.maxsize, precision=16)

        temperature_final = gas_model.eos.temperature(
            cv=cv_lim, temperature_seed=temperature_seed)
        pressure_final = gas_model.eos.pressure(
            cv=cv_lim, temperature=temperature_final)

        """
        def get_temperature_update_limit(cv, temperature):
            y = cv.species_mass_fractions
            e = gas_model.eos.internal_energy(cv)/cv.mass
            return actx.np.abs(
                gas_model.eos._pyrometheus_mech.get_temperature_update_energy(
                    e, temperature, y))

        temp_resid = get_temperature_update_limit(
            cv_lim, temperature_final)/temperature_final

        # run through a temperature solve manually, print out the updates

        num_iter = 20

        def do_temperature_iter(cv, tseed):
            y = cv.species_mass_fractions
            t_i = temperature_seed
            print(f" First: {actx.to_numpy(t_i)[0][index]=}")
            e = gas_model.eos.internal_energy(cv)/cv.mass
            for _ in range(num_iter):
                t_resid = (gas_model.eos._pyrometheus_mech.
                    get_temperature_update_energy(e, t_i, y))
                t_i = t_i + t_resid
                data_t_i = actx.to_numpy(t_i)[0][index]
                data_t_resid = actx.to_numpy(t_resid)[0][index]
                print(f"iter {_=}: {data_t_i=}, {data_t_resid=}")
        """

        #do_temperature_iter(cv_lim, temperature_seed)
        # initial state
        #if (isinstance(dd.domain_tag, BoundaryDomainTag) and
              #dd.domain_tag.tag == "isothermal_wall"):
            #print(f"{actx.to_numpy(theta_rho)=}")
            #print(f"{actx.to_numpy(theta_pressure)=}")
            #print(f"{actx.to_numpy(theta_spec)=}")
        #print(f"{actx.to_numpy(theta_rho)=}")
        #print(f"{actx.to_numpy(theta_pressure)=}")
        #print(f"{actx.to_numpy(theta_spec)=}")
        #print(f"{cv_lim.species_mass_fractions=}")
        #print(f"{temperature_final}")
        #print(f"{temp_resid=}")
        print("All done limiting")
        for ind in index:
            print(f"element {ind}")
            #data = actx.to_numpy(temp_resid)
            #print(f"temp_resid \n {data[0][ind]}")
            data = actx.to_numpy(theta_rho)
            print(f"theta_rho \n {data[0][ind]}")
            data = actx.to_numpy(theta_pressure)
            print(f"theta_pressure \n {data[0][ind]}")

            data = actx.to_numpy(entropy_min)
            print(f"entropy_min \n {data[0][ind]}")
            entropy_initial = actx.np.log(pressure_initial/cv.mass**1.4)
            data = actx.to_numpy(entropy_initial)
            print(f"entropy_initial \n {data[0][ind]}")
            entropy_final = actx.np.log(pressure_final/cv_lim.mass**1.4)
            data = actx.to_numpy(entropy_final)
            print(f"entropy_final \n {data[0][ind]}")

            data = actx.to_numpy(theta_savg)
            print(f"theta_savg \n {data[0][ind]}")
            data = actx.to_numpy(theta_smin)
            print(f"theta_smin_i \n {data[0][ind]}")
            data = actx.to_numpy(theta_smin_i)
            print(f"theta_smin_i \n {data[0][ind]}")

            for i in range(0, nspecies):
                data = actx.to_numpy(theta_spec)
                print(f"theta_Y[{i}] \n {data[i][0][ind]}")
            data = actx.to_numpy(cv_lim.mass)
            print(f"cv_lim.mass \n {data[0][ind]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(cv_lim.species_mass_fractions)
                print(f"Y_final[{i}] \n {data[i][0][ind]}")
            for i in range(0, dim):
                data = actx.to_numpy(cv_lim.momentum)
                print(f"cv_lim.momentum_final[{i}] \n {data[i][0][ind]}")
            for i in range(0, dim):
                data = actx.to_numpy(cv_lim.velocity)
                print(f"cv_lim.velocity_final[{i}] \n {data[i][0][ind]}")
            data = actx.to_numpy(temperature_final)
            print(f"temperature_final \n {data[0][ind]}")
            data = actx.to_numpy(pressure_final)
            print(f"pressure_final \n {data[0][ind]}")

            print("End of limiting")

    if viz_theta:
        return make_obj_array([cv_lim, theta_rho,
                               theta_spec, theta_pressure])
    else:
        return cv_lim
        #return cv_update_rho


@mpi_entry_point
def main(actx_class, restart_filename=None, target_filename=None,
         user_input_file=None, use_overintegration=False,
         disable_logpyle=False,
         casename=None, log_path="log_data", use_esdg=False,
         disable_fallbacks=False, axi_filename=None):

    allow_fallbacks = not disable_fallbacks
    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    first_profiling_step = 4
    last_profiling_step = 104

    if first_profiling_step > 0:
        MPI.Pcontrol(2)
        MPI.Pcontrol(0)

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    if casename is None:
        casename = "mirgecom"

    # logging and profiling
    logmgr = None
    if not disable_logpyle:
        logname = log_path + "/" + casename + ".sqlite"
        if rank == 0:
            log_dir = os.path.dirname(logname)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
        comm.Barrier()

        logmgr = initialize_logmgr(True,
            filename=logname, mode="wu", mpi_comm=comm)

    # set up driver parameters
    from mirgecom.simutil import configurate
    from mirgecom.io import read_and_distribute_yaml_data
    input_data = read_and_distribute_yaml_data(comm, user_input_file)

    use_callbacks = configurate("use_callbacks", input_data, True)

    use_gmsh = configurate("use_gmsh", input_data, True)
    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    use_tpe = configurate("use_tensor_product_elements", input_data, False)
    mesh_origin = configurate("mesh_origin", input_data, [0.0, -0.01])

    actx = initialize_actx(actx_class, comm,
                           use_axis_tag_inference_fallback=allow_fallbacks,
                           use_einsum_inference_fallback=allow_fallbacks)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)
    alloc = getattr(actx, "allocator", None)

    # i/o frequencies
    nviz = configurate("nviz", input_data, 500)
    nrestart = configurate("nrestart", input_data, 5000)
    nhealth = configurate("nhealth", input_data, 1)
    nstatus = configurate("nstatus", input_data, 1)
    nspeccheck = configurate("nspeccheck", input_data, 1)

    # Restart from previous axisymmetric run
    # Enable axi restart by setting "axi_filename" on
    # command line or in input yaml file.
    if axi_filename is None:
        axi_filename = configurate("axi_filename", None)
    restart_from_axi = axi_filename is not None

    # garbage collection frequency
    ngarbage = configurate("ngarbage", input_data, 10)

    # verbosity for what gets written to viz dumps, increase for more stuff
    viz_level = configurate("viz_level", input_data, 1)
    # control the time interval for writing viz dumps
    viz_interval_type = configurate("viz_interval_type", input_data, 0)

    # default timestepping control
    advance_time = configurate("advance_time", input_data, True)
    integrator = configurate("integrator", input_data, "rk4")
    current_dt = configurate("current_dt", input_data, 1.e-8)
    t_final = configurate("t_final", input_data, 1.e-7)
    t_viz_interval = configurate("t_viz_interval", input_data, 1.e-8)
    current_cfl = configurate("current_cfl", input_data, 1.0)
    constant_cfl = configurate("constant_cfl", input_data, False)

    # these are modified below for a restart
    current_t = configurate("current_t", input_data, 0.0)
    t_start = 0.
    t_wall_start = 0.
    current_step = 0
    first_step = 0
    last_viz_interval = 0
    force_eval = True

    # default health status bounds
    health_pres_min = configurate("health_pres_min", input_data, 0.1)
    health_pres_max = configurate("health_pres_max", input_data, 2.e6)
    health_temp_min = configurate("health_temp_min", input_data, 1.0)
    health_temp_max = configurate("health_temp_max", input_data, 5000.)
    health_mass_frac_min = configurate("health_mass_frac_min", input_data, -1.0)
    health_mass_frac_max = configurate("health_mass_frac_max", input_data, 2.0)

    # discretization and model control
    order = configurate("order", input_data, 2)
    viz_order = configurate("viz_order", input_data, order)
    quadrature_order = configurate("quadrature_order", input_data, -1)
    alpha_sc = configurate("alpha_sc", input_data, 0.3)
    kappa_sc = configurate("kappa_sc", input_data, 0.5)
    s0_sc = configurate("s0_sc", input_data, -5.0)

    drop_order_strength = configurate("drop_order_strength", input_data, 0.)
    use_drop_order = False
    if drop_order_strength > 0.:
        use_drop_order = True

    av2_mu0 = configurate("av2_mu0", input_data, 0.1)
    av2_beta0 = configurate("av2_beta0", input_data, 6.0)
    av2_kappa0 = configurate("av2_kappa0", input_data, 1.0)
    av2_d0 = configurate("av2_d0", input_data, 0.1)
    av2_prandtl0 = configurate("av2_prandtl0", input_data, 0.9)
    av2_mu_s0 = configurate("av2_mu_s0", input_data, 0.)
    av2_kappa_s0 = configurate("av2_kappa_s0", input_data, 0.)
    av2_beta_s0 = configurate("av2_beta_s0", input_data, 0.01)
    av2_d_s0 = configurate("av2_d_s0", input_data, 0.)
    smooth_char_length = configurate("smooth_char_length", input_data, 5)
    smooth_char_length_alpha = configurate("smooth_char_length_alpha",
                                           input_data, 0.025)
    use_smoothed_char_length = False
    if smooth_char_length > 0:
        use_smoothed_char_length = True

    smoothness_alpha = configurate("smoothness_alpha", input_data, 0.1)
    smoothness_tau = configurate("smoothness_tau", input_data, 0.01)

    dim = configurate("dimen", input_data, 2)
    inv_num_flux = configurate("inv_num_flux", input_data, "rusanov")
    mesh_filename = configurate("mesh_filename", input_data, "data/actii_2d.msh")
    generate_mesh = configurate("generate_mesh", input_data, True)
    mesh_partition_prefix = configurate("mesh_partition_prefix",
                                        input_data, "actii_2d")
    periodic_mesh = configurate("periodic_mesh", input_data, "False")
    noslip = configurate("noslip", input_data, True)
    use_1d_part = configurate("use_1d_part", input_data, True)
    part_tol = configurate("partition_tolerance", input_data, 0.01)

    # setting these to none in the input file toggles the check for that
    # boundary off provides support for legacy runs only where you could
    # specify boundary tags that were unused in certain cases
    use_outflow_boundary = configurate(
        "use_outflow_boundary", input_data, "none")
    use_inflow_boundary = configurate(
        "use_inflow_boundary", input_data, "none")
    use_flow_boundary = configurate(
        "use_flow_boundary", input_data, "prescribed")
    use_injection_boundary = configurate(
        "use_injection_boundary", input_data, "none")
    use_upstream_injection_boundary = configurate(
        "use_upstream_injection_boundary", input_data, "none")
    use_wall_boundary = configurate(
        "use_wall_boundary", input_data, "isothermal_noslip")
    use_interface_boundary = configurate(
        "use_interface_boundary", input_data, "none")
    use_symmetry_boundary = configurate(
        "use_symmetry_boundary", input_data, "none")
    use_slip_wall_boundary = configurate(
        "use_slip_wall_boundary", input_data, "none")
    use_noslip_wall_boundary = configurate(
        "use_noslip_wall_boundary", input_data, "none")

    outflow_pressure = configurate("outflow_pressure", input_data, 100.0)
    ramp_beginP = configurate("ramp_beginP", input_data, 100.0)
    ramp_endP = configurate("ramp_endP", input_data, 1000.0)
    ramp_time_start = configurate("ramp_time_start", input_data, 0.0)
    ramp_time_interval = configurate("ramp_time_interval", input_data, 1.e-4)

    # for each tagged boundary surface, what are they assigned to be
    # isothermal wall -> wall when current running simulation support is not needed
    #
    # wall_interface is automatically generated from a shared fluid/solid volume
    # interface and only when the solid volume is turned off by use_wall
    bndry_config = {"outflow": use_outflow_boundary,
                    "inflow": use_inflow_boundary,
                    "flow": use_flow_boundary,
                    "injection": use_injection_boundary,
                    "upstream_injection": use_upstream_injection_boundary,
                    "isothermal_wall": use_wall_boundary,
                    "wall": use_wall_boundary,
                    "slip_wall": use_slip_wall_boundary,
                    "noslip_wall": use_noslip_wall_boundary,
                    "symmetry": use_symmetry_boundary,
                    "wall_interface": use_interface_boundary}

    # list of strings that are allowed to defined boundary conditions
    allowed_boundary_types = [
        "none",
        "isothermal_noslip",
        "isothermal_slip",
        "adiabatic_noslip",
        "adiabatic_slip",
        "pressure_outflow",
        "riemann_outflow",
        "riemann_inflow",
        "prescribed",
        "isentropic_pressure_ramp"
    ]

    # boundary sanity check
    def boundary_type_sanity(boundary, boundary_type):
        if boundary_type not in allowed_boundary_types:
            error_message = ("Invalid boundary specification "
                             f"{boundary_type} for {boundary}")
            if rank == 0:
                raise RuntimeError(error_message)

    for bnd in bndry_config:
        boundary_type_sanity(bnd, bndry_config[bnd])

    # material properties and models options
    gas_mat_prop = configurate("gas_mat_prop", input_data, 0)
    nspecies = configurate("nspecies", input_data, 0)

    spec_diff = configurate("spec_diff", input_data, 1.e-4)
    eos_type = configurate("eos", input_data, 0)
    transport_type = configurate("transport", input_data, 0)
    use_lewis_transport = configurate("use_lewis_transport", input_data, False)
    # for pyrometheus, number of newton iterations
    pyro_temp_iter = configurate("pyro_temp_iter", input_data, 3)
    # for pyrometheus, toleranace for temperature residual
    pyro_temp_tol = configurate("pyro_temp_tol", input_data, 1.e-4)

    # for overwriting the default fluid material properties
    fluid_gamma = configurate("fluid_gamma", input_data, -1.)
    fluid_mw = configurate("fluid_mw", input_data, -1.)
    fluid_kappa = configurate("fluid_kappa", input_data, -1.)
    fluid_mu = configurate("mu", input_data, -1.)
    fluid_beta = configurate("beta", input_data, -1.)

    # rhs control
    use_axisymmetric = configurate("use_axisymmetric", input_data, False)
    use_combustion = configurate("use_combustion", input_data, True)
    use_wall = configurate("use_wall", input_data, True)
    use_wall_ox = configurate("use_wall_ox", input_data, True)
    use_wall_mass = configurate("use_wall_mass", input_data, True)
    use_ignition = configurate("use_ignition", input_data, 0)
    use_injection_source = configurate("use_injection_source", input_data, True)
    use_injection_source_comb = configurate("use_injection_source_comb",
                                            input_data, False)
    use_injection_source_3d = configurate("use_injection_source_3d",
                                          input_data, False)
    use_injection = configurate("use_injection", input_data, True)
    init_injection = configurate("init_injection", input_data, False)
    use_upstream_injection = configurate("use_upstream_injection", input_data, False)

    # outflow sponge location and strength
    use_sponge = configurate("use_sponge", input_data, True)
    use_time_dependent_sponge = configurate("use_time_dependent_sponge",
                                            input_data, False)
    use_gauss_outlet = configurate("use_gauss_outlet", input_data, False)
    gauss_outlet_radius = configurate("gauss_outlet_radius", input_data, 1.45)
    gauss_outlet_sigma = configurate("gauss_outlet_sigma", input_data, 0.5)
    sponge_sigma = configurate("sponge_sigma", input_data, 1.0)

    # artificial viscosity control
    #    0 - none
    #    1 - physical viscosity based, div(velocity) indicator
    #    2 - physical viscosity based, indicators and diffusion for all transport
    use_av = configurate("use_av", input_data, 0)

    # species limiter
    #    0 - none
    #    1 - limit in on call to make_fluid_state
    use_species_limiter = configurate("use_species_limiter", input_data, 0)
    limiter_smin = configurate("limiter_smin", input_data, 10)
    constant_smin = configurate("constant_smin", input_data, True)
    use_hammer_species = configurate("use_hammer_species", input_data, True)

    # Filtering is implemented according to HW Sec. 5.3
    # The modal response function is e^-(alpha * eta ^ 2s), where
    # - alpha is a user parameter (defaulted like HW's)
    # - eta := (mode - N_c)/(N - N_c)
    # - N_c := cutoff mode ( = *filter_frac* x order)
    # - s := order of the filter (divided by 2)
    # Modes below N_c are unfiltered. Modes above Nc are weighted
    # by the modal response function described above.
    #
    # Two different filters can be used with the prediction driver.
    # 1) Solution filtering: filters the solution every *soln_nfilter* steps
    # 2) RHS filtering: filters the RHS every substep
    #
    # Turn on SOLUTION filtering by setting soln_nfilter > 0
    # Turn on RHS filtering by setting use_rhs_filter = 1.
    #
    # --- Filtering settings ---
    # ------ Solution filtering
    # filter every *nfilter* steps (-1 = no filtering)
    soln_nfilter = configurate("soln_nfilter", input_data, -1)
    soln_filter_frac = configurate("soln_filter_frac", input_data, 0.5)
    soln_filter_cutoff = configurate("soln_filter_cutoff", input_data, -1)
    soln_filter_order = configurate("soln_filter_order", input_data, 8)

    # Alpha value suggested by:
    # JSH/TW Nodal DG Methods, Section 5.3
    # DOI: 10.1007/978-0-387-72067-8
    soln_filter_alpha_default = -1.0*np.log(np.finfo(float).eps)
    soln_filter_alpha = configurate("soln_filter_alpha", input_data,
                                    soln_filter_alpha_default)
    # ------ RHS filtering
    use_rhs_filter = configurate("use_rhs_filter", input_data, False)
    rhs_filter_frac = configurate("rhs_filter_frac", input_data, 0.5)
    rhs_filter_cutoff = configurate("rhs_filter_cutoff", input_data, -1)
    rhs_filter_order = configurate("rhs_filter_order", input_data, 8)
    rhs_filter_alpha = configurate("rhs_filter_alpha", input_data,
                                   soln_filter_alpha_default)

    # initialization configuration
    init_case = configurate("init_case", input_data, "y3prediction")
    actii_init_case = configurate("actii_init_case", input_data, "cav5")

    # Shock 1D flow properties
    pres_bkrnd = configurate("pres_bkrnd", input_data, 100.)
    temp_bkrnd = configurate("temp_bkrnd", input_data, 300.)
    vel_bkrnd = configurate("vel_bkrnd", input_data, 0.)
    mach = configurate("mach", input_data, 2.0)
    shock_loc_x = configurate("shock_loc_x", input_data, 0.05)
    fuel_loc_x = configurate("fuel_loc_x", input_data, 0.07)
    inlet_height = configurate("inlet_height", input_data, 0.013)

    # Shock 1D mesh properties
    mesh_size = configurate("mesh_size", input_data, 0.001)
    bl_ratio = configurate("bl_ratio", input_data, 3)
    interface_ratio = configurate("interface_ratio", input_data, 2)
    transfinite = configurate("transfinite", input_data, False)
    mesh_angle = configurate("mesh_angle", input_data, 0.)

    # Discontinuity flow properties
    pres_left = configurate("pres_left", input_data, 100.)
    pres_right = configurate("pres_right", input_data, 10.)
    temp_left = configurate("temp_left", input_data, 400.)
    temp_right = configurate("temp_right", input_data, 300.)
    sigma_disc = configurate("sigma_disc", input_data, 10.)

    # mixing layer flow properties
    vorticity_thickness = configurate("vorticity_thickness", input_data, 0.32e-3)

    # ACTII flow properties
    total_pres_inflow = configurate("total_pres_inflow", input_data, 2.745e5)
    total_temp_inflow = configurate("total_temp_inflow", input_data, 2076.43)
    mf_o2 = configurate("mass_fraction_o2", input_data, 0.273)
    mole_c2h4 = configurate("mole_c2h4", input_data, 0.5)
    mole_h2 = configurate("mole_h2", input_data, 0.5)

    # injection flow properties
    total_pres_inj = configurate("total_pres_inj", input_data, 50400.)
    total_temp_inj = configurate("total_temp_inj", input_data, 300.)
    total_pres_inj_upstream = configurate("total_pres_inj_upstream",
                                          input_data, total_pres_inj)
    total_temp_inj_upstream = configurate("total_temp_inj_upstream",
                                          input_data, total_temp_inj)
    mach_inj = configurate("mach_inj", input_data, 1.0)

    # parameters to adjust the shape of the initialization
    vel_sigma = configurate("vel_sigma", input_data, 1000)
    temp_sigma = configurate("temp_sigma", input_data, 1250)
    # adjusted to match the mass flow rate
    vel_sigma_inj = configurate("vel_sigma_inj", input_data, 5000)
    temp_sigma_inj = configurate("temp_sigma_inj", input_data, 5000)
    temp_wall = configurate("wall_temperature", input_data, 300)

    # wall stuff
    wall_penalty_amount = configurate("wall_penalty_amount", input_data, 0)
    wall_time_scale = configurate("wall_time_scale", input_data, 1)
    wall_material = configurate("wall_material", input_data, 0)

    # use fluid average diffusivity by default
    wall_insert_ox_diff = spec_diff

    # Averaging from https://www.azom.com/article.aspx?ArticleID=1630
    # for graphite
    wall_insert_rho = configurate("wall_insert_rho", input_data, 1625)
    wall_insert_cp = configurate("wall_insert_cp", input_data, 770)
    wall_insert_kappa = configurate("wall_insert_kappa", input_data, 247.5)

    # Averaging from http://www.matweb.com/search/datasheet.aspx?bassnum=MS0001
    # for steel
    wall_surround_rho = configurate("wall_surround_rho", input_data, 7.9e3)
    wall_surround_cp = configurate("wall_surround_cp", input_data, 470)
    wall_surround_kappa = configurate("wall_surround_kappa", input_data, 48)

    # initialize the ignition spark
    spark_init_time = configurate("ignition_init_time", input_data, 999999999.)
    spark_strength = configurate("ignition_strength", input_data, 2.e7)
    spark_duration = configurate("ignition_duration", input_data, 1.e-8)
    spark_diameter = configurate("ignition_diameter", input_data, 0.0025)
    spark_init_loc_x = configurate("ignition_init_loc_x", input_data, 0.677)
    spark_init_loc_y = configurate("ignition_init_loc_y", input_data, -0.021)
    spark_init_loc_z = configurate("ignition_init_loc_z", input_data, 0.0)

    # initialize the injection source
    injection_source_init_time = configurate("injection_source_init_time",
                                             input_data, 999999999.)
    injection_source_ramp_time = configurate("injection_source_ramp_time",
                                             input_data, 1.e-4)
    injection_source_mass = configurate("injection_source_mass",
                                        input_data, 2.)
    injection_source_mom_x = configurate("injection_source_mom_x",
                                         input_data, 3.)
    injection_source_mom_y = configurate("injection_source_mom_y",
                                         input_data, 3.)
    injection_source_mom_z = configurate("injection_source_mom_z",
                                         input_data, 0.)
    injection_source_energy = configurate("injection_source_energy",
                                          input_data, 1.e3)
    injection_source_diameter = configurate("injection_source_diameter",
                                            input_data, 0.0025)
    injection_source_loc_x = configurate("injection_source_loc_x",
                                              input_data, 0.677)
    injection_source_loc_y = configurate("injection_source_loc_y",
                                         input_data, -0.021)
    injection_source_loc_z = configurate("injection_source_loc_z",
                                         input_data, 0.0)
    injection_source_loc_x_comb = configurate("injection_source_loc_x_comb",
                                              input_data, 0.677)
    injection_source_loc_y_comb = configurate("injection_source_loc_y_comb",
                                         input_data, -0.021)
    injection_source_loc_z_comb = configurate("injection_source_loc_z_comb",
                                         input_data, 0.0)
    injection_source_mass_comb = configurate("injection_source_mass_comb",
                                        input_data, 2.)
    injection_source_mom_x_comb = configurate("injection_source_mom_x_comb",
                                         input_data, 3.)
    injection_source_mom_y_comb = configurate("injection_source_mom_y_comb",
                                         input_data, 3.)
    injection_source_mom_z_comb = configurate("injection_source_mom_z_comb",
                                         input_data, 0.)
    injection_source_energy_comb = configurate("injection_source_energy_comb",
                                          input_data, 1.e3)

    # initialization for the sponge
    inlet_sponge_x0 = configurate("inlet_sponge_x0", input_data, 0.225)
    inlet_sponge_thickness = configurate("inlet_sponge_thickness", input_data, 0.015)
    outlet_sponge_x0 = configurate("outlet_sponge_x0", input_data, 0.89)
    outlet_sponge_thickness = configurate("outlet_sponge_thickness",
                                          input_data, 0.04)
    top_sponge_x0 = configurate("top_sponge_x0", input_data, 0.1)
    top_sponge_thickness = configurate("outlet_sponge_thickness",
                                          input_data, 0.1)
    inj_sponge_x0 = configurate("inj_sponge_x0", input_data, 0.645)
    inj_sponge_thickness = configurate("inj_sponge_thickness", input_data, 0.005)
    upstream_inj_sponge_y0 = configurate("upstream_inj_sponge_y0",
                                         input_data, -0.01753)

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144",
                           "compiled_lsrk54", "ssprk43"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if integrator == "compiled_lsrk54":
        if rank == 0:
            print("Setting force_eval = False for pre-compiled time integration")
        force_eval = False

    if viz_interval_type > 2:
        error_message = "Invalid value for viz_interval_type [0-2]"
        raise RuntimeError(error_message)

    s0_sc = np.log10(1.0e-4 / np.power(order, 4))
    if rank == 0:
        print(f"Shock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")

    # use_av=1 specific parameters
    # flow stagnation temperature
    static_temp = 2076.43
    # steepness of the smoothed function
    theta_sc = 100
    # cutoff, smoothness below this value is ignored
    beta_sc = 0.01
    gamma_sc = 1.5

    if rank == 0:
        if use_smoothed_char_length:
            print("Smoothing characteristic length for use in artificial viscosity")
            print(f"smoothing_alpha {smooth_char_length_alpha}")

        if use_av > 0:
            print(f"Artificial viscosity {smoothness_alpha=}")
            print(f"Artificial viscosity {smoothness_tau=}")

        if use_av == 0:
            print("Artificial viscosity disabled")
        elif use_av == 1:
            print("Artificial viscosity using modified physical viscosity")
            print("Using velocity divergence indicator")
            print(f"Shock capturing parameters: alpha {alpha_sc}, "
                  f"gamma_sc {gamma_sc}"
                  f"theta_sc {theta_sc}, beta_sc {beta_sc}, Pr 0.75, "
                  f"stagnation temperature {static_temp}")
        elif use_av == 2:
            print("Artificial viscosity using modified transport properties")
            print("\t mu, beta, kappa")
            # MJA update this
            print(f"Shock capturing parameters:"
                  f"\n\tav_mu {av2_mu0}"
                  f"\n\tav_beta {av2_beta0}"
                  f"\n\tav_kappa {av2_kappa0}"
                  f"\n\tav_prantdl {av2_prandtl0}"
                  f"\nstagnation temperature {static_temp}")
        elif use_av == 3:
            print("Artificial viscosity using modified transport properties")
            print("\t mu, beta, kappa, D")
            print(f"Shock capturing parameters:"
                  f"\tav_mu {av2_mu0}"
                  f"\tav_beta {av2_beta0}"
                  f"\tav_kappa {av2_kappa0}"
                  f"\tav_d {av2_d0}"
                  f"\tav_prantdl {av2_prandtl0}"
                  f"stagnation temperature {static_temp}")
        else:
            error_message = "Unknown artifical viscosity model {}".format(use_av)
            raise RuntimeError(error_message)

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if constant_cfl == 1:
            print(f"\tConstant cfl mode, current_cfl = {current_cfl}")
        else:
            print(f"\tConstant dt mode, current_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tdimension = {dim}")
        print(f"\tTime integration {integrator}")
        print("   Boundary Conditions:")
        bnd_msg = ""
        for bname, bsetting in bndry_config.items():
            msg_action = "Checking for" if bsetting else "Ignoring"
            bnd_msg = bnd_msg + f"\t{msg_action} {bname} boundary in mesh.\n"
        if rank == 0:
            print(bnd_msg)

        if noslip:
            print("\tInterface wall boundary conditions are noslip for velocity")
        else:
            print("\tInterface wall boundary conditions are slip for velocity")

        print("#### Simluation control data: ####\n")

    if rank == 0:
        print("\n#### Visualization setup: ####")
        if viz_level >= 0:
            print("\tBasic visualization output enabled.")
            print("\t(cv, dv, cfl)")
        if viz_level >= 1:
            print("\tExtra visualization output enabled for derived quantities.")
            print("\t(velocity, mass_fractions, etc.)")
        if viz_level >= 2:
            print("\tNon-dimensional parameter visualization output enabled.")
            print("\t(Re, Pr, etc.)")
        if viz_level >= 3:
            print("\tDebug visualization output enabled.")
            print("\t(rhs, grad_cv, etc.)")
        if viz_interval_type == 0:
            print(f"\tWriting viz data every {nviz} steps.")
        if viz_interval_type == 1:
            print(f"\tWriting viz data roughly every {t_viz_interval} seconds.")
        if viz_interval_type == 2:
            print(f"\tWriting viz data exactly every {t_viz_interval} seconds.")
        print("#### Visualization setup: ####")

    if rank == 0:
        print("\n#### Simluation initialization data: ####")
        if init_case == "y3prediction" or init_case == "y3prediction_ramp":
            print("\tInitializing flow to y3prediction")
            print(f"\tInflow stagnation pressure {total_pres_inflow}")
            print(f"\tInflow stagnation temperature {total_temp_inflow}")
            print(f"\tInjection stagnation pressure {total_pres_inj}")
            print(f"\tInjection stagnation temperature {total_temp_inj}")
            print("\tUpstream injection stagnation pressure"
                  f"{total_pres_inj_upstream}")
            print("\tUpstream injection stagnation temperature"
                  f"{total_temp_inj_upstream}")
        elif init_case == "shock1d":
            print("\tInitializing flow to shock1d")
            print(f"Shock Mach number {mach}")
            print(f"Ambient pressure {pres_bkrnd}")
            print(f"Ambient temperature {temp_bkrnd}")
        elif init_case == "backward_step":
            print("\tInitializing flow to backward_step")
            print(f"Shock Mach number {mach}")
            print(f"Ambient pressure {pres_bkrnd}")
            print(f"Ambient temperature {temp_bkrnd}")
        elif init_case == "forward_step":
            print("\tInitializing flow to forward_step")
            print(f"Shock Mach number {mach}")
            print(f"Ambient pressure {pres_bkrnd}")
            print(f"Ambient temperature {temp_bkrnd}")
        elif init_case == "mixing_layer":
            print("\tInitializing flow to mixing_layer")
            print(f"Vorticity thickness {vorticity_thickness}")
            print(f"Ambient pressure {pres_bkrnd}")
        elif init_case == "mixing_layer_hot":
            print("\tInitializing flow to mixing_layer_hot")
            print(f"Vorticity thickness {vorticity_thickness}")
            print(f"Ambient pressure {pres_bkrnd}")
        elif init_case == "flame1d":
            print("\tInitializing flow to flame1d")
            print(f"Ambient pressure {pres_bkrnd}")
            print(f"Ambient temperature {temp_bkrnd}")
        elif init_case == "species_diffusion":
            print("\tInitializing flow to species diffusion")
            print(f"Ambient pressure {pres_bkrnd}")
            print(f"Ambient temperature {temp_bkrnd}")
        elif init_case == "wedge":
            print("\tInitializing flow to wedge")
            print(f"Shock Mach number {mach}")
            print(f"Ambient pressure {pres_bkrnd}")
            print(f"Ambient temperature {temp_bkrnd}")
        elif init_case == "unstart" or init_case == "unstart_ramp":
            print("\tInitializing flow to unstart")
            print(f"\tInflow stagnation pressure {total_pres_inflow}")
            print(f"\tInflow stagnation temperature {total_temp_inflow}")
            print(f"Ambient pressure {pres_bkrnd}")
            print(f"Ambient temperature {temp_bkrnd}")
        elif init_case == "discontinuity":
            print("\tInitializing flow to discontinuity")
            print(f"Pressure left {pres_left}")
            print(f"Pressure right {pres_right}")
            print(f"Temperature left {temp_left}")
            print(f"Temperature right {temp_right}")
            print(f"Sigma {sigma_disc}")
        else:
            raise SimulationConfigurationError(
                "Invalid initialization configuration specified"
                "Currently supported options are: "
                "\t y3prediction"
                "\t unstart"
                "\t unstart_ramp"
                "\t shock1d"
                "\t backward_step"
                "\t forward_step"
                "\t discontinuity"
                "\t flame1d"
                "\t wedge"
                "\t mixing_layer"
                "\t mixing_layer_hot"
                "\t species_diffusion"
            )
        print("#### Simluation initialization data: ####")

        print("\n#### Simluation setup data: ####")
        print(f"\tvel_sigma = {vel_sigma}")
        print(f"\ttemp_sigma = {temp_sigma}")
        print(f"\tvel_sigma_injection = {vel_sigma_inj}")
        print(f"\ttemp_sigma_injection = {temp_sigma_inj}")
        print("#### Simluation setup data: ####")

    # spark ignition
    spark_center = np.zeros(shape=(dim,))
    spark_center[0] = spark_init_loc_x
    spark_center[1] = spark_init_loc_y
    if dim == 3:
        spark_center[2] = spark_init_loc_z
    if rank == 0 and use_ignition > 0:
        print("\n#### Ignition control parameters ####")
        print(f"spark center ({spark_center[0]},{spark_center[1]})")
        print(f"spark FWHM {spark_diameter}")
        print(f"spark strength {spark_strength}")
        print(f"ignition time {spark_init_time}")
        print(f"ignition duration {spark_duration}")
        if use_ignition == 1:
            print("spark ignition")
        elif use_ignition == 2:
            print("heat source ignition")
        print("#### Ignition control parameters ####\n")

    if rank == 0 and use_injection_source_3d is True:
        print("injection source 3D ... it's complicated")

    # injection mass source
    injection_source_center = np.zeros(shape=(dim,))
    injection_source_center[0] = injection_source_loc_x
    injection_source_center[1] = injection_source_loc_y
    if dim == 3:
        injection_source_center[2] = injection_source_loc_z

    # combustor injection mass source
    injection_source_center_comb = np.zeros(shape=(dim,))
    injection_source_center_comb[0] = injection_source_loc_x_comb
    injection_source_center_comb[1] = injection_source_loc_y_comb
    if dim == 3:
        injection_source_center_comb[2] = injection_source_loc_z_comb

    if rank == 0 and use_injection_source_comb is True:
        print("\n#### Injection source control parameters ####")
        if dim == 2:
            print("combustor injection source center ("
                  f"{injection_source_center_comb[0]},"
                  f"{injection_source_center_comb[1]})")
        else:
            print("combustor injection source center ("
                  f"{injection_source_center_comb[0]},"
                  f"{injection_source_center_comb[1]},"
                  f"{injection_source_center_comb[2]})")

    if rank == 0 and use_injection_source is True:
        print("\n#### Injection source control parameters ####")
        if dim == 2:
            print("injection source center ("
                  f"{injection_source_center[0]},"
                  f"{injection_source_center[1]})")
        else:
            print("injection source center ("
                  f"{injection_source_center[0]},"
                  f"{injection_source_center[1]},"
                  f"{injection_source_center[2]})")

    if rank == 0 and use_injection_source is True:
        print(f"injection source FWHM {injection_source_diameter}")
        print(f"injection source mass {injection_source_mass}")
        print(f"injection source energy {injection_source_energy}")
        if dim == 2:
            print("injection source mom ("
                  f"{injection_source_mom_x}",
                  f"{injection_source_mom_y}")
        else:
            print("injection source mom ("
                  f"{injection_source_mom_x}",
                  f"{injection_source_mom_y}",
                  f"{injection_source_mom_z}")
        print(f"injection source start time {injection_source_init_time}")
        print("#### Injection source control parameters ####\n")

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "ssprk43":
        timestepper = ssprk43_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step
    if integrator == "compiled_lsrk54":
        timestepper = _compiled_stepper_wrapper

    flux_msg = "\nSetting inviscid numerical flux to: "
    if use_esdg:
        try:
            from mirgecom.inviscid import entropy_stable_inviscid_facial_flux_rusanov
        except ImportError:
            raise SimulationConfigurationError(
                "ESDG option specified, but MIRGE-Com "
                "is installed without ESDG support. "
                "Try switching your MIRGE-Com branch to "
                "mirgecom@production."
            )
        inviscid_numerical_flux_func = entropy_stable_inviscid_facial_flux_rusanov
        flux_msg = flux_msg + "ESDG/Rusanov with EC/"
        if nspecies == 7:  # FIXME: Add support for 7 passive species?
            inv_flux_type = "Renac for mixtures.\n"
        else:
            inv_flux_type = "Chandrashekar for single gas or passive species.\n"
        flux_msg = flux_msg + inv_flux_type
    else:
        inviscid_numerical_flux_func = inviscid_facial_flux_rusanov
        flux_msg = flux_msg + "Rusanov\n"

        if inv_num_flux == "hll":
            inviscid_numerical_flux_func = inviscid_facial_flux_hll
            flux_msg = flux_msg + "HLL\n"

    flux_msg = flux_msg + "Setting viscous numerical flux to: "
    if use_wall:
        viscous_numerical_flux_func = viscous_facial_flux_harmonic
        flux_msg = flux_msg + "Harmonic\n"
    else:
        viscous_numerical_flux_func = viscous_facial_flux_central
        flux_msg = flux_msg + "Central\n"

    if rank == 0:
        print(flux_msg)

    # }}}

    # constants
    mw_o = 15.999
    mw_o2 = mw_o*2
    mw_co = 28.010
    mw_n2 = 14.0067*2
    mw_c2h4 = 28.05
    mw_h2 = 1.00784*2
    mw_ar = 39.948
    univ_gas_const = 8314.59

    # working gas: O2/N2 #
    #   O2 mass fraction 0.273
    #   gamma = 1.4
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma = 1.4
    mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)

    if gas_mat_prop == 1:
        # working gas: Ar #
        #   O2 mass fraction 0.273
        #   gamma = 1.4
        #   cp = 37.135 J/mol-K,
        #   rho= 1.977 kg/m^3 @298K
        gamma = 5/3
        mw = mw_ar

    if fluid_gamma > 0:
        gamma = fluid_gamma

    mf_c2h4 = mole_c2h4*mw_c2h4/(mole_c2h4*mw_c2h4 + mole_h2*mw_h2)
    mf_h2 = 1 - mf_c2h4

    # user can reset the mw to whatever they want
    if fluid_mw > 0:
        mw = fluid_mw

    r = univ_gas_const/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75

    # viscosity @ 400C, Pa-s
    # working gas: O2/N2 #
    mu_o2 = 3.76e-5
    mu_n2 = 3.19e-5
    mu = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    beta = 0.

    if gas_mat_prop == 1:
        # working gas: Ar #
        mu_ar = 4.22e-5
        mu = mu_ar
    if not fluid_mu < 0:
        mu = fluid_mu
    if not fluid_beta < 0:
        beta = fluid_beta

    kappa = cp*mu/Pr
    if fluid_kappa > 0:
        kappa = fluid_kappa
    init_temperature = 300.0

    # don't allow limiting on flows without species
    if nspecies == 0:
        use_injection = False
        use_upstream_injection = False

    # Turn off combustion unless EOS supports it
    if nspecies < 4:
        use_combustion = False

    if nspecies > 3:
        eos_type = 1

    pyro_mech_name = configurate("pyro_mech", input_data, "uiuc_sharp")
    pyro_mech_name_full = f"y3prediction.pyro_mechs.{pyro_mech_name}"

    import importlib
    pyromechlib = importlib.import_module(pyro_mech_name_full)

    if rank == 0:
        print("\n#### Simluation material properties: ####")
        print("#### Fluid domain: ####")
        print(f"\tmu = {mu}")
        print(f"\tkappa = {kappa}")
        print(f"\tPrandtl Number  = {Pr}")
        print(f"\tnspecies = {nspecies}")
        if nspecies == 0:
            print("\tno passive scalars, uniform species mixture")
            if gas_mat_prop == 0:
                print("\tO2/N2 mix material properties.")
            else:
                print("\tAr material properties.")
            print(f"\tspecific gas constant = {r}")
            print(f"\tcp = {cp}")
        elif nspecies <= 3:
            print("\tpassive scalars to track air/fuel/inert mixture, ideal gas eos")
        elif nspecies == 5:
            print("\tfull multi-species initialization with pyrometheus eos")
            print("\tno combustion source terms")
        else:
            print("\tfull multi-species initialization with pyrometheus eos")
            print("\tcombustion source terms enabled")

        if eos_type == 0:
            print("\tIdeal Gas EOS")
        elif eos_type == 1:
            print("\tPyrometheus EOS")
            print(f"\tPyro mechanism {pyro_mech_name}")

        if use_species_limiter == 1:
            print("\nSpecies mass fractions limited to [0:1]")

        if use_wall:
            print("#### Wall domain: ####")

            if wall_material == 0:
                print("\tNon-reactive wall model")
            elif wall_material == 1:
                print("\tReactive wall model for non-porous media")
            elif wall_material == 2:
                print("\tReactive wall model for porous media")
            else:
                error_message = "Unknown wall_material {}".format(wall_material)
                raise RuntimeError(error_message)

            if use_wall_ox:
                print("\tWall oxidizer transport enabled")
            else:
                print("\tWall oxidizer transport disabled")

            if use_wall_mass:
                print("\t Wall mass loss enabled")
            else:
                print("\t Wall mass loss disabled")

            print(f"\tWall density = {wall_insert_rho}")
            print(f"\tWall cp = {wall_insert_cp}")
            print(f"\tWall O2 diff = {wall_insert_ox_diff}")
            print(f"\tWall surround density = {wall_surround_rho}")
            print(f"\tWall surround cp = {wall_surround_cp}")
            print(f"\tWall surround kappa = {wall_surround_kappa}")
            print(f"\tWall time scale = {wall_time_scale}")
            print(f"\tWall penalty = {wall_penalty_amount}")
        else:
            print("\tWall model disabled")
            use_wall_ox = False
            use_wall_mass = False

        print("#### Simluation material properties: ####")

    chem_source_tol = 1.e-10
    # make the eos
    if eos_type == 0:
        eos = IdealSingleGas(gamma=gamma, gas_const=r)
        eos_init = eos
    else:
        from mirgecom.thermochemistry import get_pyrometheus_wrapper_class
        pyro_mech = get_pyrometheus_wrapper_class(
            pyro_class=pyromechlib.Thermochemistry, temperature_niter=pyro_temp_iter,
            zero_level=chem_source_tol)(actx.np)
        eos = PyrometheusMixture(pyro_mech, temperature_guess=init_temperature)
        # seperate gas model for initialization,
        # just to make sure we get converged temperature
        pyro_mech_init = get_pyrometheus_wrapper_class(
            pyro_class=pyromechlib.Thermochemistry, temperature_niter=5,
            zero_level=chem_source_tol)(actx.np)
        eos_init = PyrometheusMixture(pyro_mech_init,
                                      temperature_guess=init_temperature)

    # set the species names
    if eos_type == 0:
        species_names = ["inert"]
        if nspecies == 2:
            species_names = ["air", "fuel"]
        elif nspecies == 3:
            species_names = ["air", "fuel", "inert"]
    else:
        species_names = pyro_mech.species_names

    print(f"{species_names=}")

    # initialize eos and species mass fractions
    y = np.zeros(nspecies, dtype=object)
    y_fuel = np.zeros(nspecies)
    if nspecies == 2:
        y[0] = 1
        y_fuel[1] = 1
    elif nspecies > 4:
        # find name species indicies
        i_c2h4 = -1
        i_h2 = -1
        i_ox = -1
        i_di = -1
        for i in range(nspecies):
            try:
                if species_names[i] == "C2H4":
                    i_c2h4 = i
                if species_names[i] == "H2":
                    i_h2 = i
                if species_names[i] == "O2":
                    i_ox = i
                if species_names[i] == "N2":
                    i_di = i
            except IndexError:
                continue

        # Set the species mass fractions to the free-stream flow
        y[i_ox] = mf_o2
        y[i_di] = 1. - mf_o2
        # Set the species mass fractions to the free-stream flow
        y_fuel[i_c2h4] = mf_c2h4
        y_fuel[i_h2] = mf_h2

    # initialize the transport model
    transport_alpha = 0.6
    transport_beta = 4.093e-7
    transport_sigma = 2.0
    transport_n = 0.666

    # use the species names to populate the default species diffusivities
    default_species_diffusivity = {}
    for species in species_names:
        default_species_diffusivity[species] = spec_diff

    input_species_diffusivity = configurate(
        "species_diffusivity", input_data, default_species_diffusivity)

    # now read the diffusivities from input
    print(f"{input_species_diffusivity}")

    species_diffusivity = spec_diff * np.ones(nspecies)
    for i in range(nspecies):
        species_diffusivity[i] = input_species_diffusivity[species_names[i]]

    transport_le = None
    if use_lewis_transport:
        transport_le = np.ones(nspecies,)

        if nspecies > 4:
            transport_le[i_h2] = 0.2

    if rank == 0:
        if transport_type == 0:
            print("\t Simple transport model:")
            print("\t\t constant viscosity, species diffusivity")
            print(f"\tmu = {mu}")
            print(f"\tkappa = {kappa}")
            print(f"\tspecies diffusivity = {spec_diff}")
        elif transport_type == 1:
            print("\t Power law transport model:")
            print("\t\t temperature dependent viscosity, species diffusivity")
            print(f"\ttransport_alpha = {transport_alpha}")
            print(f"\ttransport_beta = {transport_beta}")
            print(f"\ttransport_sigma = {transport_sigma}")
            print(f"\ttransport_n = {transport_n}")
            if use_lewis_transport:
                print(f"\ttransport Lewis Number = {transport_le}")
            else:
                print(f"\tspecies diffusivity = {species_diffusivity}")
        elif transport_type == 2:
            print("\t Pyrometheus transport model:")
            print("\t\t temperature/mass fraction dependence")
        else:
            error_message = "Unknown transport_type {}".format(transport_type)
            raise RuntimeError(error_message)

    physical_transport_model = SimpleTransport(
        viscosity=mu, bulk_viscosity=beta,
        thermal_conductivity=kappa,
        species_diffusivity=species_diffusivity)

    if transport_type == 1:
        physical_transport_model = PowerLawTransport(
            alpha=transport_alpha, beta=transport_beta,
            sigma=transport_sigma, n=transport_n,
            lewis=transport_le,
            species_diffusivity=species_diffusivity)

    transport_model = physical_transport_model
    if use_av == 1:
        transport_model = ArtificialViscosityTransportDiv(
            physical_transport=physical_transport_model,
            av_mu=alpha_sc, av_prandtl=0.75)
    elif use_av == 2:
        transport_model = ArtificialViscosityTransportDiv2(
            physical_transport=physical_transport_model,
            av_mu=av2_mu0, av_beta=av2_beta0, av_kappa=av2_kappa0,
            av_prandtl=av2_prandtl0)
    elif use_av == 3:
        transport_model = ArtificialViscosityTransportDiv3(
            physical_transport=physical_transport_model,
            av_mu=av2_mu0, av_beta=av2_beta0,
            av_kappa=av2_kappa0, av_d=av2_d0,
            av_prandtl=av2_prandtl0)

    # with transport and eos sorted out, build the gas model
    gas_model = GasModel(eos=eos, transport=transport_model)

    # quiescent initialization
    velocity_bkrnd = np.zeros(shape=(dim,))
    velocity_bkrnd[0] = vel_bkrnd
    bulk_init = Uniform(
        dim=dim,
        velocity=velocity_bkrnd,
        pressure=pres_bkrnd,
        temperature=temp_bkrnd,
        species_mass_fractions=y
    )

    # select the initialization case
    if init_case == "discontinuity":

        # init params
        disc_location = np.zeros(shape=(dim,))
        fuel_location = np.zeros(shape=(dim,))

        disc_location[0] = shock_loc_x
        fuel_location[0] = fuel_loc_x

        # normal shock properties for a calorically perfect gas
        # state 1: pre-shock
        # state 2: post-shock
        rho_bkrnd = pres_bkrnd/r/temp_bkrnd
        c_bkrnd = math.sqrt(gamma*pres_bkrnd/rho_bkrnd)
        velocity1 = -mach*c_bkrnd

        gamma1 = gamma
        gamma2 = gamma

        vel_left = np.zeros(shape=(dim,))
        vel_right = np.zeros(shape=(dim,))
        vel_cross = np.zeros(shape=(dim,))

        plane_normal = np.zeros(shape=(dim,))
        theta = mesh_angle/180.*np.pi
        plane_normal[0] = np.cos(theta)
        plane_normal[1] = np.sin(theta)
        plane_normal = plane_normal/np.linalg.norm(plane_normal)

        bulk_init = PlanarDiscontinuityMulti(
            dim=dim,
            nspecies=nspecies,
            disc_location=disc_location,
            disc_location_species=fuel_location,
            normal_dir=plane_normal,
            sigma=sigma_disc,
            pressure_left=pres_left,
            pressure_right=pres_right,
            temperature_left=temp_left,
            temperature_right=temp_right,
            velocity_left=vel_left,
            velocity_right=vel_right,
            velocity_cross=vel_cross,
            species_mass_left=y,
            species_mass_right=y_fuel,
            temp_wall=temp_bkrnd,
            vel_sigma=vel_sigma,
            temp_sigma=temp_sigma)

    elif (init_case == "forward_step" or
          init_case == "backward_step"):

        print(f"{pres_bkrnd=}")
        print(f"{temp_bkrnd=}")

        # initialization to uniform M=mach flow
        velocity_bkrnd = np.zeros(dim, dtype=object)
        if use_axisymmetric or dim == 3:
            velocity_bkrnd[1] = vel_bkrnd
        else:
            velocity_bkrnd[0] = vel_bkrnd

        mass_bkrnd = eos.get_density(pressure=pres_bkrnd, temperature=temp_bkrnd,
                                     species_mass_fractions=y)
        energy_bkrnd = mass_bkrnd*eos.get_internal_energy(temperature=temp_bkrnd,
                                                          species_mass_fractions=y)
        cv_bkrnd = make_conserved(dim=dim, mass=mass_bkrnd,
                                  momentum=mass_bkrnd*velocity_bkrnd,
                                  energy=energy_bkrnd, species_mass=mass_bkrnd*y)
        gamma = eos.gamma(cv_bkrnd, temp_bkrnd)

        c_bkrnd = np.sqrt(gamma*pres_bkrnd/mass_bkrnd)

        if use_axisymmetric:
            velocity_bkrnd[1] = velocity_bkrnd[1] + c_bkrnd*mach
        else:
            velocity_bkrnd[0] = velocity_bkrnd[0] + c_bkrnd*mach

        ysp = y
        if nspecies == 2:
            ysp[0] = 0.25
            ysp[1] = 0.75
        if nspecies == 3:
            ysp[0] = 0.39
            ysp[1] = 0.35
            ysp[2] = 0.26
        bulk_init = Uniform(
            dim=dim,
            velocity=velocity_bkrnd,
            pressure=pres_bkrnd,
            temperature=temp_bkrnd,
            species_mass_fractions=ysp
        )

    elif (init_case == "shock1d"):

        # init params
        disc_location = np.zeros(shape=(dim,))
        fuel_location = np.zeros(shape=(dim,))

        disc_location[0] = shock_loc_x
        fuel_location[0] = fuel_loc_x

        # normal shock properties for a calorically perfect gas
        # state 1: pre-shock
        # state 2: post-shock
        rho_bkrnd = pres_bkrnd/r/temp_bkrnd
        c_bkrnd = math.sqrt(gamma*pres_bkrnd/rho_bkrnd)
        velocity1 = -mach*c_bkrnd

        gamma1 = gamma
        gamma2 = gamma

        rho1 = rho_bkrnd
        pressure1 = pres_bkrnd
        temperature1 = temp_bkrnd

        pressure_ratio = (2.*gamma*mach*mach-(gamma-1.))/(gamma+1.)
        density_ratio = (gamma+1.)*mach*mach/((gamma-1.)*mach*mach+2.)

        rho2 = rho1*density_ratio
        pressure2 = pressure1*pressure_ratio
        temperature2 = pressure2/rho2/r
        # shock stationary frame
        velocity2 = velocity1*(1/density_ratio)
        temp_wall = temperature1

        # for non-calorically perfect gas, we iterate on the density ratio,
        # until we converge
        if eos_type > 0:
            if shock_loc_x < fuel_loc_x:
                y1 = y
                y2 = y
            else:
                y1 = y_fuel
                y2 = y_fuel

            rho1 = pyro_mech.get_density(pressure1, temperature1, y1)

            # guess a density ratio (rho1/rho2)
            density_ratio = 0.1
            rho2 = rho1/density_ratio
            enthalpy1 = gas_model.eos.get_internal_energy(
                temperature1, y1) + pressure1/rho1
            # iteratively solve the shock hugoniot
            error = 100
            while error > 1e-8:
                pressure2 = pressure1 + rho1*velocity1**2*(1 - density_ratio)
                enthalpy2 = enthalpy1 + 0.5*velocity1**2*(1 - (density_ratio)**2)

                # find temperature from new energy and get an updated density
                energy2 = enthalpy2 - pressure2/rho2
                temperature2 = pyro_mech.get_temperature(energy2, temperature2, y2)
                rho2_old = rho2
                rho2 = pyro_mech.get_density(pressure2, temperature2, y2)

                # compute the error in density and form a new density ratio
                error = np.abs((rho2 - rho2_old)/rho2_old)
                density_ratio = rho1/rho2
                velocity2 = velocity1*(density_ratio)

            gamma1 = (pyro_mech.get_mixture_specific_heat_cp_mass(temperature1, y1) /
                      pyro_mech.get_mixture_specific_heat_cv_mass(temperature1, y1))
            gamma2 = (pyro_mech.get_mixture_specific_heat_cp_mass(temperature2, y1) /
                      pyro_mech.get_mixture_specific_heat_cv_mass(temperature2, y1))

        # convert to shock moving frame
        velocity2 = velocity2 - velocity1
        velocity1 = 0.

        vel_left = np.zeros(shape=(dim,))
        vel_right = np.zeros(shape=(dim,))
        vel_cross = np.zeros(shape=(dim,))
        vel_cross[1] = 0

        plane_normal = np.zeros(shape=(dim,))
        theta = mesh_angle/180.*np.pi
        plane_normal[0] = np.cos(theta)
        plane_normal[1] = np.sin(theta)
        plane_normal = plane_normal/np.linalg.norm(plane_normal)

        disc_location = shock_loc_x*plane_normal
        fuel_location = fuel_loc_x*plane_normal
        vel_left = (velocity2 - velocity1)*plane_normal

        pressure1_total = pres_bkrnd*(1 + (gamma-1)/2*mach**2)**(gamma/(gamma-1))
        temperature1_total = temp_bkrnd*(1 + (gamma-1)/2*mach**2)

        mach2 = vel_left[0]/np.sqrt(gamma2*pressure2/rho2)
        pressure2_total = pressure2*(1 + (gamma-1)/2*mach2**2)**(gamma/(gamma-1))
        temperature2_total = temperature2*(1 + (gamma-1)/2*mach2**2)

        if rank == 0:
            print("#### Simluation initialization data: ####")
            print(f"\tshock Mach number {mach}")
            print(f"\tpre-shock gamma {gamma1}")
            print(f"\tpre-shock temperature {temperature1}")
            print(f"\tpre-shock pressure {pressure1}")
            print(f"\tpre-shock rho {rho1}")
            print(f"\tpre-shock velocity {velocity1}")
            print(f"\tpre-shock total pressure {pressure1_total}")
            print(f"\tpre-shock total temperature {temperature1_total}")

            print(f"\tpost-shock gamma {gamma2}")
            print(f"\tpost-shock temperature {temperature2}")
            print(f"\tpost-shock pressure {pressure2}")
            print(f"\tpost-shock rho {rho2}")
            print(f"\tpost-shock velocity {velocity2}")
            print(f"\tpost-shock total pressure {pressure2_total}")
            print(f"\tpost-shock total temperature {temperature2_total}")
            print(f"\tpost-shock mach {mach2}")

        bulk_init = PlanarDiscontinuityMulti(
            dim=dim,
            nspecies=nspecies,
            disc_location=disc_location,
            disc_location_species=fuel_location,
            normal_dir=plane_normal,
            sigma=0.001,
            pressure_left=pressure2,
            pressure_right=pressure1,
            temperature_left=temperature2,
            temperature_right=temperature1,
            velocity_left=vel_left,
            velocity_right=vel_right,
            velocity_cross=vel_cross,
            species_mass_left=y,
            species_mass_right=y_fuel,
            temp_wall=temp_bkrnd,
            vel_sigma=vel_sigma,
            temp_sigma=temp_sigma)

    if init_case == "mixing_layer":
        temperature = 300.
        pressure = 101325.

        y_mix_air = np.zeros(nspecies, dtype=object)
        y_mix_fuel = np.zeros(nspecies, dtype=object)

        # fuel is H2:0.5 N2:0.5 mole fraction
        y_mix_fuel[0] = 0.5*mw_h2/(0.5*mw_h2 + 0.5*mw_n2)
        y_mix_fuel[8] = 0.5*mw_n2/(0.5*mw_h2 + 0.5*mw_n2)
        # air is O2:0.21 N2:0.79 mole fraction
        y_mix_air[2] = 0.21*mw_o2/(0.21*mw_o2 + 0.79*mw_n2)
        y_mix_air[8] = 1 - y_mix_air[2]

        from y3prediction.mixing_layer import MixingLayerCold
        bulk_init = MixingLayerCold(
            dim=dim, nspecies=nspecies,
            mach_fuel=0.2, mach_air=0.3,
            temp_fuel=300, temp_air=500,
            y_fuel=y_mix_fuel, y_air=y_mix_air,
            vorticity_thickness=vorticity_thickness,
            pressure=pres_bkrnd
        )
    if init_case == "mixing_layer_hot":
        if rank == 0:
            print("Initializing hot mixing layer")

        import h5py

        def get_data_from_hdf5(group):
            data_dict = {}
            for key in group.keys():
                if isinstance(group[key], h5py.Group):
                    # If the key is a group, recursively explore it
                    subgroup_data = get_data_from_hdf5(group[key])
                    data_dict.update(subgroup_data)
                elif isinstance(group[key], h5py.Dataset):
                    # If it's a dataset, add it to the dictionary
                    data_dict[key] = group[key][()]
            return data_dict

        # Usage example
        inflow_fname = "r_mixing_layer_inflow.h5"
        with h5py.File(inflow_fname, "r") as hf:
            inflow_data = get_data_from_hdf5(hf)

        inflow_data = None
        if rank == 0:
            inflow_fname = "r_mixing_layer_inflow.h5"
            with h5py.File(inflow_fname, "r") as hf:
                inflow_data = get_data_from_hdf5(hf)

        inflow_data = comm.bcast(inflow_data, root=0)
        #print(f"{inflow_data=}")

        pressure = 101325.
        from y3prediction.mixing_layer import MixingLayerHot
        bulk_init = MixingLayerHot(
            dim=dim, nspecies=nspecies,
            inflow_profile=inflow_data,
            pressure=pressure
        )

    elif init_case == "flame1d":

        # init params
        disc_location = np.zeros(shape=(dim,))
        fuel_location = np.zeros(shape=(dim,))

        # the init is set up to keep species constant across the shock, so put the
        # fuel and shock discontinuities on top of each other
        disc_location[0] = shock_loc_x
        fuel_location[0] = shock_loc_x

        #mech_data = get_mechanism_input("uiuc_updated")
        mech_file = (f"{pyro_mech_name}.yaml")

        print(f"{mech_file=}")
        import cantera
        cantera_soln = cantera.Solution(f"{mech_file}", "gas")

        # Initial temperature, pressure, and mixutre mole fractions are needed to
        # set up the initial state in Cantera.
        temp_unburned = 300.0
        temp_ignition = 2000.0
        # Parameters for calculating the amounts of fuel, oxidizer, and inert species
        # for pure C2H4
        stoich_ratio = 3.0
        equiv_ratio = 1.0
        ox_di_ratio = 0.21
        # Grab the array indices for specific species
        i_fu = cantera_soln.species_index("C2H4")
        i_ox = cantera_soln.species_index("O2")
        i_di = cantera_soln.species_index("N2")
        x = np.zeros(nspecies)
        # Set the species mole fractions according to our desired fuel/air mixture
        x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
        x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
        x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
        pres_unburned = 101325.0

        # Let the user know about how Cantera is being initilized
        print(f"Input state (T,P,X) = ({temp_unburned}, {pres_unburned}, {x}")
        # Set Cantera internal gas temperature, pressure, and mole fractios
        cantera_soln.TPX = temp_unburned, pres_unburned, x
        # Pull temperature, total density, mass fractions, and pressure from Cantera
        # We need total density, and mass fractions to initialize the state.
        y_unburned = np.zeros(nspecies)
        can_t, rho_unburned, y_unburned = cantera_soln.TDY

        # *can_t*, *can_p* should not differ (significantly) from user's initial data
        # but we want to use exactly the same starting point as Cantera,
        # so we use Cantera's version of these data.

        # now find the conditions for the burned gas
        cantera_soln.TP = temp_ignition, pres_unburned
        cantera_soln.equilibrate("TP")
        temp_burned, rho_burned, y_burned = cantera_soln.TDY
        pres_burned = cantera_soln.P

        if rank == 0:
            print("#### Simluation initialization data: ####")
            #print(f"\tflame speed {mach}")
            #print(f"\tunburned gamma {gamma1}")
            print(f"\tunburned temperature {temp_unburned}")
            print(f"\tunburned pressure {pres_burned}")
            print(f"\tunburned rho {rho_unburned}")
            for i in range(nspecies):
                print(f"\tunburned Y[{species_names[i]}] {y_unburned[i]}")

            #print(f"\tburned gamma {gamma2}")
            print(f"\tburned temperature {temp_burned}")
            print(f"\tburned pressure {pres_burned}")
            print(f"\tburned rho {rho_burned}")
            for i in range(nspecies):
                print(f"\tburned Y[{species_names[i]}] {y_burned[i]}")

        vel_burned = np.zeros(shape=(dim,))
        vel_unburned = np.zeros(shape=(dim,))
        plane_normal = np.zeros(shape=(dim,))
        plane_normal[0] = 1

        #return;

        bulk_init = PlanarDiscontinuityMulti(
            dim=dim,
            nspecies=nspecies,
            disc_location=disc_location,
            disc_location_species=fuel_location,
            normal_dir=plane_normal,
            sigma=0.001,
            pressure_left=pres_unburned,
            pressure_right=pres_burned,
            temperature_left=temp_unburned,
            temperature_right=temp_burned,
            velocity_left=vel_unburned,
            velocity_right=vel_burned,
            species_mass_left=y_unburned,
            species_mass_right=y_burned,
            temp_wall=temp_bkrnd,
            vel_sigma=vel_sigma,
            temp_sigma=temp_sigma)
    elif init_case == "species_diffusion":

        velocity = np.zeros(shape=(dim,))
        pressure = pres_bkrnd
        temperature = temp_bkrnd
        rho = pressure/r/temperature

        centers = make_obj_array([np.zeros(shape=(dim,)) for i in range(nspecies)])
        spec_y0s = np.zeros(shape=(nspecies,))
        spec_amplitudes = .5*np.ones(shape=(nspecies,))

        if rank == 0:
            print("#### Simluation initialization data: ####")
            print(f"\ttemperature {temperature}")
            print(f"\tpressure {pressure}")
            print(f"\trho {rho}")
            print(f"\tvelocity {velocity}")

        bulk_init = MulticomponentLump(
            dim=dim, nspecies=nspecies,
            rho0=rho, p0=pressure, velocity=velocity,
            spec_centers=centers,
            spec_y0s=spec_y0s,
            spec_amplitudes=spec_amplitudes,
            sigma=0.1
        )

    elif init_case == "wedge":

        velocity = np.zeros(shape=(dim,))
        temperature = 300.
        pressure = 100000.
        rho = pressure/r/temperature
        c = np.sqrt(gamma*pressure/rho)
        velocity[1] = c*mach

        if rank == 0:
            print("#### Simluation initialization data: ####")
            print(f"\tshock Mach number {mach}")
            print(f"\ttemperature {temperature}")
            print(f"\tpressure {pressure}")
            print(f"\trho {rho}")
            print(f"\tvelocity {velocity}")

        bulk_init = Uniform(
            dim=dim,
            velocity=velocity,
            pressure=pressure,
            temperature=temperature
        )

    if init_case == "unstart":

        # init params
        disc_location = np.zeros(shape=(dim,))
        fuel_location = np.zeros(shape=(dim,))
        if dim == 2:
            disc_location[1] = shock_loc_x
        else:
            disc_location[0] = shock_loc_x
        fuel_location[1] = 10000.
        plane_normal = np.zeros(shape=(dim,))

        #
        # isentropic expansion based on the area ratios between the
        # inlet (r=54e-3m) and the throat (r=3.167e-3)
        #
        vel_inflow = np.zeros(shape=(dim,))
        vel_outflow = np.zeros(shape=(dim,))

        throat_height = 6.3028e-3
        inlet_area_ratio = inlet_height/throat_height
        if use_axisymmetric:
            inlet_area_ratio *= inlet_area_ratio

        inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                          gamma=gamma,
                                          mach_guess=0.01)
        pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                            P0=total_pres_inflow,
                                            gamma=gamma)
        temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                               T0=total_temp_inflow,
                                               gamma=gamma)

        if eos_type == 0:
            rho_inflow = pres_inflow/temp_inflow/r
            sos = math.sqrt(gamma*pres_inflow/rho_inflow)
            inlet_gamma = gamma
        else:
            rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                              temperature=temp_inflow,
                                              mass_fractions=y)
            inlet_gamma = (
                pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))

            gamma_error = (gamma - inlet_gamma)
            gamma_guess = inlet_gamma
            toler = 1.e-6
            # iterate over the gamma/mach since gamma = gamma(T)
            while gamma_error > toler:

                inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                                  gamma=gamma_guess,
                                                  mach_guess=0.01)
                pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                                    P0=total_pres_inflow,
                                                    gamma=gamma_guess)
                temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                                       T0=total_temp_inflow,
                                                       gamma=gamma_guess)

                rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                                  temperature=temp_inflow,
                                                  mass_fractions=y)
                inlet_gamma = \
                    (pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                     pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))
                gamma_error = (gamma_guess - inlet_gamma)
                gamma_guess = inlet_gamma

            sos = math.sqrt(inlet_gamma*pres_inflow/rho_inflow)

        theta = 0.0
        if dim == 2:
            vel_inflow[1] = inlet_mach*sos
            theta = np.pi/2.
        else:
            vel_inflow[0] = inlet_mach*sos

        plane_normal = np.zeros(shape=(dim,))
        plane_normal[0] = np.cos(theta)
        plane_normal[1] = np.sin(theta)
        plane_normal = plane_normal/np.linalg.norm(plane_normal)

        if rank == 0:
            print("#### Simluation initialization data: ####")
            print(f"\tinlet Mach number {inlet_mach}")
            print(f"\tinlet gamma {inlet_gamma}")
            print(f"\tinlet temperature {temp_inflow}")
            print(f"\tinlet pressure {pres_inflow}")
            print(f"\tinlet rho {rho_inflow}")
            print(f"\tinlet velocity x  {vel_inflow[0]}")
            print(f"\tinlet velocity y  {vel_inflow[1]}")
            #print(f"final inlet pressure {pres_inflow_final}")

        bulk_init = PlanarDiscontinuityMulti(
            dim=dim,
            nspecies=nspecies,
            disc_location=disc_location,
            disc_location_species=fuel_location,
            normal_dir=plane_normal,
            sigma=0.002,
            pressure_left=pres_inflow,
            pressure_right=pres_bkrnd,
            temperature_left=temp_inflow,
            temperature_right=temp_bkrnd,
            velocity_left=vel_inflow,
            velocity_right=vel_outflow,
            velocity_cross=vel_outflow,
            species_mass_left=y,
            species_mass_right=y_fuel,
            temp_wall=temp_bkrnd,
            y_top=0.013,
            y_bottom=-0.013,
            vel_sigma=vel_sigma,
            temp_sigma=temp_sigma)

    elif init_case == "unstart_ramp":

        #
        # isentropic expansion based on the area ratios between the
        # inlet (r=54e-3m) and the throat (r=3.167e-3)
        #
        vel_inflow = np.zeros(shape=(dim,))
        vel_outflow = np.zeros(shape=(dim,))
        vel_bulk = np.zeros(shape=(dim,))

        if use_gauss_outlet is True:
            mdot = 0.022
            outflow_density = pres_bkrnd/temp_bkrnd/r
            sigma = gauss_outlet_sigma
            #radius = 1.45
            radius = gauss_outlet_radius
            factor = 1./(2*3.14159*sigma**2*(
                1 - math.exp(-(radius**2/(2*sigma**2)))))
            if use_axisymmetric:
                vel_outflow[1] = mdot/outflow_density*factor
            else:
                vel_outflow[0] = mdot/outflow_density*factor

        throat_height = 6.3028e-3
        inlet_area_ratio = inlet_height/throat_height
        if use_axisymmetric:
            inlet_area_ratio *= inlet_area_ratio

        inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                          gamma=gamma,
                                          mach_guess=0.00001)
        temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                               T0=total_temp_inflow,
                                               gamma=gamma)

        # MJA
        # this is better than the way Isentropic Inflow does things,
        # i've removed teh repeated computation of the Isentropic Properties
        # since I know the ramp values at the start, I can just hard code
        # them irto the pressure ramp function
        # go back and update the boundary conditions to do the same thing
        #
        # also extend this to be a class so I can have one for each boundary
        inlet_ramp_beginP = getIsentropicPressure(mach=inlet_mach,
                                                  P0=ramp_beginP,
                                                  gamma=gamma)
        inlet_ramp_endP = getIsentropicPressure(mach=inlet_mach,
                                                  P0=ramp_endP,
                                                  gamma=gamma)

        def inlet_ramp_pressure(t):
            return actx.np.where(
                actx.np.greater(t, ramp_time_start),
                actx.np.minimum(
                    inlet_ramp_endP,
                    inlet_ramp_beginP + ((t - ramp_time_start) / ramp_time_interval
                        * (inlet_ramp_endP - inlet_ramp_beginP))),
                inlet_ramp_beginP)

        pres_inflow = inlet_ramp_pressure(current_t)

        # only the eos_type == 0 side of this is being exercised right now
        # we need to think more carefully about what to do when gamma
        # is variable, and how to pass that in
        if eos_type == 0:
            rho_inflow = pres_inflow/temp_inflow/r
            sos = math.sqrt(gamma*pres_inflow/rho_inflow)
            inlet_gamma = gamma
        else:
            rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                              temperature=temp_inflow,
                                              mass_fractions=y)
            inlet_gamma = (
                pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))

            gamma_error = (gamma - inlet_gamma)
            gamma_guess = inlet_gamma
            toler = 1.e-6
            # iterate over the gamma/mach since gamma = gamma(T)
            while gamma_error > toler:

                inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                                  gamma=gamma_guess,
                                                  mach_guess=0.01)
                pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                                    P0=total_pres_inflow,
                                                    gamma=gamma_guess)
                temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                                       T0=total_temp_inflow,
                                                       gamma=gamma_guess)

                rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                                  temperature=temp_inflow,
                                                  mass_fractions=y)
                inlet_gamma = \
                    (pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                     pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))
                gamma_error = (gamma_guess - inlet_gamma)
                gamma_guess = inlet_gamma

            sos = math.sqrt(inlet_gamma*pres_inflow/rho_inflow)

        if dim == 2:
            vel_inflow[1] = inlet_mach*sos
        else:
            vel_inflow[0] = inlet_mach*sos

        if rank == 0:
            print("#### Simluation initialization data: ####")
            print(f"\tinlet Mach number {inlet_mach}")
            print(f"\tinlet gamma {inlet_gamma}")
            print(f"\tinlet temperature {temp_inflow}")
            print(f"\tinlet pressure {pres_inflow}")
            print(f"\tinlet pressure begin {inlet_ramp_beginP}")
            print(f"\tinlet pressure end {inlet_ramp_endP}")
            print(f"\tinlet rho {rho_inflow}")
            print(f"\tinlet velocity x  {vel_inflow[0]}")
            print(f"\tinlet velocity y  {vel_inflow[1]}")
            #print(f"final inlet pressure {pres_inflow_final}")

        from y3prediction.unstart import InitUnstartRamp

        bulk_init = InitUnstartRamp(
            dim=dim,
            nspecies=nspecies,
            disc_sigma=500.,
            disc_loc=shock_loc_x,
            inlet_height=inlet_height,
            pressure_bulk=pres_bkrnd,
            temperature_bulk=temp_bkrnd,
            velocity_bulk=vel_bulk,
            mass_frac_bulk=y,
            pressure_inlet=pres_inflow,
            temperature_inlet=temp_inflow,
            velocity_inlet=vel_inflow,
            mass_frac_inlet=y,
            pressure_outlet=pres_bkrnd,
            temperature_outlet=temp_bkrnd,
            velocity_outlet=vel_outflow,
            mass_frac_outlet=y,
            inlet_pressure_func=inlet_ramp_pressure,
            temp_wall=temp_bkrnd,
            vel_sigma=vel_sigma,
            temp_sigma=temp_sigma,
            gauss_outlet=use_gauss_outlet,
            gauss_outlet_sigma=gauss_outlet_sigma)

    elif init_case == "y3prediction_ramp":

        #
        # isentropic expansion based on the area ratios between the
        # inlet (r=54e-3m) and the throat (r=3.167e-3)
        #
        vel_inflow = np.zeros(shape=(dim,))
        vel_outflow = np.zeros(shape=(dim,))
        vel_injection = np.zeros(shape=(dim,))
        vel_injection_upstream = np.zeros(shape=(dim,))

        throat_height = 3.61909e-3
        inlet_height = 54.129e-3
        inlet_area_ratio = inlet_height/throat_height

        inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                          gamma=gamma,
                                          mach_guess=0.01)
        temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                               T0=total_temp_inflow,
                                               gamma=gamma)

        # MJA
        # this is better than the way Isentropic Inflow does things,
        # i've removed teh repeated computation of the Isentropic Properties
        # since I know the ramp values at the start, I can just hard code
        # them into the pressure ramp function
        # go back and update the boundary conditions to do the same thing
        #
        # also extend this to be a class so I can have one for each boundary
        inlet_ramp_beginP = getIsentropicPressure(mach=inlet_mach,
                                                  P0=ramp_beginP,
                                                  gamma=gamma)
        inlet_ramp_endP = getIsentropicPressure(mach=inlet_mach,
                                                  P0=ramp_endP,
                                                  gamma=gamma)

        def inlet_ramp_pressure(t):
            return actx.np.where(
                actx.np.greater(t, ramp_time_start),
                actx.np.minimum(
                    inlet_ramp_endP,
                    inlet_ramp_beginP + ((t - ramp_time_start) / ramp_time_interval
                        * (inlet_ramp_endP - inlet_ramp_beginP))),
                inlet_ramp_beginP)

        pres_inflow = inlet_ramp_pressure(current_t)

        # only the eos_type == 0 side of this is being exercised right now
        # we need to think more carefully about what to do when gamma
        # is variable, and how to pass that in
        if eos_type == 0:
            rho_inflow = pres_inflow/temp_inflow/r
            sos = math.sqrt(gamma*pres_inflow/rho_inflow)
            inlet_gamma = gamma
        else:
            rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                              temperature=temp_inflow,
                                              mass_fractions=y)
            inlet_gamma = (
                pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))

            gamma_error = (gamma - inlet_gamma)
            gamma_guess = inlet_gamma
            toler = 1.e-6
            # iterate over the gamma/mach since gamma = gamma(T)
            while gamma_error > toler:

                inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                                  gamma=gamma_guess,
                                                  mach_guess=0.01)
                pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                                    P0=total_pres_inflow,
                                                    gamma=gamma_guess)
                temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                                       T0=total_temp_inflow,
                                                       gamma=gamma_guess)

                rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                                  temperature=temp_inflow,
                                                  mass_fractions=y)
                inlet_gamma = \
                    (pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                     pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))
                gamma_error = (gamma_guess - inlet_gamma)
                gamma_guess = inlet_gamma

            sos = math.sqrt(inlet_gamma*pres_inflow/rho_inflow)

        vel_inflow[0] = inlet_mach*sos

        if rank == 0:
            print("#### Simluation initialization data: ####")
            print(f"\tinlet Mach number {inlet_mach}")
            print(f"\tinlet gamma {inlet_gamma}")
            print(f"\tinlet temperature {temp_inflow}")
            print(f"\tinlet pressure {pres_inflow}")
            print(f"\tinlet pressure begin {inlet_ramp_beginP}")
            print(f"\tinlet pressure end {inlet_ramp_endP}")
            print(f"\tinlet rho {rho_inflow}")
            print(f"\tinlet velocity {vel_inflow[0]}")
            #print(f"final inlet pressure {pres_inflow_final}")

        """
        #MJA not yet, need to figure out what to do here
        injection_ramp_beginP = getIsentropicPressure(mach=inlet_mach,
                                                      P0=injection_ramp_beginP,
                                                      gamma=gamma)
        injection_ramp_endP = getIsentropicPressure(mach=inlet_mach,
                                                    P0=injection_ramp_endP,
                                                    gamma=gamma)

        def injection_ramp_pressure(t):
            return actx.np.where(
                actx.np.greater(t, ramp_time_start),
                actx.np.minimum(
                    injection_ramp_endP,
                    injection_ramp_beginP +
                    ((t - injection_ramp_time_start) / injection_ramp_time_interval
                        * (injection_ramp_endP - injection_ramp_beginP))),
                injection_ramp_beginP)

        pres_injection = injection_ramp_pressure(current_t)
        """

        gamma_injection = gamma
        mach_inj = 1.0
        if eos_type == 0:
            gamma_injection = gamma
        else:
            #MJA: Todo, get the gamma from cantera to get the correct
            # inflow properties
            # needs to be iterative with the call below
            gamma_injection = 0.5*(1.24 + 1.4)

        pres_injection = getIsentropicPressure(mach=mach_inj,
                                               P0=total_pres_inj,
                                               gamma=gamma_injection)
        temp_injection = getIsentropicTemperature(mach=mach_inj,
                                                  T0=total_temp_inj,
                                                  gamma=gamma_injection)

        if eos_type == 0:
            rho_injection = pres_injection/temp_injection/r
            sos = math.sqrt(gamma_injection*pres_injection/rho_injection)
        else:
            rho_injection = pyro_mech.get_density(p=pres_injection,
                                                  temperature=temp_injection,
                                                  mass_fractions=y_fuel)
            gamma_guess = \
                (pyro_mech.get_mixture_specific_heat_cp_mass(
                    temp_injection, y_fuel) /
                 pyro_mech.get_mixture_specific_heat_cv_mass(
                    temp_injection, y_fuel))

            gamma_error = np.abs(gamma_guess - gamma_injection)
            toler = 1.e-6
        # iterate over the gamma/mach since gamma = gamma(T)
            while gamma_error > toler:

                pres_injection = getIsentropicPressure(mach=mach_inj,
                                                       P0=total_pres_inj,
                                                       gamma=gamma_guess)
                temp_injection = getIsentropicTemperature(mach=mach_inj,
                                                          T0=total_temp_inj,
                                                          gamma=gamma_guess)
                rho_injection = pyro_mech.get_density(p=pres_injection,
                                                      temperature=temp_injection,
                                                      mass_fractions=y_fuel)
                gamma_injection = \
                    (pyro_mech.get_mixture_specific_heat_cp_mass(
                        temp_injection, y_fuel) /
                     pyro_mech.get_mixture_specific_heat_cv_mass(
                         temp_injection, y_fuel))
                gamma_error = np.abs(gamma_guess - gamma_injection)
                gamma_guess = gamma_injection

            sos = math.sqrt(gamma_injection*pres_injection/rho_injection)

        vel_injection[0] = -mach_inj*sos

        if rank == 0:
            print("\t********")
            print(f"\tinjector Mach number {mach_inj}")
            print(f"\tinjector gamma {gamma_injection}")
            print(f"\tinjector temperature {temp_injection}")
            print(f"\tinjector pressure {pres_injection}")
            print(f"\tinjector rho {rho_injection}")
            print(f"\tinjector velocity {vel_injection[0]}")

        # upstream injection
        gamma_injection_upstream = gamma_injection
        # injection mach number
        pres_injection_upstream = \
            getIsentropicPressure(mach=mach_inj,
                                  P0=total_pres_inj_upstream,
                                  gamma=gamma_injection_upstream)
        temp_injection_upstream = \
            getIsentropicTemperature(mach=mach_inj,
                                     T0=total_temp_inj_upstream,
                                     gamma=gamma_injection_upstream)

        if eos_type == 0:
            rho_injection_upstream = \
                pres_injection_upstream/temp_injection_upstream/r
            sos_upstream = math.sqrt(
                gamma_injection_upstream *
                pres_injection_upstream/rho_injection_upstream)
        else:
            rho_injection_upstream = \
                pyro_mech.get_density(
                    p=pres_injection_upstream,
                    temperature=temp_injection_upstream,
                    mass_fractions=y_fuel)
            gamma_guess = \
                (pyro_mech.get_mixture_specific_heat_cp_mass(
                    temp_injection_upstream, y_fuel) /
                 pyro_mech.get_mixture_specific_heat_cv_mass(
                    temp_injection_upstream, y_fuel))

            gamma_error = np.abs(gamma_guess - gamma_injection_upstream)
            toler = 1.e-6
            # iterate over the gamma/mach since gamma = gamma(T)
            while gamma_error > toler:

                pres_injection_upstream = \
                    getIsentropicPressure(mach=mach_inj,
                                          P0=total_pres_inj_upstream,
                                          gamma=gamma_guess)
                temp_injection_upstream = \
                    getIsentropicTemperature(mach=mach_inj,
                                             T0=total_temp_inj_upstream,
                                             gamma=gamma_guess)
                rho_injection_upstream = \
                    pyro_mech.get_density(
                        p=pres_injection_upstream,
                        temperature=temp_injection_upstream,
                        mass_fractions=y_fuel)
                gamma_injection_upstream = \
                    (pyro_mech.get_mixture_specific_heat_cp_mass(
                        temp_injection_upstream, y_fuel) /
                     pyro_mech.get_mixture_specific_heat_cv_mass(
                        temp_injection_upstream, y_fuel))
                gamma_error = np.abs(gamma_guess -
                                       gamma_injection_upstream)
                gamma_guess = gamma_injection_upstream

            sos_upstream = math.sqrt(
                gamma_injection_upstream*pres_injection_upstream /
                rho_injection_upstream)

        vel_injection_upstream[1] = mach_inj*sos_upstream

        if rank == 0:
            print("\t********")
            print(f"\tUpstream injector Mach number {mach_inj}")
            print("\tUpstream injector gamma "
                  f"{gamma_injection_upstream}")
            print("\tUpstream injector temperature "
                  f"{temp_injection_upstream}")
            print("\tUpstream injector pressure "
                  f"{pres_injection_upstream}")
            print(f"\tUpstream injector rho {rho_injection_upstream}")
            print("\tUpstream injector velocity "
                  f"{vel_injection_upstream[1]}")
            print("#### Simluation initialization data: ####\n")

        from y3prediction.actii_y3_cav8 import InitACTIIRamp
        if actii_init_case == "cav5":
            error_message = "Ramping init not fully implemented for cav5 config"

        bulk_init = InitACTIIRamp(
            dim=dim,
            nspecies=nspecies,
            disc_sigma=500.,
            pressure_bulk=pres_bkrnd,
            temperature_bulk=temp_bkrnd,
            velocity_bulk=vel_outflow,
            mass_frac_bulk=y,
            pressure_inlet=pres_inflow,
            temperature_inlet=temp_inflow,
            velocity_inlet=vel_inflow,
            mass_frac_inlet=y,
            pressure_outlet=pres_bkrnd,
            temperature_outlet=temp_bkrnd,
            velocity_outlet=vel_outflow,
            mass_frac_outlet=y,
            pressure_injection=pres_injection,
            temperature_injection=temp_injection,
            velocity_injection=vel_injection,
            mass_frac_injection=y_fuel,
            pressure_injection_upstream=pres_injection_upstream,
            temperature_injection_upstream=temp_injection_upstream,
            velocity_injection_upstream=vel_injection_upstream,
            mass_frac_injection_upstream=y_fuel,
            inlet_pressure_func=inlet_ramp_pressure,
            temp_wall=temp_bkrnd,
            temp_sigma_injection=temp_sigma_inj,
            vel_sigma_injection=vel_sigma_inj,
            vel_sigma=vel_sigma,
            temp_sigma=temp_sigma)

    elif init_case == "y3prediction":
        #
        # stagnation tempertuare 2076.43 K
        # stagnation pressure 2.745e5 Pa
        #
        # isentropic expansion based on the area ratios between the
        # inlet (r=54e-3m) and the throat (r=3.167e-3)
        #
        vel_inflow = np.zeros(shape=(dim,))
        vel_outflow = np.zeros(shape=(dim,))
        vel_injection = np.zeros(shape=(dim,))
        vel_injection_upstream = np.zeros(shape=(dim,))

        throat_height = 3.61909e-3
        inlet_height = 54.129e-3
        outlet_height = 34.5e-3
        inlet_area_ratio = inlet_height/throat_height
        outlet_area_ratio = outlet_height/throat_height

        inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                          gamma=gamma,
                                          mach_guess=0.01)
        pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                            P0=total_pres_inflow,
                                            gamma=gamma)
        temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                               T0=total_temp_inflow,
                                               gamma=gamma)

        if eos_type == 0:
            rho_inflow = pres_inflow/temp_inflow/r
            sos = math.sqrt(gamma*pres_inflow/rho_inflow)
            inlet_gamma = gamma
        else:
            rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                              temperature=temp_inflow,
                                              mass_fractions=y)
            inlet_gamma = (
                pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))

            gamma_error = (gamma - inlet_gamma)
            gamma_guess = inlet_gamma
            toler = 1.e-6
            # iterate over the gamma/mach since gamma = gamma(T)
            while gamma_error > toler:

                inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                                  gamma=gamma_guess,
                                                  mach_guess=0.01)
                pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                                    P0=total_pres_inflow,
                                                    gamma=gamma_guess)
                temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                                       T0=total_temp_inflow,
                                                       gamma=gamma_guess)

                rho_inflow = pyro_mech.get_density(p=pres_inflow,
                                                  temperature=temp_inflow,
                                                  mass_fractions=y)
                inlet_gamma = \
                    (pyro_mech.get_mixture_specific_heat_cp_mass(temp_inflow, y) /
                     pyro_mech.get_mixture_specific_heat_cv_mass(temp_inflow, y))
                gamma_error = (gamma_guess - inlet_gamma)
                gamma_guess = inlet_gamma

            sos = math.sqrt(inlet_gamma*pres_inflow/rho_inflow)

        vel_inflow[0] = inlet_mach*sos

        if rank == 0:
            print("#### Simluation initialization data: ####")
            print(f"\tinlet Mach number {inlet_mach}")
            print(f"\tinlet gamma {inlet_gamma}")
            print(f"\tinlet temperature {temp_inflow}")
            print(f"\tinlet pressure {pres_inflow}")
            print(f"\tinlet rho {rho_inflow}")
            print(f"\tinlet velocity {vel_inflow[0]}")
            #print(f"final inlet pressure {pres_inflow_final}")

        outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                           gamma=gamma,
                                           mach_guess=1.1)
        pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                             P0=total_pres_inflow,
                                             gamma=gamma)
        temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                                T0=total_temp_inflow,
                                                gamma=gamma)

        if eos_type == 0:
            rho_outflow = pres_outflow/temp_outflow/r
            sos = math.sqrt(gamma*pres_outflow/rho_outflow)
            outlet_gamma = gamma
        else:
            rho_outflow = pyro_mech.get_density(p=pres_outflow,
                                                temperature=temp_outflow,
                                                mass_fractions=y)
            outlet_gamma = \
                (pyro_mech.get_mixture_specific_heat_cp_mass(temp_outflow, y) /
                 pyro_mech.get_mixture_specific_heat_cv_mass(temp_outflow, y))

            gamma_error = (gamma - outlet_gamma)
            gamma_guess = outlet_gamma
            toler = 1.e-6
            # iterate over the gamma/mach since gamma = gamma(T)
            while gamma_error > toler:

                outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                                  gamma=gamma_guess,
                                                  mach_guess=1.1)
                pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                                    P0=total_pres_inflow,
                                                    gamma=gamma_guess)
                temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                                       T0=total_temp_inflow,
                                                       gamma=gamma_guess)
                rho_outflow = pyro_mech.get_density(p=pres_outflow,
                                                    temperature=temp_outflow,
                                                    mass_fractions=y)
                outlet_gamma = \
                    (pyro_mech.get_mixture_specific_heat_cp_mass(temp_outflow, y) /
                     pyro_mech.get_mixture_specific_heat_cv_mass(temp_outflow, y))
                gamma_error = (gamma_guess - outlet_gamma)
                gamma_guess = outlet_gamma

        vel_outflow[0] = outlet_mach*math.sqrt(gamma*pres_outflow/rho_outflow)

        if rank == 0:
            print("\t********")
            print(f"\toutlet Mach number {outlet_mach}")
            print(f"\toutlet gamma {outlet_gamma}")
            print(f"\toutlet temperature {temp_outflow}")
            print(f"\toutlet pressure {pres_outflow}")
            print(f"\toutlet rho {rho_outflow}")
            print(f"\toutlet velocity {vel_outflow[0]}")

        gamma_injection = gamma
        if nspecies > 0:
            # injection mach number
            if eos_type == 0:
                gamma_injection = gamma
            else:
                #MJA: Todo, get the gamma from cantera to get the correct
                # inflow properties
                # needs to be iterative with the call below
                gamma_injection = 0.5*(1.24 + 1.4)

            pres_injection = getIsentropicPressure(mach=mach_inj,
                                                   P0=total_pres_inj,
                                                   gamma=gamma_injection)
            temp_injection = getIsentropicTemperature(mach=mach_inj,
                                                      T0=total_temp_inj,
                                                      gamma=gamma_injection)

            if eos_type == 0:
                rho_injection = pres_injection/temp_injection/r
                sos = math.sqrt(gamma_injection*pres_injection/rho_injection)
            else:
                rho_injection = pyro_mech.get_density(p=pres_injection,
                                                      temperature=temp_injection,
                                                      mass_fractions=y_fuel)
                gamma_guess = \
                    (pyro_mech.get_mixture_specific_heat_cp_mass(
                        temp_injection, y_fuel) /
                     pyro_mech.get_mixture_specific_heat_cv_mass(
                        temp_injection, y_fuel))

                gamma_error = np.abs(gamma_guess - gamma_injection)
                toler = 1.e-6
                # iterate over the gamma/mach since gamma = gamma(T)
                while gamma_error > toler:

                    pres_injection = getIsentropicPressure(mach=mach_inj,
                                                           P0=total_pres_inj,
                                                           gamma=gamma_guess)
                    temp_injection = getIsentropicTemperature(mach=mach_inj,
                                                              T0=total_temp_inj,
                                                              gamma=gamma_guess)
                    rho_injection = pyro_mech.get_density(p=pres_injection,
                                                          temperature=temp_injection,
                                                          mass_fractions=y_fuel)
                    gamma_injection = \
                        (pyro_mech.get_mixture_specific_heat_cp_mass(
                            temp_injection, y_fuel) /
                         pyro_mech.get_mixture_specific_heat_cv_mass(
                             temp_injection, y_fuel))
                    gamma_error = np.abs(gamma_guess - gamma_injection)
                    gamma_guess = gamma_injection

                sos = math.sqrt(gamma_injection*pres_injection/rho_injection)

            vel_injection[0] = -mach_inj*sos

            if rank == 0:
                print("\t********")
                print(f"\tinjector Mach number {mach_inj}")
                print(f"\tinjector gamma {gamma_injection}")
                print(f"\tinjector temperature {temp_injection}")
                print(f"\tinjector pressure {pres_injection}")
                print(f"\tinjector rho {rho_injection}")
                print(f"\tinjector velocity {vel_injection[0]}")

            # upstream injection
            if use_upstream_injection:
                gamma_injection_upstream = gamma_injection
                if nspecies > 0:
                    # injection mach number
                    pres_injection_upstream = \
                        getIsentropicPressure(mach=mach_inj,
                                              P0=total_pres_inj_upstream,
                                              gamma=gamma_injection_upstream)
                    temp_injection_upstream = \
                        getIsentropicTemperature(mach=mach_inj,
                                                 T0=total_temp_inj_upstream,
                                                 gamma=gamma_injection_upstream)

                    if eos_type == 0:
                        rho_injection_upstream = \
                            pres_injection_upstream/temp_injection_upstream/r
                        sos_upstream = math.sqrt(
                            gamma_injection_upstream *
                            pres_injection_upstream/rho_injection_upstream)
                    else:
                        rho_injection_upstream = \
                            pyro_mech.get_density(
                                p=pres_injection_upstream,
                                temperature=temp_injection_upstream,
                                mass_fractions=y_fuel)
                        gamma_guess = \
                            (pyro_mech.get_mixture_specific_heat_cp_mass(
                                temp_injection_upstream, y_fuel) /
                             pyro_mech.get_mixture_specific_heat_cv_mass(
                                temp_injection_upstream, y_fuel))

                        gamma_error = np.abs(gamma_guess - gamma_injection_upstream)
                        toler = 1.e-6
                        # iterate over the gamma/mach since gamma = gamma(T)
                        while gamma_error > toler:

                            pres_injection_upstream = \
                                getIsentropicPressure(mach=mach_inj,
                                                      P0=total_pres_inj_upstream,
                                                      gamma=gamma_guess)
                            temp_injection_upstream = \
                                getIsentropicTemperature(mach=mach_inj,
                                                         T0=total_temp_inj_upstream,
                                                         gamma=gamma_guess)
                            rho_injection_upstream = \
                                pyro_mech.get_density(
                                    p=pres_injection_upstream,
                                    temperature=temp_injection_upstream,
                                    mass_fractions=y_fuel)
                            gamma_injection_upstream = \
                                (pyro_mech.get_mixture_specific_heat_cp_mass(
                                    temp_injection_upstream, y_fuel) /
                                 pyro_mech.get_mixture_specific_heat_cv_mass(
                                    temp_injection_upstream, y_fuel))
                            gamma_error = np.abs(gamma_guess -
                                                   gamma_injection_upstream)
                            gamma_guess = gamma_injection_upstream

                        sos_upstream = math.sqrt(
                            gamma_injection_upstream*pres_injection_upstream /
                            rho_injection_upstream)

                    vel_injection_upstream[1] = mach_inj*sos_upstream

                    if rank == 0:
                        print("\t********")
                        print(f"\tUpstream injector Mach number {mach_inj}")
                        print("\tUpstream injector gamma "
                              f"{gamma_injection_upstream}")
                        print("\tUpstream injector temperature "
                              f"{temp_injection_upstream}")
                        print("\tUpstream injector pressure "
                              f"{pres_injection_upstream}")
                        print(f"\tUpstream injector rho {rho_injection_upstream}")
                        print("\tUpstream injector velocity "
                              f"{vel_injection_upstream[1]}")
                        print("#### Simluation initialization data: ####\n")

        else:
            if rank == 0:
                print("\t********")
                print("\tnspecies=0, injection disabled")

        # read geometry files
        geometry_bottom = None
        geometry_top = None
        if rank == 0:
            from numpy import loadtxt
            geometry_bottom = loadtxt("data/nozzleBottom.dat",
                                      comments="#", unpack=False)
            geometry_top = loadtxt("data/nozzleTop.dat",
                                   comments="#", unpack=False)
        geometry_bottom = comm.bcast(geometry_bottom, root=0)
        geometry_top = comm.bcast(geometry_top, root=0)

        inj_ymin = -0.0243245
        inj_ymax = -0.0227345

        if actii_init_case == "cav8":
            from y3prediction.actii_y3_cav8 import InitACTII
        else:
            from y3prediction.actii_y3_cav5 import InitACTII

        bulk_init = InitACTII(dim=dim,
                              geom_top=geometry_top, geom_bottom=geometry_bottom,
                              P0=total_pres_inflow, T0=total_temp_inflow,
                              temp_wall=temp_wall, temp_sigma=temp_sigma,
                              vel_sigma=vel_sigma, nspecies=nspecies,
                              mass_frac=y, gamma_guess=inlet_gamma,
                              inj_gamma_guess=gamma_injection,
                              inj_pres=total_pres_inj,
                              inj_temp=total_temp_inj,
                              inj_vel=vel_injection,
                              inj_pres_u=total_pres_inj_upstream,
                              inj_temp_u=total_temp_inj_upstream,
                              inj_vel_u=vel_injection_upstream,
                              inj_mass_frac=y_fuel,
                              inj_temp_sigma=temp_sigma_inj,
                              inj_vel_sigma=vel_sigma_inj,
                              inj_ytop=inj_ymax, inj_ybottom=inj_ymin,
                              inj_mach=mach_inj, injection=use_injection)

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:09d}-{rank:04d}.pkl"
    )

    from mirgecom.restart import read_restart_data
    restart_nspecies = 0
    if restart_from_axi:
        axi_restart_data = read_restart_data(actx, axi_filename)
        vol_to_axi_mesh = axi_restart_data["volume_to_local_mesh_data"]
        axi_vol_meshes = {
            vol: mesh for vol, (mesh, _) in vol_to_axi_mesh.items()}
        restart_order = int(axi_restart_data["order"])
        restart_nspecies = axi_restart_data["nspecies"]

    if restart_filename:  # read the grid from restart data
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"
        restart_data = read_restart_data(actx, restart_filename)

        current_step = restart_data["step"]
        first_step = current_step
        current_t = restart_data["t"]
        last_viz_interval = restart_data["last_viz_interval"]
        t_start = current_t
        if use_wall:
            t_wall_start = restart_data["t_wall"]
        volume_to_local_mesh_data = restart_data["volume_to_local_mesh_data"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        restart_nspecies = restart_data["nspecies"]

        restart_nparts = restart_data["num_parts"]
        if restart_nparts != nparts:
            error_message = \
                "Incorrect number or ranks in target: {}".format(restart_nparts)
            raise RuntimeError(error_message)

    else:  # generate the grid from scratch

        # eventually encapsulate these inside a class for the respective inits
        if init_case == "shock1d" or init_case == "flame1d":

            def get_mesh_data():
                if generate_mesh is True:
                    if rank == 0:
                        print("Generating mesh from scratch")
                    from y3prediction.shock1d import get_mesh
                    mesh, tag_to_elements = get_mesh(
                        dim=dim, angle=0.*mesh_angle, size=mesh_size,
                        mesh_origin=mesh_origin,
                        bl_ratio=bl_ratio, interface_ratio=interface_ratio,
                        transfinite=transfinite, use_wall=use_wall,
                        use_quads=use_tpe, use_gmsh=use_gmsh)()
                else:
                    if rank == 0:
                        print("Reading mesh")
                    from meshmode.mesh.io import read_gmsh
                    mesh_construction_kwargs = {
                        "force_positive_orientation":  True,
                        "skip_element_orientation_test":  True}
                    mesh, tag_to_elements = read_gmsh(
                        mesh_filename, force_ambient_dim=dim,
                        mesh_construction_kwargs=mesh_construction_kwargs,
                        return_tag_to_elements_map=True)

                volume_to_tags = {"fluid": ["fluid"]}
                if use_wall:
                    volume_to_tags["wall"] = ["wall_insert"]
                else:
                    from mirgecom.simutil import extract_volumes
                    mesh, tag_to_elements = extract_volumes(
                        mesh, tag_to_elements, volume_to_tags["fluid"],
                        "wall_interface")

                import sys
                import numpy
                numpy.set_printoptions(threshold=sys.maxsize)
                #print(f"{mesh=}")

                # apply periodicity
                if periodic_mesh is True:
                    from meshmode.mesh.processing import (
                        glue_mesh_boundaries, BoundaryPairMapping)

                    from meshmode import AffineMap
                    bdry_pair_mappings_and_tols = []
                    offset = [0., 0.02]
                    bdry_pair_mappings_and_tols.append((
                        BoundaryPairMapping(
                            "periodic_y_bottom",
                            "periodic_y_top",
                            AffineMap(offset=offset)),
                        1e-12))

                    if use_wall:
                        bdry_pair_mappings_and_tols.append((
                            BoundaryPairMapping(
                                "solid_wall_bottom",
                                "solid_wall_top",
                                AffineMap(offset=offset)),
                            1e-12))

                    mesh = glue_mesh_boundaries(mesh, bdry_pair_mappings_and_tols)

                # print(f"{mesh=}")
                from meshmode.mesh.processing import rotate_mesh_around_axis
                if mesh_angle > 0:
                    mesh = rotate_mesh_around_axis(mesh, theta=theta)

                return mesh, tag_to_elements, volume_to_tags
        # eventually encapsulate these inside a class for the respective inits
        elif init_case == "backward_step":
            def get_mesh_data():
                if generate_mesh is True:
                    if rank == 0:
                        print("Generating mesh from scratch")
                    #from y3prediction.backward_step import get_mesh
                    from y3prediction.forward_step import get_mesh
                    mesh, tag_to_elements = get_mesh(
                        dim=dim, size=mesh_size,
                        bl_ratio=bl_ratio, interface_ratio=interface_ratio,
                        transfinite=transfinite, use_wall=use_wall,
                        use_quads=use_tpe, use_gmsh=use_gmsh)()
                else:
                    if rank == 0:
                        print("Reading mesh")
                    from meshmode.mesh.io import read_gmsh
                    mesh_construction_kwargs = {
                        "force_positive_orientation":  True,
                        "skip_element_orientation_test":  True}
                    mesh, tag_to_elements = read_gmsh(
                        mesh_filename, force_ambient_dim=dim,
                        mesh_construction_kwargs=mesh_construction_kwargs,
                        return_tag_to_elements_map=True)

                volume_to_tags = {"fluid": ["fluid"]}
                if use_wall:
                    volume_to_tags["wall"] = ["wall_insert"]
                else:
                    from mirgecom.simutil import extract_volumes
                    mesh, tag_to_elements = extract_volumes(
                        mesh, tag_to_elements, volume_to_tags["fluid"],
                        "wall_interface")

                return mesh, tag_to_elements, volume_to_tags
        elif init_case == "forward_step":
            def get_mesh_data():
                if generate_mesh is True:
                    if rank == 0:
                        print("Generating mesh from scratch")
                    from y3prediction.forward_step import get_mesh
                    mesh, tag_to_elements = get_mesh(
                        dim=dim, size=mesh_size,
                        bl_ratio=bl_ratio, interface_ratio=interface_ratio,
                        transfinite=transfinite, use_wall=use_wall,
                        use_quads=use_tpe, use_gmsh=use_gmsh)()
                else:
                    if rank == 0:
                        print("Reading mesh")
                    from meshmode.mesh.io import read_gmsh
                    mesh_construction_kwargs = {
                        "force_positive_orientation":  True,
                        "skip_element_orientation_test":  True}
                    mesh, tag_to_elements = read_gmsh(
                        mesh_filename, force_ambient_dim=dim,
                        mesh_construction_kwargs=mesh_construction_kwargs,
                        return_tag_to_elements_map=True)

                volume_to_tags = {"fluid": ["fluid"]}
                if use_wall:
                    volume_to_tags["wall"] = ["wall_insert"]
                else:
                    from mirgecom.simutil import extract_volumes
                    mesh, tag_to_elements = extract_volumes(
                        mesh, tag_to_elements, volume_to_tags["fluid"],
                        "wall_interface")

                return mesh, tag_to_elements, volume_to_tags
        elif init_case == "mixing_layer" or init_case == "mixing_layer_hot":
            if rank == 0:
                print("Generating mesh from scratch")

            def get_mesh_data():
                from y3prediction.mixing_layer import get_mesh
                mesh, tag_to_elements = get_mesh(
                    dim=dim, size=mesh_size, layer_ratio=bl_ratio,
                    vorticity_thickness=vorticity_thickness,
                    transfinite=transfinite,
                    use_quads=use_tpe)()
                volume_to_tags = {"fluid": ["fluid"]}
                return mesh, tag_to_elements, volume_to_tags

        elif init_case == "wedge":
            if rank == 0:
                print("Generating mesh from scratch")

            def get_mesh_data():
                from y3prediction.wedge import get_mesh
                mesh, tag_to_elements = get_mesh(
                    dim=dim, size=mesh_size, bl_ratio=bl_ratio,
                    transfinite=transfinite, use_wall=use_wall,
                    use_quads=use_tpe, use_gmsh=use_gmsh)()

                volume_to_tags = {"fluid": ["fluid"]}
                if use_wall:
                    volume_to_tags["wall"] = ["wall_insert"]
                else:
                    from mirgecom.simutil import extract_volumes
                    mesh, tag_to_elements = extract_volumes(
                        mesh, tag_to_elements, volume_to_tags["fluid"],
                        "wall_interface")

                return mesh, tag_to_elements, volume_to_tags
        elif init_case == "species_diffusion":
            if rank == 0:
                print("Generating mesh from scratch")

            def get_mesh_data():
                from y3prediction.species_diffusion import get_mesh
                mesh, tag_to_elements = get_mesh(
                    dim=dim, size=mesh_size,
                    transfinite=transfinite,
                    use_quads=use_tpe)()

                volume_to_tags = {"fluid": ["fluid"]}
                return mesh, tag_to_elements, volume_to_tags
        else:
            if rank == 0:
                print(f"Reading mesh from {mesh_filename}")

            def get_mesh_data():
                from meshmode.mesh.io import read_gmsh
                mesh_construction_kwargs = {
                    "force_positive_orientation":  True,
                    "skip_element_orientation_test":  True}
                mesh, tag_to_elements = read_gmsh(
                    mesh_filename, force_ambient_dim=dim,
                    mesh_construction_kwargs=mesh_construction_kwargs,
                    return_tag_to_elements_map=True)
                volume_to_tags = {
                    "fluid": ["fluid"]}
                if use_wall:
                    volume_to_tags["wall"] = ["wall_insert", "wall_surround"]
                else:
                    from mirgecom.simutil import extract_volumes
                    mesh, tag_to_elements = extract_volumes(
                        mesh, tag_to_elements, volume_to_tags["fluid"],
                        "wall_interface")
                return mesh, tag_to_elements, volume_to_tags

        # use a pre-partitioned mesh
        if os.path.isdir(mesh_filename):
            pkl_filename = (mesh_filename + "/" + mesh_partition_prefix
                            + f"_mesh_np{nparts}_rank{rank}.pkl")
            if rank == 0:
                print("Reading mesh from pkl files in directory"
                      f" {mesh_filename}.")
            if not os.path.exists(pkl_filename):
                raise RuntimeError(f"Mesh pkl file ({pkl_filename})"
                                   " not found.")
            with open(pkl_filename, "rb") as pkl_file:
                global_nelements, volume_to_local_mesh_data = \
                    pickle.load(pkl_file)

        else:
            def my_partitioner(mesh, tag_to_elements, num_ranks):
                from mirgecom.simutil import geometric_mesh_partitioner
                return geometric_mesh_partitioner(
                    mesh, num_ranks, auto_balance=True, imbalance_tolerance=part_tol,
                    debug=False)

            part_func = my_partitioner if use_1d_part else None
            volume_to_local_mesh_data, global_nelements = distribute_mesh(
                comm, get_mesh_data, partition_generator_func=part_func)

    fluid_nelements = volume_to_local_mesh_data["fluid"][0].nelements
    wall_nelements = 0
    if use_wall:
        wall_nelements = volume_to_local_mesh_data["wall"][0].nelements
    local_nelements = fluid_nelements + wall_nelements
    vol_meshes = {vol: mesh for vol, (mesh, _) in volume_to_local_mesh_data.items()}

    # Early grep-ready nelement report
    for rnk in range(nparts):
        if rnk == rank:
            print(f"Rank({rank}) mesh partition")
            print("---------------------------")
            if fluid_nelements > 0:
                print(f"Number of fluid elements: {fluid_nelements}")
            if wall_nelements > 0:
                print(f"Number of wall elements: {wall_nelements}")
            print(f"Number of elements: {local_nelements}")
            print("---------------------------")
        comm.Barrier()

    # target data, used for sponge and prescribed boundary condtitions
    if target_filename:  # read the grid from restart data
        target_filename = f"{target_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        target_data = read_restart_data(actx, target_filename)
        global_nelements = target_data["global_nelements"]
        target_order = int(target_data["order"])

        target_nparts = target_data["num_parts"]
        if target_nparts != nparts:
            error_message = \
                "Incorrect number or ranks in target: {}".format(target_nparts)
            raise RuntimeError(error_message)

        target_nspecies = target_data["nspecies"]
        """
        if target_nspecies != nspecies:
            error_message = \
                "Incorrect number of species in target: {}".format(target_nspecies)
            raise RuntimeError(error_message)
        """

        target_nelements = target_data["global_nelements"]
        if target_nelements != global_nelements:
            error_message = \
                "Incorrect number of elements in target: {}".format(target_nelements)
            raise RuntimeError(error_message)
    else:
        logger.warning("No target file specied, using restart as target")

    disc_msg = f"Making {dim}D order {order} discretization"
    if quadrature_order < 0:
        # TPE does 2*p+1 on the inside
        if use_tpe:
            disc_msg = disc_msg + "Adjusting quadrature order for TPE"
            quadrature_order = order
        else:
            quadrature_order = 2*order + 1

    if use_overintegration:
        disc_msg = disc_msg + f" with quadrature order {quadrature_order}"
    disc_msg = disc_msg + "."
    if rank == 0:
        logger.info(disc_msg)

    dcoll = create_discretization_collection(
        actx, volume_meshes=vol_meshes, order=order,
        quadrature_order=quadrature_order, tensor_product_elements=use_tpe)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    fluid_nodes = force_evaluation(actx, actx.thaw(dcoll.nodes(dd_vol_fluid)))

    def check_boundary(boundary, name):

        #print(f"check_boundary {boundary=} {name=}")
        try:
            force_evaluation(actx, actx.thaw(dcoll.nodes(boundary)))
        except ValueError:
            if rank == 0:
                print(f"Could not find boundary named {name} in fluid domain,",
                       "boundary type will be unused")
            return False

        if rank == 0:
            print(f"Found boundary named {name} in fluid domain")
        return True

    # setup element boundary assignments
    bndry_elements = {}
    for bnd_name in bndry_config:
        # skip disabled boundaries
        if bndry_config[bnd_name] != "none":
            # check to see if any elements are assigned to this named boundary,
            # if not, disabled it
            bndry_elements[bnd_name] = dd_vol_fluid.trace(bnd_name)
            bnd_exists = check_boundary(bndry_elements[bnd_name], bnd_name)
            if not bnd_exists:
                bndry_config[bnd_name] = "none"

    if rank == 0:
        print("### Boundary Condition Summary ###")
        print("The following boundary conditions are enabled:")
        for bnd_name in bndry_config:
            if bndry_config[bnd_name] != "none":
                print(f"\t{bnd_name} = {bndry_config[bnd_name]}")

        print("The following boundary conditions are unused:")
        for bnd_name in bndry_config:
            if bndry_config[bnd_name] == "none":
                print(f"\t{bnd_name}")

        print("### Boundary Condition Summary ###")

    bndry_mapping = {
        "isothermal_noslip": IsothermalWallBoundary(temp_wall),
        "adiabatic_noslip": AdiabaticNoslipWallBoundary(),
        "adiabatic_slip": AdiabaticSlipBoundary(),
        "isothermal_slip": IsothermalSlipWallBoundary(),
        "pressure_outflow": PressureOutflowBoundary(outflow_pressure)
        #"riemann_inflow": RiemannInflowBoundary(cv_inflow, temp_inflow)
    }

    wall_farfield = DirichletDiffusionBoundary(temp_wall)

    def assign_fluid_boundaries(all_boundaries, bndry_mapping):

        for bnd_name in bndry_config:
            bndry_type = bndry_config[bnd_name]
            if bndry_type != "none":
                all_boundaries[bndry_elements[bnd_name].domain_tag] \
                    = bndry_mapping[bndry_type]
        return all_boundaries

    dd_vol_wall = None
    if use_wall:
        dd_vol_wall = DOFDesc(VolumeDomainTag("wall"), DISCR_TAG_BASE)
        wall_nodes = force_evaluation(actx, actx.thaw(dcoll.nodes(dd_vol_wall)))

        wall_vol_discr = dcoll.discr_from_dd(dd_vol_wall)
        wall_tag_to_elements = volume_to_local_mesh_data["wall"][1]

        try:
            wall_insert_mask = mask_from_elements(
                wall_vol_discr, actx, wall_tag_to_elements["wall_insert"])
        except KeyError:
            wall_insert_mask = 0
            if rank == 0:
                print("No elements matching wall_insert")
                #wall_insert_mask = actx.np.zeros_like(wall_tag_to_elements)

        try:
            wall_surround_mask = mask_from_elements(
                wall_vol_discr, actx, wall_tag_to_elements["wall_surround"])
        except KeyError:
            wall_surround_mask = 0
            if rank == 0:
                print("No elements matching wall_surround")
                #wall_surround_mask = actx.np.zeros_like(wall_tag_to_elements)

        wall_ffld_bnd = dd_vol_wall.trace("wall_farfield")

    from grudge.dt_utils import characteristic_lengthscales
    char_length_fluid = force_evaluation(actx,
        characteristic_lengthscales(actx, dcoll, dd=dd_vol_fluid))

    # put the lengths on the nodes vs elements
    xpos_fluid = fluid_nodes[0]
    char_length_fluid = char_length_fluid + actx.np.zeros_like(xpos_fluid)

    smoothness_diffusivity = \
        smooth_char_length_alpha*char_length_fluid**2/current_dt

    if use_wall:
        xpos_wall = wall_nodes[0]
        char_length_wall = force_evaluation(actx,
            characteristic_lengthscales(actx, dcoll, dd=dd_vol_wall))
        xpos_wall = wall_nodes[0]
        char_length_wall = char_length_wall + actx.np.zeros_like(xpos_wall)
        """
        smoothness_diffusivity_wall = \
            smooth_char_length_alpha*char_length_wall**2/current_dt
        """

    def compute_smoothed_char_length(href_fluid, comm_ind):
        # regular boundaries

        smooth_neumann = NeumannDiffusionBoundary(0)
        fluid_smoothness_boundaries = {}
        for bnd_name in bndry_config:
            if bndry_config[bnd_name] != "none":
                fluid_smoothness_boundaries[bndry_elements[bnd_name]] =\
                    smooth_neumann

        """
        fluid_smoothness_boundaries = assign_fluid_boundaries(
            outflow=smooth_neumann,
            inflow=smooth_neumann,
            injection=smooth_neumann,
            flow=smooth_neumann,
            wall=smooth_neumann,
            interface=smooth_neumann)
        """

        if use_wall:
            fluid_smoothness_boundaries.update({
                 dd_bdry.domain_tag: NeumannDiffusionBoundary(0)
                 for dd_bdry in filter_part_boundaries(
                     dcoll, volume_dd=dd_vol_fluid, neighbor_volume_dd=dd_vol_wall)})

        smooth_href_fluid_rhs = diffusion_operator(
            dcoll, smoothness_diffusivity, fluid_smoothness_boundaries,
            href_fluid,
            quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
            comm_tag=(_SmoothCharDiffFluidCommTag, comm_ind))*current_dt

        return smooth_href_fluid_rhs

    compute_smoothed_char_length_compiled = \
        actx.compile(compute_smoothed_char_length)

    """
    def compute_smoothed_char_length_wall(href_wall, comm_ind):
        smooth_neumann = NeumannDiffusionBoundary(0)
        wall_smoothness_boundaries = {
            wall_ffld_bnd.domain_tag: smooth_neumann,
        }

        wall_smoothness_boundaries.update({
             dd_bdry.domain_tag: NeumannDiffusionBoundary(0)
             for dd_bdry in filter_part_boundaries(
                 dcoll, volume_dd=dd_vol_wall, neighbor_volume_dd=dd_vol_fluid)})

        smooth_href_wall_rhs = diffusion_operator(
                dcoll, smoothness_diffusivity_wall, wall_smoothness_boundaries,
                href_wall,
                quadrature_tag=quadrature_tag, dd=dd_vol_wall,
                comm_tag=(_SmoothCharDiffWallCommTag, comm_ind))*current_dt

        return smooth_href_wall_rhs

    if use_wall:
        compute_smoothed_char_length_wall_compiled = \
            actx.compile(compute_smoothed_char_length_wall)
    """

    smoothed_char_length_fluid = char_length_fluid

    if use_smoothed_char_length:
        for i in range(smooth_char_length):
            smoothed_char_length_fluid_rhs = \
                compute_smoothed_char_length_compiled(smoothed_char_length_fluid, i)
            smoothed_char_length_fluid = smoothed_char_length_fluid + \
                                         smoothed_char_length_fluid_rhs

        """
        if use_wall:
            smoothed_char_length_wall = char_length_wall
            for i in range(smooth_char_length):
                smoothed_char_length_wall_rhs = \
                    compute_smoothed_char_length_wall_compiled(
                        smoothed_char_length_wall, i)
                smoothed_char_length_wall = smoothed_char_length_wall + \
                                            smoothed_char_length_wall_rhs
        """

        smoothed_char_length_fluid = force_evaluation(actx,
                                                      smoothed_char_length_fluid)

        """
        if use_wall:
            smoothed_char_length_wall = force_evaluation(actx,
                                                         smoothed_char_length_wall)
        """

    if rank == 0:
        logger.info("Before restart/init")

    #########################
    # Convenience Functions #
    #########################
    def drop_order(dcoll, field, theta, dd=dd_vol_fluid,
                   positivity_preserving=False):
        # Compute cell averages of the state
        def cancel_polynomials(grp):
            return actx.from_numpy(
                np.asarray([1 if sum(mode_id) == 0
                            else 0 for mode_id in grp.mode_ids()]))

        dd_nodal = dd
        dd_modal = dd_nodal.with_discr_tag(DISCR_TAG_MODAL)

        modal_map = dcoll.connection_from_dds(dd_nodal, dd_modal)
        nodal_map = dcoll.connection_from_dds(dd_modal, dd_nodal)

        modal_discr = dcoll.discr_from_dd(dd_modal)
        modal_field = modal_map(field)

        # cancel the ``high-order"" polynomials p > 0 and keep the average
        filtered_modal_field = DOFArray(
            actx,
            tuple(actx.einsum("ej,j->ej",
                              vec_i,
                              cancel_polynomials(grp),
                              arg_names=("vec", "filter"),
                              tagged=(FirstAxisIsElementsTag(),))
                  for grp, vec_i in zip(modal_discr.groups, modal_field))
        )

        # convert back to nodal to have the average at all points
        cell_avgs = nodal_map(filtered_modal_field)

        if positivity_preserving:
            cell_avgs = actx.np.where(actx.np.greater(cell_avgs, 1e-5),
                                                      cell_avgs, 1e-5)

        return theta*(field - cell_avgs) + cell_avgs

    def _drop_order_cv(cv, flipped_smoothness, theta_factor, dd=None):

        smoothness = 1.0 - theta_factor*flipped_smoothness

        density_lim = drop_order(dcoll, cv.mass, smoothness)
        momentum_lim = make_obj_array([
            drop_order(dcoll, cv.momentum[0], smoothness),
            drop_order(dcoll, cv.momentum[1], smoothness)])
        energy_lim = drop_order(dcoll, cv.energy, smoothness)

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=density_lim, energy=energy_lim,
                              momentum=momentum_lim, species_mass=cv.species_mass)

    drop_order_cv = actx.compile(_drop_order_cv)

    def element_minimum(dcoll, field, dd=dd_vol_fluid,
                        positivity_preserving=False):

        # convert back to nodal to have the average at all points
        cell_min = op.elementwise_min(dcoll, dd, field)

        return cell_min

    def element_maximum(dcoll, field, dd=dd_vol_fluid,
                        positivity_preserving=False):

        # convert back to nodal to have the average at all points
        cell_max = op.elementwise_max(dcoll, dd, field)

        return cell_max

    def _neighbor_maximum(field):

        from grudge.trace_pair import interior_trace_pairs
        itp = interior_trace_pairs(dcoll, field, volume_dd=dd_vol_fluid,
                                   comm_tag=_FluidAvgCVTag)

        dd_allfaces = dd_vol_fluid.trace(FACE_RESTR_ALL)
        is_int_face = dcoll.zeros(actx, dd=dd_allfaces, dtype=int)

        for tpair in itp:
            is_int_face = is_int_face + op.project(
                dcoll, tpair.dd, dd_allfaces, actx.np.zeros_like(tpair.ext) + 1)

        face_data = actx.np.where(is_int_face, 0., -np.inf)

        for tpair in itp:
            face_data = face_data + op.project(
                dcoll, tpair.dd, dd_allfaces, tpair.ext)

            # Make sure MPI communication happens, ugh
            face_data = face_data + (
                0*op.project(dcoll, tpair.dd, dd_allfaces, tpair.int))

        def function(face_data):
            # Reshape from (nelements*nfaces, 1) to (nfaces, nelements, 1)
            # to get per-element data
            node_data_per_group = []
            for igrp, group in enumerate(
                    dcoll.discr_from_dd(dd_vol_fluid).mesh.groups):
                nelements = group.nelements
                nfaces = group.nfaces
                el_face_data = face_data[igrp].reshape(nfaces, nelements,
                                                       face_data[igrp].shape[1])
                if actx.supports_nonscalar_broadcasting:
                    el_data = actx.np.max(el_face_data, axis=0)[:, 0:1]
                    node_data = actx.np.broadcast_to(
                        el_data, dcoll.zeros(actx, dd=dd_vol_fluid)[igrp].shape)
                else:
                    el_data_np = np.max(actx.to_numpy(el_face_data), axis=0)[:, 0:1]
                    node_data_np = np.ascontiguousarray(np.broadcast_to(el_data_np,
                        dcoll.zeros(actx, dd=dd_vol_fluid)[igrp].shape))
                    node_data = actx.from_numpy(node_data_np)

                node_data_per_group.append(node_data)
            return DOFArray(actx, node_data_per_group)

        from arraycontext import rec_map_array_container
        el_data = rec_map_array_container(function, face_data, leaf_class=DOFArray)

        return el_data

    def _neighbor_minimum(field):

        from grudge.trace_pair import interior_trace_pairs
        itp = interior_trace_pairs(dcoll, field, volume_dd=dd_vol_fluid,
                                          comm_tag=_FluidAvgCVTag)

        dd_allfaces = dd_vol_fluid.trace(FACE_RESTR_ALL)
        is_int_face = dcoll.zeros(actx, dd=dd_allfaces, dtype=int)

        for tpair in itp:
            is_int_face = is_int_face + op.project(
                dcoll, tpair.dd, dd_allfaces, actx.np.zeros_like(tpair.ext) + 1)

        face_data = actx.np.where(is_int_face, 0., np.inf)

        for tpair in itp:
            face_data = face_data + op.project(
                dcoll, tpair.dd, dd_allfaces, tpair.ext)

            # Make sure MPI communication happens, ugh
            face_data = face_data + (
                0*op.project(dcoll, tpair.dd, dd_allfaces, tpair.int))

        def function(face_data):
            # Reshape from (nelements*nfaces, 1) to (nfaces, nelements, 1)
            # to get per-element data
            node_data_per_group = []
            for igrp, group in enumerate(
                    dcoll.discr_from_dd(dd_vol_fluid).mesh.groups):
                nelements = group.nelements
                nfaces = group.nfaces
                el_face_data = face_data[igrp].reshape(nfaces, nelements,
                                                       face_data[igrp].shape[1])
                if actx.supports_nonscalar_broadcasting:
                    el_data = actx.np.min(el_face_data, axis=0)[:, 0:1]
                    node_data = actx.np.broadcast_to(
                        el_data, dcoll.zeros(actx, dd=dd_vol_fluid)[igrp].shape)
                else:
                    el_data_np = np.min(actx.to_numpy(el_face_data), axis=0)[:, 0:1]
                    node_data_np = np.ascontiguousarray(np.broadcast_to(el_data_np,
                        dcoll.zeros(actx, dd=dd_vol_fluid)[igrp].shape))
                    node_data = actx.from_numpy(node_data_np)

                node_data_per_group.append(node_data)
            return DOFArray(actx, node_data_per_group)

        from arraycontext import rec_map_array_container
        el_data = rec_map_array_container(function, face_data, leaf_class=DOFArray)

        return el_data

    def _neighbor_maximum_cv(cv):

        from grudge.trace_pair import interior_trace_pairs
        itp = interior_trace_pairs(dcoll, cv, volume_dd=dd_vol_fluid,
                                   comm_tag=_FluidAvgCVTag)

        dd_allfaces = dd_vol_fluid.trace(FACE_RESTR_ALL)
        is_int_face = dcoll.zeros(actx, dd=dd_allfaces, dtype=int)

        for tpair in itp:
            is_int_face = is_int_face + op.project(
                dcoll, tpair.dd, dd_allfaces, actx.np.zeros_like(tpair.ext) + 1)

        face_data = actx.np.where(is_int_face, 0., -np.inf)

        for tpair in itp:
            face_data = face_data + op.project(
                dcoll, tpair.dd, dd_allfaces, tpair.ext)

            # Make sure MPI communication happens, ugh
            face_data = face_data + (
                0*op.project(dcoll, tpair.dd, dd_allfaces, tpair.int))

        def function(face_data):
            # Reshape from (nelements*nfaces, 1) to (nfaces, nelements, 1)
            # to get per-element data
            node_data_per_group = []
            for igrp, group in enumerate(
                    dcoll.discr_from_dd(dd_vol_fluid).mesh.groups):
                nelements = group.nelements
                nfaces = group.nfaces
                el_face_data = face_data[igrp].reshape(nfaces, nelements,
                                                       face_data[igrp].shape[1])
                if actx.supports_nonscalar_broadcasting:
                    el_data = actx.np.max(el_face_data, axis=0)[:, 0:1]
                    node_data = actx.np.broadcast_to(
                        el_data, dcoll.zeros(actx, dd=dd_vol_fluid)[igrp].shape)
                else:
                    el_data_np = np.max(actx.to_numpy(el_face_data), axis=0)[:, 0:1]
                    node_data_np = np.ascontiguousarray(np.broadcast_to(el_data_np,
                        dcoll.zeros(actx, dd=dd_vol_fluid)[igrp].shape))
                    node_data = actx.from_numpy(node_data_np)

                node_data_per_group.append(node_data)
            return DOFArray(actx, node_data_per_group)

        from arraycontext import rec_map_array_container
        el_data = rec_map_array_container(function, face_data, leaf_class=DOFArray)

        return el_data

    def _neighbor_minimum_cv(cv):

        from grudge.trace_pair import interior_trace_pairs
        itp = interior_trace_pairs(dcoll, cv, volume_dd=dd_vol_fluid,
                                          comm_tag=_FluidAvgCVTag)

        dd_allfaces = dd_vol_fluid.trace(FACE_RESTR_ALL)
        is_int_face = dcoll.zeros(actx, dd=dd_allfaces, dtype=int)

        for tpair in itp:
            is_int_face = is_int_face + op.project(
                dcoll, tpair.dd, dd_allfaces, actx.np.zeros_like(tpair.ext) + 1)

        face_data = actx.np.where(is_int_face, 0., np.inf)

        for tpair in itp:
            face_data = face_data + op.project(
                dcoll, tpair.dd, dd_allfaces, tpair.ext)

            # Make sure MPI communication happens, ugh
            face_data = face_data + (
                0*op.project(dcoll, tpair.dd, dd_allfaces, tpair.int))

        def function(face_data):
            # Reshape from (nelements*nfaces, 1) to (nfaces, nelements, 1)
            # to get per-element data
            node_data_per_group = []
            for igrp, group in enumerate(
                    dcoll.discr_from_dd(dd_vol_fluid).mesh.groups):
                nelements = group.nelements
                nfaces = group.nfaces
                el_face_data = face_data[igrp].reshape(nfaces, nelements,
                                                       face_data[igrp].shape[1])
                if actx.supports_nonscalar_broadcasting:
                    el_data = actx.np.min(el_face_data, axis=0)[:, 0:1]
                    node_data = actx.np.broadcast_to(
                        el_data, dcoll.zeros(actx, dd=dd_vol_fluid)[igrp].shape)
                else:
                    el_data_np = np.min(actx.to_numpy(el_face_data), axis=0)[:, 0:1]
                    node_data_np = np.ascontiguousarray(np.broadcast_to(el_data_np,
                        dcoll.zeros(actx, dd=dd_vol_fluid)[igrp].shape))
                    node_data = actx.from_numpy(node_data_np)

                node_data_per_group.append(node_data)
            return DOFArray(actx, node_data_per_group)

        from arraycontext import rec_map_array_container
        el_data = rec_map_array_container(function, face_data, leaf_class=DOFArray)

        return el_data

    if soln_filter_cutoff < 0:
        soln_filter_cutoff = int(soln_filter_frac * order)
    if rhs_filter_cutoff < 0:
        rhs_filter_cutoff = int(rhs_filter_frac * order)

    if soln_filter_cutoff >= order and soln_nfilter > 0:
        raise ValueError("Invalid setting for solution filter (cutoff >= order).")
    if rhs_filter_cutoff >= order and use_rhs_filter:
        raise ValueError("Invalid setting for RHS filter (cutoff >= order).")

    from mirgecom.filter import (
        exponential_mode_response_function as xmrfunc,
        filter_modally
    )
    soln_frfunc = partial(xmrfunc, alpha=soln_filter_alpha,
                          filter_order=soln_filter_order)
    rhs_frfunc = partial(xmrfunc, alpha=rhs_filter_alpha,
                         filter_order=rhs_filter_order)

    def filter_cv(cv, filter_dd=dd_vol_fluid):
        return filter_modally(dcoll, soln_filter_cutoff, soln_frfunc, cv,
                              dd=filter_dd)

    def filter_fluid_rhs(rhs):
        return filter_modally(dcoll, rhs_filter_cutoff, rhs_frfunc, rhs,
                              dd=dd_vol_fluid)

    def filter_wall_rhs(rhs):
        return filter_modally(dcoll, rhs_filter_cutoff, rhs_frfunc, rhs,
                              dd=dd_vol_wall)

    filter_cv_compiled = actx.compile(filter_cv)
    filter_rhs_fluid_compiled = actx.compile(filter_fluid_rhs)
    filter_rhs_wall_compiled = actx.compile(filter_wall_rhs)

    if soln_nfilter >= 0 and rank == 0:
        logger.info("Solution filtering settings:")
        logger.info(f" - filter every {soln_nfilter} steps")
        logger.info(f" - filter alpha  = {soln_filter_alpha}")
        logger.info(f" - filter cutoff = {soln_filter_cutoff}")
        logger.info(f" - filter order  = {soln_filter_order}")

    if use_rhs_filter and rank == 0:
        logger.info("RHS filtering settings:")
        logger.info(f" - filter alpha  = {rhs_filter_alpha}")
        logger.info(f" - filter cutoff = {rhs_filter_cutoff}")
        logger.info(f" - filter order  = {rhs_filter_order}")

    if use_species_limiter == 1:
        logger.info("Limiting species mass fractions:")
    elif use_species_limiter == 2:
        logger.info("Positivity-preserving limiter enabled:")

    def my_limiter_func(cv, temperature_seed, entropy_min, gas_model, dd):
        limiter_func = None
        if use_species_limiter == 1:
            limiter_func = limit_fluid_state(
                dcoll, cv, temperature_seed, gas_model, dd)
        elif use_species_limiter == 2:
            limiter_func = limit_fluid_state_lv(
                dcoll, cv, temperature_seed, entropy_min, gas_model,
                dd)
        return limiter_func

    limiter_func = None
    if use_species_limiter > 0:
        limiter_func = my_limiter_func

    ########################################
    # Helper functions for building states #
    ########################################

    def _create_fluid_state(cv, temperature_seed, smoothness_mu,
                            smoothness_beta, smoothness_kappa, smoothness_d,
                            entropy_min):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temperature_seed,
                                smoothness_mu=smoothness_mu,
                                smoothness_beta=smoothness_beta,
                                smoothness_kappa=smoothness_kappa,
                                smoothness_d=smoothness_d,
                                entropy_min=entropy_min,
                                limiter_func=limiter_func,
                                limiter_dd=dd_vol_fluid)

    create_fluid_state = actx.compile(_create_fluid_state)

    def update_dv(cv, temperature, smoothness_mu, smoothness_beta,
                  smoothness_kappa, smoothness_d):
        if eos_type == 0:
            return GasDependentVars(
                temperature=temperature,
                pressure=eos.pressure(cv, temperature),
                speed_of_sound=eos.sound_speed(cv, temperature),
                smoothness_mu=smoothness_mu,
                smoothness_beta=smoothness_beta,
                smoothness_kappa=smoothness_kappa,
                smoothness_d=smoothness_d)
        else:
            return MixtureDependentVars(
                temperature=temperature,
                pressure=eos.pressure(cv, temperature),
                speed_of_sound=eos.sound_speed(cv, temperature),
                species_enthalpies=eos.species_enthalpies(cv, temperature),
                smoothness_mu=smoothness_mu,
                smoothness_beta=smoothness_beta,
                smoothness_kappa=smoothness_kappa,
                smoothness_d=smoothness_d)

    def update_tv(cv, dv):
        return gas_model.transport.transport_vars(cv, dv, eos)

    def update_fluid_state(cv, dv, tv):
        from mirgecom.gas_model import ViscousFluidState
        return ViscousFluidState(cv, dv, tv)

    def get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = gas_model.eos.internal_energy(cv)/cv.mass
        return actx.np.abs(
            pyro_mech.get_temperature_update_energy(e, temperature, y))

    get_temperature_update_compiled = actx.compile(get_temperature_update)

    #
    # Setup the wall model
    #
    if use_wall:
        def experimental_kappa(temperature):
            return (
                1.766e-10 * temperature**3
                - 4.828e-7 * temperature**2
                + 6.252e-4 * temperature
                + 6.707e-3)

        def puma_kappa(mass_loss_frac):
            return (
                0.0988 * mass_loss_frac**2
                - 0.2751 * mass_loss_frac
                + 0.201)

        def puma_effective_surface_area(mass_loss_frac):
            # Original fit function: -1.1012e5*x**2 - 0.0646e5*x + 1.1794e5
            # Rescale by x==0 value and rearrange
            return 1.1794e5 * (
                1
                - 0.0547736137 * mass_loss_frac
                - 0.9336950992 * mass_loss_frac**2)

        def _get_wall_kappa_fiber(mass, temperature):
            mass_loss_frac = (
                (wall_insert_rho - mass)/wall_insert_rho
                * wall_insert_mask)
            scaled_insert_kappa = (
                experimental_kappa(temperature)
                * puma_kappa(mass_loss_frac)
                / puma_kappa(0))
            return (
                scaled_insert_kappa * wall_insert_mask
                + wall_surround_kappa * wall_surround_mask)

        def _get_wall_kappa_inert(mass, temperature):
            return (
                wall_insert_kappa * wall_insert_mask
                + wall_surround_kappa * wall_surround_mask)

        def _get_wall_effective_surface_area_fiber(mass):
            mass_loss_frac = (
                (wall_insert_rho - mass)/wall_insert_rho
                * wall_insert_mask)
            return (
                puma_effective_surface_area(mass_loss_frac) * wall_insert_mask)

        def _mass_loss_rate_fiber(mass, ox_mass, temperature, eff_surf_area):
            actx = mass.array_context
            alpha = (
                (0.00143+0.01*actx.np.exp(-1450.0/temperature))
                / (1.0+0.0002*actx.np.exp(13000.0/temperature)))
            k = alpha*actx.np.sqrt(
                (univ_gas_const*temperature)/(2.0*np.pi*mw_o2))
            return (mw_co/mw_o2 + mw_o/mw_o2 - 1)*ox_mass*k*eff_surf_area

        # inert
        wall_model = WallModel(
            heat_capacity=(
                wall_insert_cp * wall_insert_mask
                + wall_surround_cp * wall_surround_mask),
            thermal_conductivity_func=_get_wall_kappa_inert)

        # non-porous
        if wall_material == 1:
            wall_model = WallModel(
                heat_capacity=(
                    wall_insert_cp * wall_insert_mask
                    + wall_surround_cp * wall_surround_mask),
                thermal_conductivity_func=_get_wall_kappa_fiber,
                effective_surface_area_func=_get_wall_effective_surface_area_fiber,
                mass_loss_func=_mass_loss_rate_fiber,
                oxygen_diffusivity=wall_insert_ox_diff * wall_insert_mask)
        # porous
        elif wall_material == 2:
            wall_model = WallModel(
                heat_capacity=(
                    wall_insert_cp * wall_insert_mask
                    + wall_surround_cp * wall_surround_mask),
                thermal_conductivity_func=_get_wall_kappa_fiber,
                effective_surface_area_func=_get_wall_effective_surface_area_fiber,
                mass_loss_func=_mass_loss_rate_fiber,
                oxygen_diffusivity=wall_insert_ox_diff * wall_insert_mask)

        def _create_wall_dependent_vars(wv):
            return wall_model.dependent_vars(wv)

        create_wall_dependent_vars_compiled = actx.compile(
            _create_wall_dependent_vars)

    if rank == 0:
        logger.info("Smoothness functions processing")

    # smoothness used with av = 1
    def compute_smoothness(cv, dv, grad_cv):

        div_v = np.trace(velocity_gradient(cv, grad_cv))

        gamma = gas_model.eos.gamma(cv=cv, temperature=dv.temperature)
        r = gas_model.eos.gas_const(cv)
        c_star = actx.np.sqrt(gamma*r*(2/(gamma+1)*static_temp))
        href = smoothed_char_length_fluid
        indicator = -gamma_sc*href*div_v/c_star

        smoothness = actx.np.log(
            1 + actx.np.exp(theta_sc*(indicator - beta_sc)))/theta_sc
        return smoothness*gamma_sc*href

    def lmax(s):
        b = 1000
        return (s/np.pi*actx.np.arctan(b*s) +
                0.5*s - 1/np.pi*actx.np.arctan(b) + 0.5)

    def lmin(s):
        return s - lmax(s)

    # smoothness used for beta with av = 2
    def compute_smoothness_mbk(cv, dv, grad_cv, grad_t):

        from mirgecom.fluid import velocity_gradient
        vel_grad = velocity_gradient(cv, grad_cv)

        """
        # find the average gradient in each cell
        element_vols = abs(op.elementwise_integral(
            dcoll, dd_vol_fluid, actx.np.zeros_like(cv.mass) + 1.0))
        vel_grad_avg = element_average(dcoll, dd_vol_fluid, vel_grad,
                                   volumes=element_vols)
        ones = 1. + actx.np.zeros_like(cv.mass)
        vel_grad = ones*vel_grad_avg
        """

        div_v = np.trace(vel_grad)
        # use a constant, the smallest (strongest) value in each
        div_v = op.elementwise_min(dcoll, dd_vol_fluid, div_v)

        gamma = gas_model.eos.gamma(cv=cv, temperature=dv.temperature)
        # somehow this can be negative ... which is bad
        r = actx.np.abs(gas_model.eos.gas_const(cv))
        c_star = actx.np.sqrt(gamma*r*(2/(gamma+1)*static_temp))
        #c_star = 460
        href = smoothed_char_length_fluid
        indicator = -href*div_v/c_star

        # limit the indicator range
        # multiply by href, since we won't have access to it inside transport
        indicator_max = 2/actx.np.sqrt(gamma - 1)
        #indicator_max = 3.16
        smoothness_beta = (lmin(lmax(indicator - av2_beta_s0) - indicator_max)
                           + indicator_max)*href

        #smoothness_beta = (smoothness_beta * cv.mass
                           #* actx.np.sqrt(np.dot(cv.velocity, cv.velocity)
                           #+ dv.speed_of_sound**2))

        grad_t_mag = actx.np.sqrt(np.dot(grad_t, grad_t))
        grad_t_mag = op.elementwise_max(dcoll, dd_vol_fluid, grad_t_mag)
        indicator = href*grad_t_mag/static_temp

        # limit the indicator range
        # multiply by href, since we won't have access to it inside transport
        #indicator_min = 1.0
        #indicator_min = 0.01
        #indicator_min = 0.000001
        indicator_max = 2
        smoothness_kappa = (lmin(lmax(indicator - av2_kappa_s0) - indicator_max)
                            + indicator_max)*href

        vmax = actx.np.sqrt(np.dot(cv.velocity, cv.velocity) +
                            2*c_star/(gamma - 1))

        # just the determinant
        # scaled_grad = vel_grad/vmax
        #indicator = href*actx.np.abs(scaled_grad[0][1]*scaled_grad[1][0])

        # Frobenius norm
        if dim == 2:
            indicator = href*actx.np.sqrt(vel_grad[0][1]*vel_grad[0][1] +
                                          vel_grad[1][0]*vel_grad[1][0])/vmax
        else:
            indicator = href*actx.np.sqrt(vel_grad[0][1]*vel_grad[0][1] +
                                          vel_grad[0][2]*vel_grad[0][2] +
                                          vel_grad[1][0]*vel_grad[1][0] +
                                          vel_grad[1][2]*vel_grad[1][2] +
                                          vel_grad[2][0]*vel_grad[2][0] +
                                          vel_grad[2][1]*vel_grad[2][1])/vmax

        indicator = op.elementwise_max(dcoll, dd_vol_fluid, indicator)

        # limit the indicator range
        # multiply by href, since we won't have access to it inside transport
        #indicator_min = 1.0
        indicator_max = 2
        smoothness_mu = (lmin(lmax(indicator - av2_mu_s0) - indicator_max)
                         + indicator_max)*href

        return make_obj_array([smoothness_mu, smoothness_beta, smoothness_kappa])

    # smoothness used for beta with av = 3
    def compute_smoothness_mbkd(cv, dv, grad_cv, grad_t):

        from mirgecom.fluid import species_mass_fraction_gradient
        y_grad = species_mass_fraction_gradient(cv, grad_cv)

        y_grad_max = y_grad[0]
        """
        y_grad_max = actx.np.max(y_grad)
        for i in range(1, nspecies):
            y_grad_max = actx.np.max(y_grad_max, y_grad[i])
        """
        grad_y_mag = actx.np.sqrt(np.dot(y_grad_max, y_grad_max))

        href = smoothed_char_length_fluid
        indicator = href*grad_y_mag

        # limit the indicator range
        indicator_max = 1.0
        smoothness_d = (lmin(lmax(indicator - av2_d_s0) - indicator_max)
                           + indicator_max)*href

        smoothness_mu, smoothness_beta, smoothness_kappa = \
            compute_smoothness_mbk(cv, dv, grad_cv, grad_t)

        return make_obj_array([smoothness_mu, smoothness_beta,
                               smoothness_kappa, smoothness_d])

    def update_smoothness(state, time):
        cv = state.cv
        tseed = state.tseed
        av_smu = state.av_smu
        av_sbeta = state.av_sbeta
        av_skappa = state.av_skappa
        av_sd = state.av_sd
        smin = state.smin

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed,
                                       smoothness_mu=av_smu,
                                       smoothness_beta=av_sbeta,
                                       smoothness_kappa=av_skappa,
                                       smoothness_d=av_sd,
                                       entropy_min=smin,
                                       limiter_func=limiter_func,
                                       limiter_dd=dd_vol_fluid)
        cv = fluid_state.cv  # reset cv to the limited version
        dv = fluid_state.dv

        wv = None
        if use_wall:
            wv = state.wv
            wdv = wall_model.dependent_vars(wv)

            grad_fluid_cv, grad_fluid_t, grad_wall_t = coupled_grad_operator(
                dcoll,
                gas_model,
                dd_vol_fluid, dd_vol_wall,
                uncoupled_fluid_boundaries,
                uncoupled_wall_boundaries,
                fluid_state, wdv.thermal_conductivity, wdv.temperature,
                time=time,
                interface_noslip=noslip,
                quadrature_tag=quadrature_tag,
                limiter_func=limiter_func,
                entropy_min=smin,
                comm_tag=_InitCommTag)
        else:
            fluid_operator_states_quad = make_operator_fluid_states(
                dcoll, fluid_state, gas_model, uncoupled_fluid_boundaries,
                quadrature_tag, dd=dd_vol_fluid, limiter_func=limiter_func,
                entropy_min=smin)

            grad_fluid_cv = grad_cv_operator(
                dcoll=dcoll, gas_model=gas_model, dd=dd_vol_fluid,
                state=fluid_state, boundaries=uncoupled_fluid_boundaries,
                time=time, quadrature_tag=quadrature_tag,
                limiter_func=limiter_func, entropy_min=smin,
                operator_states_quad=fluid_operator_states_quad)

            grad_fluid_t = fluid_grad_t_operator(
                dcoll=dcoll, gas_model=gas_model, dd=dd_vol_fluid,
                state=fluid_state, boundaries=uncoupled_fluid_boundaries,
                time=time, quadrature_tag=quadrature_tag,
                limiter_func=limiter_func, entropy_min=smin,
                operator_states_quad=fluid_operator_states_quad)

        # now compute the smoothness part
        if use_av == 1:
            av_smu = compute_smoothness(cv, dv, grad_fluid_cv)
        elif use_av == 2:
            av_smu, av_sbeta, av_skappa = \
                compute_smoothness_mbk(cv, dv, grad_fluid_cv, grad_fluid_t)
        elif use_av == 3:
            av_smu, av_sbeta, av_skappa, av_sd = \
                compute_smoothness_mbkd(cv, dv, grad_fluid_cv, grad_fluid_t)

        # update the stepper_state
        state = state.replace(cv=cv,
                              av_smu=av_smu,
                              av_sbeta=av_sbeta,
                              av_skappa=av_skappa,
                              av_sd=av_sd)
        if use_wall:
            # Make sure wall_grad_t gets used so the communication ends up in the DAG
            for idim in range(dim):
                wv = replace(wv, energy=wv.energy + 0.*grad_wall_t[idim])
            state = state.replace(wv=wv)

        return state

    # this one gets used in init/viz
    #compute_smoothness_compiled = actx.compile(compute_smoothness_wrapper) # noqa
    compute_smoothness_compiled = actx.compile(compute_smoothness) # noqa
    update_smoothness_compiled = actx.compile(update_smoothness) # noqa

    def get_production_rates(cv, temperature):
        return eos.get_production_rates(cv, temperature)

    compute_production_rates = actx.compile(get_production_rates)

    if rank == 0:
        logger.info("Initial flow conditions processing")

    ##################################
    # Set up flow initial conditions #
    ##################################

    restart_wv = None
    temperature_seed = None
    restart_av_smu = None
    restart_av_sbeta = None
    restart_av_skappa = None
    restart_av_sd = None
    restart_entropy_min = None
    # Restart from a given filename assumed to have same geometry
    if restart_filename:
        if rank == 0:
            logger.info("Restarting soln.")

        #temperature_seed = restart_data["temperature_seed"]
        #restart_cv = restart_data["cv"]
        #restart_av_smu = restart_data["av_smu"]
        #restart_av_sbeta = restart_data["av_sbeta"]
        #restart_av_skappa = restart_data["av_skappa"]

        #
        # sometimes the restart data is missing it tagging, say when restarting from
        # a non-lazy simulation
        # make a dummy init to get the tagging and append the restart data to it
        #
        dummy_cv = bulk_init(
            dcoll=dcoll, x_vec=fluid_nodes, eos=eos_init,
            time=current_t)

        dummy_cv = force_evaluation(actx, dummy_cv)

        if use_wall:
            restart_wv = restart_data["wv"]

        dummy_temperature_seed = actx.np.zeros_like(dummy_cv.mass) + init_temperature
        dummy_temperature_seed = force_evaluation(actx, dummy_temperature_seed)

        dummy_av_smu = actx.np.zeros_like(dummy_cv.mass)
        dummy_av_sbeta = actx.np.zeros_like(dummy_cv.mass)
        dummy_av_skappa = actx.np.zeros_like(dummy_cv.mass)
        dummy_av_sd = actx.np.zeros_like(dummy_cv.mass)
        dummy_entropy_min = actx.np.zeros_like(dummy_cv.mass) + limiter_smin

        # need this for species transition
        temperature_seed = restart_data["temperature_seed"] +\
                           0.*dummy_temperature_seed

        # figure out cv first
        if restart_nspecies != nspecies:
            if rank == 0:
                print(f"Transitioning restart from {restart_nspecies} to {nspecies}")
                print("Preserving pressure and temperature")

            restart_eos = IdealSingleGas(gamma=gamma, gas_const=r)

            restart_cv = restart_data["cv"]

            mass = restart_cv.mass
            velocity = restart_cv.momentum/mass
            species_mass_frac_multi = 0.*mass*y

            pressure = restart_eos.pressure(restart_cv)
            temperature = restart_eos.temperature(restart_cv, temperature_seed)

            if nspecies > 3:
                if restart_nspecies == 0:
                    species_mass_frac_multi[i_ox] = mf_o2
                    species_mass_frac_multi[i_di] = (1. - mf_o2)

                if restart_nspecies > 0:
                    species = restart_cv.species_mass_fractions

                    # air is species 0 in scalar sim
                    species_mass_frac_multi[i_ox] = mf_o2*species[0]
                    species_mass_frac_multi[i_di] = (1. - mf_o2)*species[0]

                    # fuel is species 1 in scalar sim
                    species_mass_frac_multi[i_c2h4] = mf_c2h4*species[1]
                    species_mass_frac_multi[i_h2] = mf_h2*species[1]

                internal_energy = eos.get_internal_energy(temperature=temperature,
                    species_mass_fractions=species_mass_frac_multi)

                modified_mass = eos.get_density(pressure, temperature,
                                                species_mass_frac_multi)

                total_energy = modified_mass*(
                    internal_energy + np.dot(velocity, velocity)/(2.0))

                modified_cv = make_conserved(
                    dim,
                    mass=modified_mass,
                    momentum=modified_mass*velocity,
                    energy=total_energy,
                    species_mass=modified_mass*species_mass_frac_multi)
            else:
                modified_cv = make_conserved(
                    dim,
                    mass=restart_cv.mass,
                    momentum=restart_cv.momentum,
                    energy=restart_cv.energy,
                    species_mass=restart_cv.mass*y)

            restart_cv = modified_cv
        else:
            restart_cv = restart_data["cv"] + 0.*dummy_cv

        # get rest of the actual restart data
        restart_av_smu = restart_data["av_smu"] + 0.*dummy_av_smu
        restart_av_sbeta = restart_data["av_sbeta"] + 0.*dummy_av_sbeta
        restart_av_skappa = restart_data["av_skappa"] + 0.*dummy_av_skappa

        # backwards compatibility, for before entropy_min
        try:
            restart_entropy_min = restart_data["entropy_min"] + 0.*dummy_entropy_min
        except (KeyError):
            restart_entropy_min = actx.np.zeros_like(restart_cv.mass)\
                + limiter_smin + 0.*dummy_entropy_min
            if rank == 0:
                print("no data for entropy_min in restart file")
                print(f"entropy_min will be initialzed to {limiter_smin=}")

        # this is so we can restart from legacy, before use_av=3
        try:
            restart_av_sd = restart_data["av_sd"] + 0.*dummy_av_sd
        except (KeyError):
            restart_av_sd = actx.np.zeros_like(restart_av_smu)
            if rank == 0:
                print("no data for av_sd in restart file")
                print("av_sd will be initialzed to 0 on the mesh")

        if use_wall:
            restart_wv = restart_data["wv"]

        if restart_order != order:
            restart_dcoll = create_discretization_collection(
                actx,
                volume_meshes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=restart_order, tensor_product_elements=use_tpe)
            from meshmode.discretization.connection import make_same_mesh_connection
            fluid_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_fluid),
                restart_dcoll.discr_from_dd(dd_vol_fluid)
            )
            if use_wall:
                wall_connection = make_same_mesh_connection(
                    actx,
                    dcoll.discr_from_dd(dd_vol_wall),
                    restart_dcoll.discr_from_dd(dd_vol_wall)
                )
            restart_cv = fluid_connection(restart_data["cv"])
            restart_av_smu = fluid_connection(restart_data["av_smu"])
            restart_av_sbeta = fluid_connection(restart_data["av_sbeta"])
            restart_av_skappa = fluid_connection(restart_data["av_skappa"])

            # this is so we can restart from legacy, before use_av=3
            try:
                restart_av_sd = fluid_connection(restart_data["av_sd"])
            except (KeyError):
                restart_av_sd = actx.np.zeros_like(restart_av_smu)
                if rank == 0:
                    print("no data for av_sd in restart file")
                    print("av_sd will be initialzed to 0 on the mesh")

            temperature_seed = fluid_connection(restart_data["temperature_seed"])
            if use_wall:
                restart_wv = wall_connection(restart_data["wv"])

    # Or restart from a previous axisymmetric version of geometry
    elif restart_from_axi:

        from mirgecom.simutil import remap_dofarrays_in_structure

        if rank == 0:
            logger.info("Restarting from axisymmetric soln.")

        # For now, hardcode the axi->3D assuming Y is the axis
        # of symmetry in the axi run and X is the long axis in
        # 3D.
        def target_point_map(target_point):
            tx = target_point[0]
            ty = target_point[1]
            tz = target_point[2]
            y_axi = tx
            x_axi = np.sqrt(ty*ty + tz*tz)
            return np.array([x_axi, y_axi])

        # Take care to respect volume-specific data items
        axi_fluid_items = {
            "tseed": axi_restart_data["temperature_seed"],
            "cv": axi_restart_data["cv"],
            "av_smu": axi_restart_data["av_smu"],
            "av_sbeta": axi_restart_data["av_sbeta"],
            "av_skappa": axi_restart_data["av_skappa"]
        }

        # this is so we can restart from legacy, before use_av=3
        try:
            axi_av_sd = axi_restart_data["av_sd"]
        except (KeyError):
            axi_av_sd = actx.np.zeros_like(axi_fluid_items["av_smu"])
            if rank == 0:
                print("no data for av_sd in axi restart file")
                print("av_sd will be initialzed to 0 on the mesh")
        axi_fluid_items["av_sd"] = axi_av_sd

        x_vol = "fluid"
        fluid_restart_items = remap_dofarrays_in_structure(
            actx, axi_fluid_items, axi_vol_meshes,
            vol_meshes, target_point_map=target_point_map,
            volume_id=x_vol)

        axi_cv = fluid_restart_items["cv"]
        axi_mom = axi_cv.momentum
        fl_y = fluid_nodes[1]
        fl_z = fluid_nodes[2]
        fl_r = actx.np.sqrt(fl_y*fl_y + fl_z*fl_z)
        xfer_mom_x = axi_mom[1]  # y-component of axi maps to Vx
        xfer_mom_y = axi_mom[0] * fl_y/fl_r  # Vy_axi * r_hat_x
        xfer_mom_z = axi_mom[0] * fl_z/fl_r  # Vy_axi * r_hat_y
        restart_mom = make_obj_array([xfer_mom_x, xfer_mom_y, xfer_mom_z])
        restart_cv = axi_cv.replace(momentum=restart_mom)

        temperature_seed = fluid_restart_items["tseed"]
        restart_av_smu = fluid_restart_items["av_smu"]
        restart_av_sbeta = fluid_restart_items["av_sbeta"]
        restart_av_skappa = fluid_restart_items["av_skappa"]
        restart_av_sd = fluid_restart_items["av_sd"]

        if use_wall:
            axi_wall_items = {}
            axi_wall_items["wv"] = axi_restart_data["wv"]
            x_vol = "wall"
            wall_restart_items = remap_dofarrays_in_structure(
                actx, axi_wall_items, axi_vol_meshes,
                vol_meshes, target_point_map=target_point_map,
                volume_id=x_vol)
            restart_wv = wall_restart_items["wv"]

    # Intialize the solution from restarted data
    if restart_filename or restart_from_axi:

        restart_fluid_state = create_fluid_state(
            cv=restart_cv, temperature_seed=temperature_seed,
            smoothness_mu=restart_av_smu, smoothness_beta=restart_av_sbeta,
            smoothness_kappa=restart_av_skappa, smoothness_d=restart_av_sd,
            entropy_min=restart_entropy_min)

        # update current state with injection intialization
        if init_injection:
            if use_injection:
                restart_cv = bulk_init.add_injection(restart_fluid_state.cv,
                                                     restart_fluid_state.pressure,
                                                     restart_fluid_state.temperature,
                                                     eos=eos_init,
                                                     x_vec=fluid_nodes)
                restart_fluid_state = create_fluid_state(
                    cv=restart_cv, temperature_seed=temperature_seed,
                    smoothness_mu=restart_av_smu, smoothness_beta=restart_av_sbeta,
                    smoothness_kappa=restart_av_skappa, smoothness_d=restart_av_sd,
                    entropy_min=restart_entropy_min)
                temperature_seed = restart_fluid_state.temperature

            if use_upstream_injection:
                restart_cv = bulk_init.add_injection_upstream(
                    restart_fluid_state.cv, restart_fluid_state.pressure,
                    restart_fluid_state.temperature, eos=eos_init, x_vec=fluid_nodes)
                restart_fluid_state = create_fluid_state(
                    cv=restart_cv, temperature_seed=temperature_seed,
                    smoothness_mu=restart_av_smu, smoothness_beta=restart_av_sbeta,
                    smoothness_kappa=restart_av_skappa, smoothness_d=restart_av_sd,
                    entropy_min=restart_entropy_min)
                temperature_seed = restart_fluid_state.temperature

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

    # Not a restart of any sort, Set the current state from time 0
    else:
        if rank == 0:
            logger.info("Initializing soln.")
        restart_cv = bulk_init(
            dcoll=dcoll, x_vec=fluid_nodes, eos=eos_init,
            time=current_t)

        restart_cv = force_evaluation(actx, restart_cv)

        temperature_seed = actx.np.zeros_like(restart_cv.mass) + init_temperature
        temperature_seed = force_evaluation(actx, temperature_seed)

        restart_av_smu = actx.np.zeros_like(restart_cv.mass)
        restart_av_sbeta = actx.np.zeros_like(restart_cv.mass)
        restart_av_skappa = actx.np.zeros_like(restart_cv.mass)
        restart_av_sd = actx.np.zeros_like(restart_cv.mass)
        restart_entropy_min = actx.np.zeros_like(restart_cv.mass) + limiter_smin

        # get the initial temperature field to use as a seed
        restart_fluid_state = create_fluid_state(cv=restart_cv,
                                                 temperature_seed=temperature_seed,
                                                 smoothness_mu=restart_av_smu,
                                                 smoothness_beta=restart_av_sbeta,
                                                 smoothness_kappa=restart_av_skappa,
                                                 smoothness_d=restart_av_sd,
                                                 entropy_min=restart_entropy_min)
        temperature_seed = restart_fluid_state.temperature

        # this is a little funky, need a better way of handling this
        # most of the initializations just create the initial cv and exit
        # but I've started breaking off certain parts to use in other pieces of the
        # driver. See adding injection to an already running simulation (restart)
        # or developing a time-dependent sponge.
        if init_case == "y3prediction_ramp" or init_case == "unstart_ramp":
            restart_cv = bulk_init.add_inlet(
                cv=restart_fluid_state.cv, pressure=restart_fluid_state.pressure,
                temperature=restart_fluid_state.temperature,
                eos=eos_init, x_vec=fluid_nodes, time=current_t)
            restart_fluid_state = create_fluid_state(
                cv=restart_cv, temperature_seed=temperature_seed,
                smoothness_mu=restart_av_smu, smoothness_beta=restart_av_sbeta,
                smoothness_kappa=restart_av_skappa,
                smoothness_d=restart_av_sd,
                entropy_min=restart_entropy_min)
            temperature_seed = restart_fluid_state.temperature

            restart_cv = bulk_init.add_outlet(
                cv=restart_fluid_state.cv, pressure=restart_fluid_state.pressure,
                temperature=restart_fluid_state.temperature,
                eos=eos_init, x_vec=fluid_nodes, time=current_t)
            restart_fluid_state = create_fluid_state(
                cv=restart_cv, temperature_seed=temperature_seed,
                smoothness_mu=restart_av_smu, smoothness_beta=restart_av_sbeta,
                smoothness_kappa=restart_av_skappa,
                smoothness_d=restart_av_sd,
                entropy_min=restart_entropy_min)
            temperature_seed = restart_fluid_state.temperature

        # update current state with injection intialization
        if use_injection:
            restart_cv = bulk_init.add_injection(
                cv=restart_fluid_state.cv, pressure=restart_fluid_state.pressure,
                temperature=restart_fluid_state.temperature,
                x_vec=fluid_nodes, eos=eos_init, time=current_t)
            restart_fluid_state = create_fluid_state(
                cv=restart_cv, temperature_seed=temperature_seed,
                smoothness_mu=restart_av_smu, smoothness_beta=restart_av_sbeta,
                smoothness_kappa=restart_av_skappa,
                smoothness_d=restart_av_sd,
                entropy_min=restart_entropy_min)
            temperature_seed = restart_fluid_state.temperature

        if use_upstream_injection:
            restart_cv = bulk_init.add_injection_upstream(
                cv=restart_fluid_state.cv, pressure=restart_fluid_state.pressure,
                temperature=restart_fluid_state.temperature,
                x_vec=fluid_nodes, eos=eos_init, time=current_t)
            restart_fluid_state = create_fluid_state(
                cv=restart_cv, temperature_seed=temperature_seed,
                smoothness_mu=restart_av_smu, smoothness_beta=restart_av_sbeta,
                smoothness_kappa=restart_av_skappa,
                smoothness_d=restart_av_sd,
                entropy_min=restart_entropy_min)
            temperature_seed = restart_fluid_state.temperature

        # Ideally we would compute the smoothness variables here,
        # but we need the boundary conditions (and hence the target state) first,
        # so we defer until after those are setup

        # initialize the wall
        if use_wall:
            wall_mass = (
                wall_insert_rho * wall_insert_mask
                + wall_surround_rho * wall_surround_mask)
            wall_cp = (
                wall_insert_cp * wall_insert_mask
                + wall_surround_cp * wall_surround_mask)
            restart_wv = WallVars(
                mass=wall_mass,
                energy=wall_mass * wall_cp * temp_wall,
                ox_mass=actx.np.zeros_like(wall_mass))

    if use_wall:
        restart_wv = force_evaluation(actx, restart_wv)

    ##################################
    # Set up flow target state       #
    ##################################

    if rank == 0:
        logger.info("Initializing target state.")

    if target_filename:
        if rank == 0:
            logger.info("Reading target soln.")
        if target_order != order:
            target_dcoll = create_discretization_collection(
                actx, volume_meshes=vol_meshes,
                order=target_order, tensor_product_elements=use_tpe)
            from meshmode.discretization.connection import make_same_mesh_connection
            fluid_connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd(dd_vol_fluid),
                target_dcoll.discr_from_dd(dd_vol_fluid)
            )
            target_cv = fluid_connection(target_data["cv"])
            target_av_smu = fluid_connection(target_data["av_smu"])
            target_av_sbeta = fluid_connection(target_data["av_sbeta"])
            target_av_skappa = fluid_connection(target_data["av_skappa"])
            target_av_sd = fluid_connection(target_data["av_sd"])
        else:

            #
            # sometimes the restart data is missing it tagging,
            # say when restarting from a non-lazy simulation
            # make a dummy init to get the tagging and append the restart data to it
            #
            dummy_cv = bulk_init(
                dcoll=dcoll, x_vec=fluid_nodes, eos=eos_init,
                time=current_t)

            dummy_cv = force_evaluation(actx, dummy_cv)
            dummy_zeros = actx.np.zeros_like(dummy_cv.mass)

            target_cv = target_data["cv"] + 0.*dummy_cv
            target_av_smu = target_data["av_smu"] + 0.*dummy_zeros
            target_av_sbeta = target_data["av_sbeta"] + 0.*dummy_zeros
            target_av_skappa = target_data["av_skappa"] + 0.*dummy_zeros
            # this is so we can restart from legacy, before use_av=3
            try:
                target_av_sd = target_data["av_sd"] + 0.*dummy_zeros
            except (KeyError):
                target_av_sd = actx.np.zeros_like(target_av_smu)
                if rank == 0:
                    print("no data for av_sd in target file")
                    print("av_sd will be initialzed to 0 on the mesh")

        if target_nspecies != nspecies:
            if rank == 0:
                print(f"Transitioning target from {target_nspecies} to {nspecies}")
                print("Preserving pressure and temperature")

            target_eos = IdealSingleGas(gamma=gamma, gas_const=r)

            mass = target_cv.mass
            velocity = target_cv.momentum/mass
            species_mass_frac_multi = 0.*mass*y

            pressure = target_eos.pressure(target_cv)
            temperature = target_eos.temperature(target_cv, temperature_seed)

            if nspecies > 2:
                if target_nspecies == 0:
                    species_mass_frac_multi[i_ox] = mf_o2
                    species_mass_frac_multi[i_di] = (1. - mf_o2)

                if target_nspecies > 0:
                    species = target_cv.species_mass_fractions

                    # air is species 0 in scalar sim
                    species_mass_frac_multi[i_ox] = mf_o2*species[0]
                    species_mass_frac_multi[i_di] = (1. - mf_o2)*species[0]

                    # fuel is species 1 in scalar sim
                    species_mass_frac_multi[i_c2h4] = mf_c2h4*species[1]
                    species_mass_frac_multi[i_h2] = mf_h2*species[1]

                internal_energy = eos.get_internal_energy(temperature=temperature,
                    species_mass_fractions=species_mass_frac_multi)

                modified_mass = eos.get_density(pressure, temperature,
                                                species_mass_frac_multi)

                total_energy = modified_mass*(
                    internal_energy + np.dot(velocity, velocity)/(2.0))

                modified_cv = make_conserved(
                    dim,
                    mass=modified_mass,
                    momentum=modified_mass*velocity,
                    energy=total_energy,
                    species_mass=modified_mass*species_mass_frac_multi)
            else:
                modified_cv = make_conserved(
                    dim,
                    mass=target_cv.mass,
                    momentum=target_cv.momentum,
                    energy=target_cv.energy,
                    species_mass=target_cv.mass*y)

            target_cv = modified_cv

        target_cv = force_evaluation(actx, target_cv)
        target_av_smu = force_evaluation(actx, target_av_smu)
        target_av_sbeta = force_evaluation(actx, target_av_sbeta)
        target_av_skappa = force_evaluation(actx, target_av_skappa)
        target_av_sd = force_evaluation(actx, target_av_sd)
        target_entropy_min = actx.np.zeros_like(target_cv.mass) + limiter_smin

        target_fluid_state = create_fluid_state(cv=target_cv,
                                                temperature_seed=temperature_seed,
                                                smoothness_mu=target_av_smu,
                                                smoothness_beta=target_av_sbeta,
                                                smoothness_kappa=target_av_skappa,
                                                smoothness_d=target_av_sd,
                                                entropy_min=target_entropy_min)

    else:
        # Set the current state from time 0
        target_cv = restart_cv
        target_av_smu = restart_av_smu
        target_av_sbeta = restart_av_sbeta
        target_av_skappa = restart_av_skappa
        target_av_sd = restart_av_sd
        target_entropy_min = actx.np.zeros_like(target_cv.mass) + limiter_smin

        target_fluid_state = restart_fluid_state

    if rank == 0:
        logger.info("More gradient processing")

    target_boundaries = {}

    def grad_cv_operator_target(fluid_state, time):
        return grad_cv_operator(dcoll=dcoll, gas_model=gas_model,
                                dd=dd_vol_fluid,
                                boundaries=target_boundaries,
                                state=fluid_state,
                                time=time,
                                quadrature_tag=quadrature_tag)

    grad_cv_operator_target_compiled = actx.compile(grad_cv_operator_target) # noqa

    def grad_t_operator_target(fluid_state, time):
        return fluid_grad_t_operator(
            dcoll=dcoll,
            gas_model=gas_model,
            dd=dd_vol_fluid,
            boundaries=target_boundaries,
            state=fluid_state,
            time=time,
            quadrature_tag=quadrature_tag)

    grad_t_operator_target_compiled = actx.compile(grad_t_operator_target)

    # use dummy boundaries to update the smoothness state for the target
    if use_av > 0:
        target_bndry_mapping = bndry_mapping
        target_bndry_mapping["prescribed"] = DummyBoundary()
        target_bndry_mapping["isentropic_pressure_ramp"] = DummyBoundary()

        target_boundaries = assign_fluid_boundaries(
            target_boundaries, target_bndry_mapping)

        target_grad_cv = grad_cv_operator_target_compiled(
            target_fluid_state, time=0.)
        # the target is not used along the wall, so we won't jump
        # through all the hoops to get the proper gradient

        if use_av == 1:
            target_av_smu = compute_smoothness(
                cv=target_cv, dv=target_fluid_state.dv, grad_cv=target_grad_cv)
        elif use_av == 2:
            target_grad_t = grad_t_operator_target_compiled(
                target_fluid_state, time=0.)

            target_av_sbeta, target_av_skappa, target_av_smu = \
                compute_smoothness_mbk(
                    cv=target_cv, dv=target_fluid_state.dv,
                    grad_cv=target_grad_cv, grad_t=target_grad_t)

        target_av_smu = force_evaluation(actx, target_av_smu)
        target_av_sbeta = force_evaluation(actx, target_av_sbeta)
        target_av_skappa = force_evaluation(actx, target_av_skappa)
        target_av_sd = force_evaluation(actx, target_av_sd)

        target_fluid_state = create_fluid_state(
            cv=target_cv, temperature_seed=temperature_seed,
            smoothness_mu=target_av_smu, smoothness_beta=target_av_sbeta,
            smoothness_kappa=target_av_skappa,
            smoothness_d=target_av_sd,
            entropy_min=target_entropy_min)

    ##################################
    # Set up the boundary conditions #
    ##################################

    # pressure ramp function
    # linearly ramp the pressure from beginP to finalP over t_ramp_interval seconds
    # provides an offset to start the ramping after t_ramp_start
    #
    inlet_mach = configurate("inlet_mach", input_data, 0.1)
    ramp_beginP = configurate("ramp_beginP", input_data, 100.0)
    ramp_endP = configurate("ramp_endP", input_data, 1000.0)
    ramp_time_start = configurate("ramp_time_start", input_data, 0.0)
    ramp_time_interval = configurate("ramp_time_interval", input_data, 1.e-4)

    def inflow_ramp_pressure(t):
        return actx.np.where(
            actx.np.greater(t, ramp_time_start),
            actx.np.minimum(
                ramp_endP,
                ramp_beginP + ((t - ramp_time_start) / ramp_time_interval
                    * (ramp_endP - ramp_beginP))),
            ramp_beginP)

    if init_case == "unstart" or init_case == "unstart_ramp":
        normal_dir = np.zeros(shape=(dim,))
        if use_axisymmetric:
            normal_dir[1] = 1
            inflow_state = IsentropicInflow(
                dim=dim,
                temp_wall=temp_wall,
                temp_sigma=temp_sigma,
                vel_sigma=vel_sigma,
                smooth_x0=-0.013,
                smooth_x1=0.013,
                normal_dir=normal_dir,
                gamma=gamma,
                nspecies=nspecies,
                mass_frac=y,
                T0=total_temp_inflow,
                P0=ramp_beginP,
                mach=inlet_mach,
                p_fun=inflow_ramp_pressure)
        else:
            normal_dir[0] = 1
            if dim == 2:
                inflow_state = IsentropicInflow(
                    dim=dim,
                    temp_wall=temp_wall,
                    temp_sigma=temp_sigma,
                    vel_sigma=vel_sigma,
                    smooth_y0=-0.013,
                    smooth_y1=0.013,
                    normal_dir=normal_dir,
                    gamma=gamma,
                    nspecies=nspecies,
                    mass_frac=y,
                    T0=total_temp_inflow,
                    P0=ramp_beginP,
                    mach=inlet_mach,
                    p_fun=inflow_ramp_pressure)
            else:
                r0 = np.zeros(shape=(dim,))
                r0[0] = -0.3925
                inflow_state = IsentropicInflow(
                    dim=dim,
                    temp_wall=temp_wall,
                    temp_sigma=temp_sigma,
                    vel_sigma=vel_sigma,
                    smooth_r0=r0,
                    smooth_r1=0.013,
                    normal_dir=normal_dir,
                    gamma=gamma,
                    nspecies=nspecies,
                    mass_frac=y,
                    T0=total_temp_inflow,
                    P0=ramp_beginP,
                    mach=inlet_mach,
                    p_fun=inflow_ramp_pressure)
    else:
        inflow_state = IsentropicInflow(
            dim=dim,
            temp_wall=temp_wall,
            temp_sigma=temp_sigma,
            vel_sigma=vel_sigma,
            smooth_y0=-0.0270645,
            smooth_y1=0.0270645,
            gamma=gamma,
            nspecies=nspecies,
            mass_frac=y,
            T0=total_temp_inflow,
            P0=ramp_beginP,
            mach=inlet_mach,
            p_fun=inflow_ramp_pressure)

    def get_inflow_boundary_solution(dcoll, dd_bdry, gas_model,
                                     state_minus, time, **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        tmp = inflow_state(x_vec=nodes, gas_model=gas_model, time=time, **kwargs)
        return tmp

    def get_target_state_on_boundary(btag):
        return project_fluid_state(
            dcoll, dd_vol_fluid,
            dd_vol_fluid.trace(btag).with_discr_tag(quadrature_tag),
            target_fluid_state, gas_model,
            entropy_stable=use_esdg
        )

    # is there a way to generalize this?
    if bndry_config["inflow"] == "isentropic_pressure_ramp":
        prescribed_inflow_boundary = PrescribedFluidBoundary(
            boundary_state_func=get_inflow_boundary_solution)

        bndry_config["inflow"] = "isentropic_pressure_ramp"
        bndry_mapping["isentropic_pressure_ramp"] = prescribed_inflow_boundary

    if bndry_config["flow"] == "prescribed":
        flow_ref_state = \
            get_target_state_on_boundary("flow")

        flow_ref_state = force_evaluation(actx, flow_ref_state)

        def _target_flow_state_func(**kwargs):
            return flow_ref_state

        prescribed_flow_boundary = PrescribedFluidBoundary(
            boundary_state_func=_target_flow_state_func)

        bndry_config["flow"] = "prescribed_flow"
        bndry_mapping["prescribed_flow"] = prescribed_flow_boundary

    if bndry_config["inflow"] == "prescribed":
        inflow_ref_state = \
            get_target_state_on_boundary("inflow")

        inflow_ref_state = force_evaluation(actx, inflow_ref_state)

        def _target_inflow_state_func(**kwargs):
            return inflow_ref_state

        prescribed_inflow_boundary = PrescribedFluidBoundary(
            boundary_state_func=_target_inflow_state_func)

        bndry_config["inflow"] = "prescribed_inflow"
        bndry_mapping["prescribed_inflow"] = prescribed_inflow_boundary

    if bndry_config["outflow"] == "prescribed":
        outflow_ref_state = \
            get_target_state_on_boundary("outflow")

        outflow_ref_state = force_evaluation(actx, outflow_ref_state)

        def _target_outflow_state_func(**kwargs):
            return outflow_ref_state

        prescribed_outflow_boundary = PrescribedFluidBoundary(
            boundary_state_func=_target_outflow_state_func)

        bndry_config["outflow"] = "prescribed_outflow"
        bndry_mapping["prescribed_outflow"] = prescribed_outflow_boundary

    if bndry_config["upstream_injection"] == "prescribed":
        upstream_injection_ref_state = \
            get_target_state_on_boundary("upstream_injection")

        upstream_injection_ref_state = force_evaluation(
            actx, upstream_injection_ref_state)

        def _target_upstream_injection_state_func(**kwargs):
            return upstream_injection_ref_state

        prescribed_upstream_injection_boundary = PrescribedFluidBoundary(
            boundary_state_func=_target_upstream_injection_state_func)

        bndry_config["upstream_injection"] = "prescribed_upstream_injection"
        bndry_mapping["prescribed_upstream_injection"] = \
            prescribed_upstream_injection_boundary

    if bndry_config["injection"] == "prescribed":
        injection_ref_state = \
            get_target_state_on_boundary("injection")

        injection_ref_state = force_evaluation(actx, injection_ref_state)

        def _target_injection_state_func(**kwargs):
            return injection_ref_state

        prescribed_injection_boundary = PrescribedFluidBoundary(
            boundary_state_func=_target_injection_state_func)

        bndry_config["injection"] = "prescribed_injection"
        bndry_mapping["prescribed_injection"] = prescribed_injection_boundary

    if bndry_config["wall_interface"] == "prescribed":
        interface_ref_state = \
            get_target_state_on_boundary("wall_interface")

        interface_ref_state = force_evaluation(actx, interface_ref_state)

        def _target_interface_state_func(**kwargs):
            return interface_ref_state

        prescribed_interface_boundary = PrescribedFluidBoundary(
            boundary_state_func=_target_interface_state_func)

        bndry_config["wall_interface"] = "prescribed_interface"
        bndry_mapping["prescribed_interface"] = prescribed_interface_boundary

    uncoupled_fluid_boundaries = {}
    uncoupled_fluid_boundaries = assign_fluid_boundaries(
        uncoupled_fluid_boundaries, bndry_mapping)

    # check the boundary condition coverage
    from meshmode.mesh import check_bc_coverage
    try:
        bound_list = []
        for bound in list(uncoupled_fluid_boundaries.keys()):
            bound_list.append(bound.tag)
        print(f"{bound_list=}")
        check_bc_coverage(mesh=dcoll.discr_from_dd(dd_vol_fluid).mesh,
                          boundary_tags=bound_list,
                          incomplete_ok=True)
    except (ValueError, RuntimeError):
        print(f"{uncoupled_fluid_boundaries=}")
        raise SimulationConfigurationError(
            "Invalid boundary configuration specified"
        )

    if use_wall:
        uncoupled_wall_boundaries = {
            wall_ffld_bnd.domain_tag: wall_farfield  # pylint: disable=no-member
        }

    current_wv = None
    if use_wall:
        current_wv = force_evaluation(actx, restart_wv)

    restart_stepper_state = make_stepper_state(
        cv=restart_cv,
        tseed=temperature_seed,
        wv=restart_wv,
        av_smu=restart_av_smu,
        av_sbeta=restart_av_sbeta,
        av_skappa=restart_av_skappa,
        av_sd=restart_av_sd,
        smin=restart_entropy_min)

    # finish initializing the smoothness for non-restarts
    if not restart_filename:
        if use_av > 0:
            restart_stepper_state = update_smoothness_compiled(
                state=restart_stepper_state, time=current_t)

    restart_cv = force_evaluation(actx, restart_stepper_state.cv)
    temperature_seed = force_evaluation(actx, temperature_seed)
    restart_av_smu = force_evaluation(actx, restart_stepper_state.av_smu)
    restart_av_sbeta = force_evaluation(actx, restart_stepper_state.av_sbeta)
    restart_av_skappa = force_evaluation(actx, restart_stepper_state.av_skappa)
    restart_av_sd = force_evaluation(actx, restart_stepper_state.av_sd)
    restart_entropy_min = force_evaluation(actx, restart_stepper_state.smin)

    # set the initial data used by the simulation
    current_fluid_state = create_fluid_state(cv=restart_cv,
                                             temperature_seed=temperature_seed,
                                             smoothness_mu=restart_av_smu,
                                             smoothness_beta=restart_av_sbeta,
                                             smoothness_kappa=restart_av_skappa,
                                             smoothness_d=restart_av_sd,
                                             entropy_min=restart_entropy_min)

    if use_wall:
        current_wv = force_evaluation(actx, restart_stepper_state.wv)

    stepper_state = make_stepper_state(
        cv=current_fluid_state.cv,
        tseed=temperature_seed,
        wv=current_wv,
        av_smu=current_fluid_state.dv.smoothness_mu,
        av_sbeta=current_fluid_state.dv.smoothness_beta,
        av_skappa=current_fluid_state.dv.smoothness_kappa,
        av_sd=current_fluid_state.dv.smoothness_d,
        smin=restart_entropy_min)

    ####################
    # Ignition Sources #
    ####################

    # if you divide by 2.355, 50% of the spark is within this diameter
    # if you divide by 6, 99% of the energy is deposited in this time
    #spark_diameter /= 2.355
    spark_diameter /= 6.0697
    spark_duration /= 6.0697

    # gaussian application in time
    def spark_time_func(t):
        expterm = actx.np.exp((-(t - spark_init_time)**2) /
                              (2*spark_duration*spark_duration))
        return expterm

    if use_ignition == 2:
        from y3prediction.utils import HeatSource
        ignition_source = HeatSource(dim=dim, center=spark_center,
                                      amplitude=spark_strength,
                                      amplitude_func=spark_time_func,
                                      width=spark_diameter)
    else:
        from y3prediction.utils import SparkSource
        ignition_source = SparkSource(dim=dim, center=spark_center,
                                      amplitude=spark_strength,
                                      amplitude_func=spark_time_func,
                                      width=spark_diameter)

    # gaussian application in space
    if dim == 2:
        # 99% falls within this diameter
        #injection_source_diameter /= 4.684
        # 95% falls within this diameter
        injection_source_diameter /= 2.83
    else:
        # 95% falls in this diameter
        injection_source_diameter /= 5.08

    def injection_source_time_func(t):
        scaled_time = injection_source_init_time - t
        # this gives about 99% of the change in the requested time
        xtanh = 4.684*scaled_time/injection_source_ramp_time
        return 0.5*(1.0 - actx.np.tanh(xtanh))

    from y3prediction.utils import StateSource, StateSource3d
    source_mass = injection_source_mass
    source_mom = np.zeros(shape=(dim,))
    source_mom[0] = injection_source_mom_x
    source_mom[1] = injection_source_mom_y
    if dim == 3:
        source_mom[2] = injection_source_mom_z
    source_energy = injection_source_energy
    source_y = y_fuel
    injection_source = StateSource(dim=dim, nspecies=nspecies,
                                   center=injection_source_center,
                                   mass_amplitude=source_mass,
                                   mom_amplitude=source_mom,
                                   energy_amplitude=source_energy,
                                   y_amplitude=source_y,
                                   amplitude_func=injection_source_time_func,
                                   #amplitude_func=None,
                                   axisymmetric=use_axisymmetric,
                                   width=injection_source_diameter)

    source_mass = injection_source_mass_comb
    source_mom = np.zeros(shape=(dim,))
    source_mom[0] = injection_source_mom_x_comb
    source_mom[1] = injection_source_mom_y_comb
    if dim == 3:
        source_mom[2] = injection_source_mom_z_comb
    source_energy = injection_source_energy_comb
    injection_source_comb = StateSource(dim=dim, nspecies=nspecies,
                                        center=injection_source_center_comb,
                                        mass_amplitude=source_mass,
                                        mom_amplitude=source_mom,
                                        energy_amplitude=source_energy,
                                        y_amplitude=source_y,
                                        amplitude_func=injection_source_time_func,
                                        #amplitude_func=None,
                                        axisymmetric=use_axisymmetric,
                                        width=injection_source_diameter)
    injection_source_3d = StateSource3d(dim=dim, nspecies=nspecies,
                                        center=injection_source_center_comb,
                                        mass_amplitude=source_mass,
                                        mom_amplitude=source_mom,
                                        energy_amplitude=source_energy,
                                        y_amplitude=source_y,
                                        amplitude_func=injection_source_time_func,
                                        width=injection_source_diameter)

    if rank == 0:
        logger.info("Sponges processsing")
    ##################
    # Sponge Sources #
    ##################

    # initialize the sponge field
    sponge_amp = sponge_sigma/current_dt/1000
    from y3prediction.utils import InitSponge

    if init_case == "y3prediction" or init_case == "y3prediction_ramp":
        sponge_init_inlet = InitSponge(x0=inlet_sponge_x0,
                                       thickness=inlet_sponge_thickness,
                                       amplitude=sponge_amp,
                                       direction=-1.0)
        sponge_init_outlet = InitSponge(x0=outlet_sponge_x0,
                                        thickness=outlet_sponge_thickness,
                                        amplitude=sponge_amp)
        if use_injection:
            sponge_init_injection =\
                InitSponge(x0=inj_sponge_x0, thickness=inj_sponge_thickness,
                           amplitude=sponge_amp,
                           xmax=0.66, ymax=-0.01)

        #if use_upstream_injection:
            sponge_init_upstream_injection =\
                InitSponge(x0=upstream_inj_sponge_y0,
                           thickness=inj_sponge_thickness,
                           amplitude=sponge_amp,
                           xmin=0.53, xmax=0.535,
                           ymin=-0.02253, direction=-2.0)

        def _sponge_sigma(sponge_field, x_vec):
            sponge_field = sponge_init_outlet(sponge_field=sponge_field, x_vec=x_vec)
            sponge_field = sponge_init_inlet(sponge_field=sponge_field, x_vec=x_vec)
            if use_injection:
                sponge_field = sponge_init_injection(
                    sponge_field=sponge_field, x_vec=x_vec)
            #if use_upstream_injection:
                sponge_field = sponge_init_upstream_injection(
                    sponge_field=sponge_field, x_vec=x_vec)
            return sponge_field

    elif (init_case == "shock1d" or
          init_case == "flame1d" or
          init_case == "forward_step" or
          init_case == "backward_step"):

        sponge_init_inlet = InitSponge(x0=inlet_sponge_x0,
                                       thickness=inlet_sponge_thickness,
                                       amplitude=sponge_amp,
                                       direction=-1.0)
        sponge_init_outlet = InitSponge(x0=outlet_sponge_x0,
                                        thickness=outlet_sponge_thickness,
                                        amplitude=sponge_amp)

        def _sponge_sigma(sponge_field, x_vec):
            sponge_field = sponge_init_outlet(sponge_field=sponge_field, x_vec=x_vec)
            sponge_field = sponge_init_inlet(sponge_field=sponge_field, x_vec=x_vec)
            return sponge_field

    elif init_case == "mixing_layer" or init_case == "mixing_layer_hot":

        top_sponge_y0 = 0.006
        top_sponge_thickness = 0.002
        bottom_sponge_y0 = -0.006
        bottom_sponge_thickness = 0.002
        sponge_init_bottom = InitSponge(x0=bottom_sponge_y0,
                                        thickness=bottom_sponge_thickness,
                                        amplitude=sponge_amp,
                                        direction=-2.0)
        sponge_init_top = InitSponge(x0=top_sponge_y0,
                                     thickness=top_sponge_thickness,
                                     amplitude=sponge_amp,
                                     direction=2.0)

        def _sponge_sigma(sponge_field, x_vec):
            sponge_field = sponge_init_bottom(sponge_field=sponge_field, x_vec=x_vec)
            sponge_field = sponge_init_top(sponge_field=sponge_field, x_vec=x_vec)
            return sponge_field

    elif init_case == "unstart" or init_case == "unstart_ramp":
        if dim == 2:
            sponge_init_inlet = InitSponge(x0=inlet_sponge_x0,
                                           thickness=inlet_sponge_thickness,
                                           amplitude=sponge_amp,
                                           direction=-2)
            sponge_init_outlet = InitSponge(x0=outlet_sponge_x0,
                                            thickness=outlet_sponge_thickness,
                                            amplitude=sponge_amp,
                                            direction=2)
            sponge_init_top = InitSponge(x0=top_sponge_x0,
                                         thickness=top_sponge_thickness,
                                         amplitude=sponge_amp,
                                         direction=1)
        else:
            sponge_init_inlet = InitSponge(x0=inlet_sponge_x0,
                                           thickness=inlet_sponge_thickness,
                                           amplitude=sponge_amp,
                                           direction=-1)
            sponge_init_outlet = InitSponge(x0=outlet_sponge_x0,
                                            thickness=outlet_sponge_thickness,
                                            amplitude=sponge_amp,
                                            direction=1)

        def _sponge_sigma(sponge_field, x_vec):
            sponge_field = sponge_init_outlet(sponge_field=sponge_field, x_vec=x_vec)
            sponge_field = sponge_init_inlet(sponge_field=sponge_field, x_vec=x_vec)
            if dim == 2:
                sponge_field = sponge_init_top(
                    sponge_field=sponge_field, x_vec=x_vec)
            return sponge_field

    sponge_sigma = actx.np.zeros_like(restart_cv.mass)
    if use_sponge:
        get_sponge_sigma = actx.compile(_sponge_sigma)
        sponge_sigma = force_evaluation(actx, get_sponge_sigma(sponge_sigma,
                                                               fluid_nodes))

    def _sponge_source(sigma, cv, sponge_cv):
        """Create sponge source."""
        return sigma*(sponge_cv - cv)

    vis_timer = None
    monitor_memory = True
    monitor_performance = 2

    from contextlib import nullcontext
    gc_timer = nullcontext()

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        gc_timer_init = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer_init)
        gc_timer = gc_timer_init.get_sub_timer()

        if monitor_performance > 0:
            logmgr.add_watches([
                ("t_step.max", "| Performance:\n| \t walltime: {value:6g} s")
            ])

        if monitor_performance > 1:

            logmgr.add_watches([
                ("t_vis.max", "\n| \t visualization time: {value:6g} s\n"),
                ("t_gc.max", "| \t garbage collection time: {value:6g} s\n"),
                ("t_log.max", "| \t log walltime: {value:6g} s\n")
            ])

        if monitor_memory:
            logmgr_add_device_memory_usage(logmgr, queue)
            logmgr_add_mempool_usage(logmgr, alloc)

            logmgr.add_watches([
                ("memory_usage_python.max",
                 "| Memory:\n| \t host memory: {value:7g} Mb\n")
            ])

            try:
                logmgr.add_watches([
                    ("memory_usage_gpu.max",
                     "| \t device memory: {value:7g} Mb\n")
                ])
            except KeyError:
                pass

            logmgr.add_watches([
                ("memory_usage_hwm.max",
                 "| \t host memory hwm: {value:7g} Mb\n")])

            from mirgecom.array_context import actx_class_is_numpy

            if not actx_class_is_numpy(actx_class):
                # numpy has no CL mempool
                logmgr.add_watches([
                    ("memory_usage_mempool_managed.max",
                    "| \t device mempool total: {value:7g} Mb\n"),
                    ("memory_usage_mempool_active.max",
                    "| \t device mempool active: {value:7g} Mb")
                ])

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

    if rank == 0:
        logger.info("Viz & utilities processsing")

    # avoid making a second discretization if viz_order == order
    wall_visualizer = None
    if viz_order == order:
        fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)
        if use_wall:
            wall_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_wall)
    else:
        fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid,
                                           vis_order=viz_order)
        if use_wall:
            wall_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_wall,
                                              vis_order=viz_order)
    #    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    # some utility functions
    def vol_min_loc(dd_vol, x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(dcoll, dd_vol, x,
                                           initial=np.inf))[()]

    def vol_max_loc(dd_vol, x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(dcoll, dd_vol, x,
                                           initial=-np.inf))[()]

    def vol_min(dd_vol, x):
        return actx.to_numpy(nodal_min(dcoll, dd_vol, x,
                                       initial=np.inf))[()]

    def vol_max(dd_vol, x):
        return actx.to_numpy(nodal_max(dcoll, dd_vol, x,
                                       initial=-np.inf))[()]

    def global_range_check(dd_vol, array, min_val, max_val):
        return global_reduce(
            check_range_local(
                dcoll, dd_vol, array, min_val, max_val), op="lor")

    def my_write_status_lite(step, t, t_wall):
        status_msg = (f"\n--     step {step:9d}:"
                      f"\n----   fluid sim time {t:1.8e}")
        if use_wall:
            status_msg += (f", wall sim time {t_wall:1.8e}")

        if rank == 0:
            logger.info(status_msg)

    def my_write_status_fluid(fluid_state, dt, cfl_fluid):
        cv = fluid_state.cv
        dv = fluid_state.dv

        status_msg = (f"----   dt {dt:1.3e},"
                      f" cfl_fluid {cfl_fluid:1.8f}")

        pmin = vol_min(dd_vol_fluid, dv.pressure)
        pmax = vol_max(dd_vol_fluid, dv.pressure)
        tmin = vol_min(dd_vol_fluid, dv.temperature)
        tmax = vol_max(dd_vol_fluid, dv.temperature)

        from pytools.obj_array import obj_array_vectorize
        y_min = obj_array_vectorize(lambda x: vol_min(dd_vol_fluid, x),
                                      cv.species_mass_fractions)
        y_max = obj_array_vectorize(lambda x: vol_max(dd_vol_fluid, x),
                                      cv.species_mass_fractions)

        dv_status_msg = (
            f"\n------ P       (min, max) (Pa) = ({pmin:1.9e}, {pmax:1.9e})")
        dv_status_msg += (
            f"\n------ T_fluid (min, max) (K)  = ({tmin:7g}, {tmax:7g})")

        if eos_type == 1:
            # check the temperature convergence
            # a single call to get_temperature_update is like taking an additional
            # Newton iteration and gives us a residual
            temp_resid = get_temperature_update_compiled(
                cv, dv.temperature)/dv.temperature
            temp_err_min = vol_min(dd_vol_fluid, temp_resid)
            temp_err_max = vol_max(dd_vol_fluid, temp_resid)
            dv_status_msg += (
                f"\n------ T_resid (min, max)      = "
                f"({temp_err_min:1.5e}, {temp_err_max:1.5e})")

        for i in range(nspecies):
            dv_status_msg += (
                f"\n------ y_{species_names[i]:5s} (min, max)      = "
                f"({y_min[i]:1.3e}, {y_max[i]:1.3e})")
        #dv_status_msg += "\n"
        status_msg += dv_status_msg

        if rank == 0:
            logger.info(status_msg)

    def my_write_status_wall(wall_temperature, dt, cfl_wall):
        status_msg = (f"----   wall dt {dt:1.3e},"
                      f" cfl_wall {cfl_wall:1.8f}")

        twmin = vol_min(dd_vol_wall, wall_temperature)
        twmax = vol_max(dd_vol_wall, wall_temperature)

        status_msg += (
            f"\n------ T_wall  (min, max) (K)  = ({twmin:7g}, {twmax:7g})")

        if rank == 0:
            logger.info(status_msg)

    def compute_viz_fields_coupled(fluid_state, wv, wdv, time):

        cv = fluid_state.cv
        dv = fluid_state.dv

        grad_fluid_cv, grad_fluid_t, grad_wall_t = coupled_grad_operator(
            dcoll,
            gas_model,
            dd_vol_fluid, dd_vol_wall,
            uncoupled_fluid_boundaries,
            uncoupled_wall_boundaries,
            fluid_state, wdv.thermal_conductivity, wdv.temperature,
            time=time,
            interface_noslip=noslip,
            quadrature_tag=quadrature_tag,
            limiter_func=limiter_func,
            comm_tag=_InitCommTag)

        av_smu = actx.np.zeros_like(cv.mass)
        av_sbeta = actx.np.zeros_like(cv.mass)
        av_skappa = actx.np.zeros_like(cv.mass)
        av_sd = actx.np.zeros_like(cv.mass)

        # now compute the smoothness part
        if use_av == 1:
            av_smu = compute_smoothness(cv, dv, grad_fluid_cv)
        elif use_av == 2:
            av_smu, av_sbeta, av_skappa = \
                compute_smoothness_mbk(cv, dv, grad_fluid_cv, grad_fluid_t)
        elif use_av == 3:
            av_smu, av_sbeta, av_skappa, av_sd = \
                compute_smoothness_mbkd(cv, dv, grad_fluid_cv, grad_fluid_t)

        from mirgecom.fluid import (
            velocity_gradient,
            species_mass_fraction_gradient
        )
        grad_v = velocity_gradient(cv, grad_fluid_cv)
        grad_y = species_mass_fraction_gradient(cv, grad_fluid_cv)

        return make_obj_array([av_smu, av_sbeta, av_skappa, av_sd,
                               grad_v, grad_y, grad_fluid_t, grad_fluid_cv,
                               grad_wall_t])

    compute_viz_fields_coupled_compiled = actx.compile(compute_viz_fields_coupled)

    def compute_viz_fields(fluid_state, time):

        cv = fluid_state.cv
        dv = fluid_state.dv

        grad_fluid_cv = grad_cv_operator(
            dcoll=dcoll, gas_model=gas_model, dd=dd_vol_fluid,
            state=fluid_state, boundaries=uncoupled_fluid_boundaries,
            time=time, quadrature_tag=quadrature_tag)

        grad_fluid_t = fluid_grad_t_operator(
            dcoll=dcoll, gas_model=gas_model, dd=dd_vol_fluid,
            state=fluid_state, boundaries=uncoupled_fluid_boundaries,
            time=time, quadrature_tag=quadrature_tag)

        av_smu = actx.np.zeros_like(cv.mass)
        av_sbeta = actx.np.zeros_like(cv.mass)
        av_skappa = actx.np.zeros_like(cv.mass)
        av_sd = actx.np.zeros_like(cv.mass)

        # now compute the smoothness part
        if use_av == 1:
            av_smu = compute_smoothness(cv, dv, grad_fluid_cv)
        elif use_av == 2:
            av_smu, av_sbeta, av_skappa = \
                compute_smoothness_mbk(cv, dv, grad_fluid_cv, grad_fluid_t)
        elif use_av == 3:
            av_smu, av_sbeta, av_skappa, av_sd = \
                compute_smoothness_mbkd(cv, dv, grad_fluid_cv, grad_fluid_t)

        from mirgecom.fluid import (
            velocity_gradient,
            species_mass_fraction_gradient
        )
        grad_v = velocity_gradient(cv, grad_fluid_cv)
        grad_y = species_mass_fraction_gradient(cv, grad_fluid_cv)

        return make_obj_array([av_smu, av_sbeta, av_skappa, av_sd,
                               grad_v, grad_y, grad_fluid_t, grad_fluid_cv])

    compute_viz_fields_compiled = actx.compile(compute_viz_fields)

    def grad_cv(fluid_state, time):
        return grad_cv_operator(dcoll=dcoll, gas_model=gas_model,
                                dd=dd_vol_fluid,
                                boundaries=uncoupled_fluid_boundaries,
                                state=fluid_state,
                                time=time,
                                quadrature_tag=quadrature_tag)

    grad_cv_compiled = actx.compile(grad_cv) # noqa

    def my_write_viz(step, t, t_wall, viz_state, viz_dv,
                     theta_rho, theta_Y, theta_pres,
                     ts_field_fluid, ts_field_wall, dump_number):

        if rank == 0:
            print(f"******** Writing Fluid Visualization File {dump_number}"
                  f" at step {step},"
                  f" sim time {t:1.6e} s ********")

        if use_wall:
            fluid_state = viz_state[0]
            tseed = viz_state[1]
            entropy_min = viz_state[2]
            wv = viz_state[3]
            dv = viz_dv[0]
            wdv = viz_dv[1]
        else:
            fluid_state = viz_state[0]
            tseed = viz_state[1]
            entropy_min = viz_state[2]
            dv = viz_dv
            wv = None
            wdv = None

        cv = fluid_state.cv

        # basic viz quantities, things here are difficult (or impossible) to compute
        # in post-processing
        fluid_viz_fields = [("cv", cv),
                            ("dv", dv),
                            ("dt" if constant_cfl else "cfl", ts_field_fluid)]

        if use_wall:
            wall_kappa = wdv.thermal_conductivity
            wall_temperature = wdv.temperature

            if rank == 0:
                print(f"******** Writing Wall Visualization File {dump_number}"
                      f" at step {step},"
                      f" sim time {t_wall:1.6e} s ********")

            wall_viz_fields = [
                ("wv", wv),
                ("wall_kappa", wall_kappa),
                ("wall_temperature", wall_temperature),
                ("dt" if constant_cfl else "cfl", ts_field_wall)
            ]

        # extra viz quantities, things here are often used for post-processing
        if viz_level > 0:
            mach = cv.speed / dv.speed_of_sound
            fluid_viz_ext = [("mach", mach),
                             ("velocity", cv.velocity)]
            fluid_viz_fields.extend(fluid_viz_ext)

            internal_energy_density = cv.energy - 0.5*cv.mass*np.dot(
                cv.velocity, cv.velocity)
            internal_energy = internal_energy_density/cv.mass
            enthalpy = internal_energy + dv.pressure/cv.mass

            fluid_viz_ext = [("internal_energy", internal_energy),
                             ("internal_energy_density", internal_energy_density),
                             ("enthalpy", enthalpy)]
            fluid_viz_fields.extend(fluid_viz_ext)

            # species mass fractions
            fluid_viz_fields.extend(
                ("Y_"+species_names[i], cv.species_mass_fractions[i])
                for i in range(nspecies))

            # entropy
            gamma = gas_model.eos.gamma(cv, dv.temperature)
            """
            if eos_type == 1:

                species_entropy = np.zeros(nspecies, dtype=object)
                entropy = actx.np.zeros_like(cv.mass)
                for i in range(nspecies):
                    species_entropy[i] = \
                        pyro_mech.get_species_entropies_r(dv.temperature)
                    entropy = entropy +\
                        species_entropy[i]*cv.species_mass_fractions[i]
                entropy = entropy*pyro_mech.get_specific_gas_constant(
                    cv.species_mass_fractions)
            else:
                entropy = actx.np.log(dv.pressure/(cv.mass**gamma))
                """
            entropy = actx.np.log(dv.pressure/(cv.mass**gamma))

            fluid_viz_ext = [("entropy", entropy),
                             ("entropy_min", entropy_min),
                             ("gamma", gamma)]
            fluid_viz_fields.extend(fluid_viz_ext)

            if eos_type == 1:
                temp_resid = get_temperature_update_compiled(
                    cv, dv.temperature)/dv.temperature
                production_rates = compute_production_rates(cv,
                                                            dv.temperature)
                fluid_viz_ext = [("temp_resid", temp_resid),
                                 ("tseed", tseed),
                                 ("production_rates", production_rates)]
                fluid_viz_fields.extend(fluid_viz_ext)

            # expand to include species diffusivities?
            fluid_viz_ext = [("mu", fluid_state.viscosity),
                             ("beta", fluid_state.bulk_viscosity),
                             ("kappa", fluid_state.thermal_conductivity)]
            fluid_viz_fields.extend(fluid_viz_ext)

            if transport_type > 0:
                fluid_diffusivity = fluid_state.species_diffusivity
                fluid_viz_fields.extend(
                    ("D_"+species_names[i], fluid_diffusivity[i])
                    for i in range(nspecies))

            if nparts > 1:
                fluid_viz_ext = [("rank", rank)]
                fluid_viz_fields.extend(fluid_viz_ext)

            if use_wall:
                wall_viz_ext = [("wall_kappa", wall_kappa)]
                wall_viz_fields.extend(wall_viz_ext)

                if nparts > 1:
                    wall_viz_ext = [("rank", rank)]
                    wall_viz_fields.extend(wall_viz_ext)

        # additional viz quantities, add in some non-dimensional numbers
        if viz_level > 1:
            cell_Re = (cv.mass*cv.speed*char_length_fluid /
                fluid_state.viscosity)
            cp = gas_model.eos.heat_capacity_cp(cv, fluid_state.temperature)
            alpha_heat = fluid_state.thermal_conductivity/cp/cv.mass
            nu = (4./3.*fluid_state.viscosity + fluid_state.bulk_viscosity) / \
                  fluid_state.mass_density

            cell_Pe_momentum = char_length_fluid*fluid_state.wavespeed/nu

            cell_Pe_thermal = char_length_fluid*fluid_state.wavespeed/alpha_heat

            from mirgecom.viscous import get_local_max_species_diffusivity
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    fluid_state.array_context,
                    fluid_state.species_diffusivity
                )

            cell_Pe_diffusion = char_length_fluid*fluid_state.wavespeed/d_alpha_max

            # these are useful if our transport properties
            # are not constant on the mesh
            # prandtl
            # schmidt_number
            # damkohler_number
            viz_ext = [("Re", cell_Re),
                       ("Pe_momentum", cell_Pe_momentum),
                       ("Pe_thermal", cell_Pe_thermal),
                       ("Pe_diffusion", cell_Pe_diffusion)]
            fluid_viz_fields.extend(viz_ext)
            viz_ext = [("char_length_fluid", char_length_fluid),
                       ("char_length_fluid_smooth", smoothed_char_length_fluid),
                       ("sponge_sigma", sponge_sigma)]
            fluid_viz_fields.extend(viz_ext)

            cfl_fluid_inv = char_length_fluid / (fluid_state.wavespeed)
            cfl_fluid_visc = char_length_fluid**2 / nu
            cfl_fluid_spec_diff = char_length_fluid**2 / d_alpha_max
            cfl_fluid_heat_diff = (char_length_fluid**2 / alpha_heat)

            viz_ext = [
                       ("cfl_fluid_inv", current_dt/cfl_fluid_inv),
                       ("cfl_fluid_visc", current_dt/cfl_fluid_visc),
                       ("cfl_fluid_heat_diff", current_dt/cfl_fluid_heat_diff),
                       ("cfl_fluid_spec_diff", current_dt/cfl_fluid_spec_diff)]
            fluid_viz_fields.extend(viz_ext)

            if use_species_limiter == 2:
                viz_ext = [("theta_rho", theta_rho),
                           ("theta_pressure", theta_pres)]
                fluid_viz_fields.extend(viz_ext)

                fluid_viz_fields.extend(
                    ("theta_Y_"+species_names[i], theta_Y[i])
                    for i in range(nspecies))

            if use_wall:
                cell_alpha = wall_model.thermal_diffusivity(
                    wv.mass, wall_temperature, wall_kappa)
                viz_ext = [("alpha", cell_alpha)]
                wall_viz_fields.extend(viz_ext)

            # this gives us the DOFArray indices for each element.
            discr = dcoll.discr_from_dd(dd_vol_fluid)
            nelem = discr.groups[0].nelements
            ndof = discr.groups[0].nunit_dofs

            el_indices = DOFArray(actx, data=(actx.from_numpy(np.outer(
                np.indices((nelem,)), np.ones(ndof))),))
            viz_ext = [("el_indices", el_indices)]
            fluid_viz_fields.extend(viz_ext)

            if use_wall:
                discr = dcoll.discr_from_dd(dd_vol_wall)
                nelem = discr.groups[0].nelements
                ndof = discr.groups[0].nunit_dofs
                el_indices = DOFArray(actx, data=(actx.from_numpy(np.outer(
                    np.indices((nelem,)), np.ones(ndof))),))
                viz_ext = [("el_indices", el_indices)]
                wall_viz_fields.extend(viz_ext)

            # get grad_cv to compute a numerical schlieren
            grad_fluid_cv = grad_cv_compiled(
                fluid_state=fluid_state, time=t)
            grad_rho = grad_fluid_cv.mass

            norm_grad_rho = actx.np.sqrt(np.dot(grad_rho, grad_rho))
            norm_grad_rho_max = vol_max(dd_vol_fluid, norm_grad_rho)
            norm_grad_rho_min = vol_min(dd_vol_fluid, norm_grad_rho)
            schlieren_beta = 10.

            zero = actx.np.zeros_like(cv.mass)
            ratio = actx.np.where(actx.np.greater(norm_grad_rho_max - 1.e-10,
                                                  norm_grad_rho_min),
                                  ((norm_grad_rho - norm_grad_rho_min - 1.e-10) /
                                  (norm_grad_rho_max - norm_grad_rho_min)), zero)
            schlieren = 1. - actx.np.exp(-schlieren_beta*ratio)
            viz_ext = [("schlieren", schlieren)]
            fluid_viz_fields.extend(viz_ext)

        # debbuging viz quantities, things here are used for diagnosing run issues
        if viz_level > 2:

            if use_wall:
                viz_stuff = compute_viz_fields_coupled_compiled(
                    fluid_state=fluid_state,
                    wv=wv,
                    wdv=wdv,
                    time=t)
            else:
                viz_stuff = compute_viz_fields_compiled(
                    fluid_state=fluid_state,
                    time=t)

            av_smu = viz_stuff[0]
            av_sbeta = viz_stuff[1]
            av_skappa = viz_stuff[2]
            av_sd = viz_stuff[3]
            grad_v = viz_stuff[4]
            grad_y = viz_stuff[5]
            grad_fluid_t = viz_stuff[6]
            grad_cv = viz_stuff[7]

            """
            if use_wall:
                grad_wall_t = viz_stuff[8]
            """

            viz_ext = [("smoothness_mu", av_smu),
                       ("smoothness_beta", av_sbeta),
                       ("smoothness_kappa", av_skappa),
                       ("smoothness_d", av_sd)]
            fluid_viz_fields.extend(viz_ext)

            if use_drop_order:
                smoothness = smoothness_indicator(dcoll, cv.mass, dd=dd_vol_fluid,
                                                  kappa=kappa_sc, s0=s0_sc)
                viz_ext = [("smoothness", smoothness)]
                fluid_viz_fields.extend(viz_ext)

            # write out grad_cv
            viz_ext = [("grad_rho", grad_cv.mass),
                       ("grad_e", grad_cv.energy),
                       ("grad_rhou", grad_cv.momentum[0]),
                       ("grad_rhov", grad_cv.momentum[1])]
            if dim == 3:
                viz_ext.extend([("grad_rhow", grad_cv.momentum[2])])

            viz_ext.extend(("grad_rhoY_"+species_names[i], grad_cv.species_mass[i])
                           for i in range(nspecies))
            fluid_viz_fields.extend(viz_ext)

            viz_ext = [("grad_temperature", grad_fluid_t),
                       ("grad_v_x", grad_v[0]),
                       ("grad_v_y", grad_v[1])]
            if dim == 3:
                viz_ext.extend([("grad_v_z", grad_v[2])])

            viz_ext.extend(("grad_Y_"+species_names[i], grad_y[i])
                           for i in range(nspecies))
            fluid_viz_fields.extend(viz_ext)

            """
            # write out the grid metrics
            from grudge.geometry import inverse_metric_derivative_mat
            metric = inverse_metric_derivative_mat(
                actx, dcoll, dd_vol_fluid,
                _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

            viz_ext = [("metric_x", metric[0]),
                       ("metric_y", metric[1])]
            if dim == 3:
                viz_ext.extend([("metric_z", metric[2])])

            fluid_viz_fields.extend(viz_ext)
            """

            """
            if use_wall:
                viz_ext = [("grad_temperature_wall", grad_wall_t)]
                wall_viz_fields.extend(viz_ext)
            """

            """
            elem_average = element_average_cv(cv)
            elem_minimum = element_minimum_cv(cv)
            elem_maximum = element_maximum_cv(cv)
            neighbor_min_avg_cv = neighbor_minimum_cv(elem_average)
            neighbor_min_min_cv = neighbor_minimum_cv(elem_minimum)
            neighbor_max_avg_cv = neighbor_maximum_cv(elem_average)
            neighbor_max_max_cv = neighbor_maximum_cv(elem_maximum)

            elem_average_pres = element_average(dcoll, dv.pressure)
            elem_minimum_pres = element_minimum(dcoll, dv.pressure)
            elem_maximum_pres = element_maximum(dcoll, dv.pressure)
            neighbor_min_avg_pres = _neighbor_minimum(elem_average_pres)
            neighbor_min_min_pres = _neighbor_minimum(elem_minimum_pres)
            neighbor_max_avg_pres = _neighbor_maximum(elem_average_pres)
            neighbor_max_max_pres = _neighbor_maximum(elem_maximum_pres)

            viz_ext = [("element_average", elem_average),
                       ("element_minimum", elem_minimum),
                       ("element_maximum", elem_maximum),
                       ("neighbor_min_min_pres", neighbor_min_min_pres),
                       ("neighbor_max_max_pres", neighbor_max_max_pres),
                       ("neighbor_min_avg_pres", neighbor_min_avg_pres),
                       ("neighbor_max_avg_pres", neighbor_max_avg_pres)]
            fluid_viz_fields.extend(viz_ext)
        """

        write_visfile(
            dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=dump_number, t=t,
            overwrite=True, comm=comm, vis_timer=vis_timer)

        if rank == 0:
            print("******** Done Writing Fluid Visualization File ********")

        if use_wall:
            write_visfile(
                dcoll, wall_viz_fields, wall_visualizer,
                vizname=vizname+"-wall", step=dump_number, t=t_wall,
                overwrite=True, comm=comm, vis_timer=vis_timer)

            if rank == 0:
                print("******** Done Writing Wall Visualization File ********")

    def my_write_restart(step, t, t_wall, state):
        if rank == 0:
            print(f"******** Writing Restart File at step {step}, "
                  f"sim time {t:1.6e} s ********")

        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)

        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": state.cv,
                "av_smu": state.av_smu,
                "av_sbeta": state.av_sbeta,
                "av_skappa": state.av_skappa,
                "av_sd": state.av_sd,
                "temperature_seed": state.tseed,
                "entropy_min": state.smin,
                "nspecies": nspecies,
                "t": t,
                "step": step,
                "order": order,
                "last_viz_interval": last_viz_interval,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }

            if use_wall:
                restart_data["wv"] = state.wv
                restart_data["t_wall"] = t_wall

            write_restart_file(actx, restart_data, restart_fname, comm)

        if rank == 0:
            print("******** Done Writing Restart File ********")

    def report_violators(ary, data_min, data_max):

        data = np.ravel(actx.to_numpy(ary)[0])
        nodes_x = np.ravel(actx.to_numpy(fluid_nodes)[0])
        nodes_y = np.ravel(actx.to_numpy(fluid_nodes)[1])
        if dim == 3:
            nodes_z = np.ravel(actx.to_numpy(fluid_nodes)[2])

        mask = (data < data_min) | (data > data_max)

        if np.any(mask):
            guilty_node_x = nodes_x[mask]
            guilty_node_y = nodes_y[mask]
            guilty_node_z = None
            if dim == 3:
                guilty_node_z = nodes_z[mask]
            guilty_data = data[mask]
            for i in range(len(guilty_data)):
                if dim == 2:
                    logger.info("Violation at nodal location "
                                f"({guilty_node_x[i]}, {guilty_node_y[i]}): "
                                f"data value {guilty_data[i]}")
                else:
                    logger.info("Violation at nodal location "
                                f"({guilty_node_x[i]}, {guilty_node_y[i]}, "
                                f"{guilty_node_z[i]}): "
                                f"data value {guilty_data[i]}")
                if i > 5:
                    logger.info("Violators truncated at 5")
                    break

    def spec_check(cv):
        health_error = False
        ysum = actx.np.zeros_like(cv.mass)
        spec_tol = 1e-16
        for i in range(nspecies):
            yspec = cv.species_mass_fractions[i]
            ysum = ysum + yspec
            if global_range_check(dd_vol_fluid, yspec, 0.0, 1+spec_tol):
                health_error = True
                y_min = vol_min(dd_vol_fluid, yspec)
                y_max = vol_max(dd_vol_fluid, yspec)
                y_min_loc = vol_min_loc(dd_vol_fluid, yspec)
                y_max_loc = vol_max_loc(dd_vol_fluid, yspec)
                if rank == 0:
                    logger.info("Species mass fraction range violation:\n"
                                "\tSpecified Limits "
                                f"({health_mass_frac_min=}, "
                                f"{health_mass_frac_max=})\n"
                                f"\tGlobal Range     {species_names[i]}:"
                                f"({y_min:1.3e}, {y_max:1.3e})")
                logger.info(f"{rank=}: "
                            f"Local Range      {species_names[i]}: "
                            f"({y_min_loc:1.3e}, {y_max_loc:1.3e})")
                print(f"{rank=}: "
                      f"Local Range      {species_names[i]}: "
                      f"({y_min_loc:1.3e}, {y_max_loc:1.3e})")
                report_violators(yspec, 0.0, 1.+spec_tol)

        ysum_m1 = actx.np.abs(ysum - 1.0)
        sum_tol = 1e-15
        if global_range_check(dd_vol_fluid, ysum_m1, 0., sum_tol):
            health_error = True
            local_max = actx.np.max(ysum)
            local_min = actx.np.min(ysum)
            global_min = vol_min(dd_vol_fluid, ysum)
            global_max = vol_max(dd_vol_fluid, ysum)
            global_min_loc = vol_min_loc(dd_vol_fluid, ysum)
            global_max_loc = vol_max_loc(dd_vol_fluid, ysum)
            if rank == 0:
                logger.info("Total species mass fraction range violation:\n"
                            f"{sum_tol=}), {global_min=}, {global_max=}\n"
                            f"{global_min_loc=}, {global_max_loc=}")
            logger.info(f"{rank=}: "
                        f"Local sum:      {actx.to_numpy(local_max)=},"
                        f" {actx.to_numpy(local_min)=}")
            print(f"{rank=}: {actx.to_numpy(local_max)=}, "
                  f"{actx.to_numpy(local_min)=}")
            report_violators(ysum, 1.-sum_tol, 1.+sum_tol)

        return health_error

    def my_health_check(fluid_state, wall_temperature):
        health_error = False
        cv = fluid_state.cv
        dv = fluid_state.dv

        dv_fields = ["temperature",
                     "pressure",
                     "smoothness_mu",
                     "smoothness_kappa",
                     "smoothness_d",
                     "smoothness_beta"]

        for field in dv_fields:
            field_name = field
            field_val = getattr(dv, field_name)
            if check_naninf_local(dcoll, dd_vol_fluid, field_val):
                health_error = True
                logger.info(f"{rank=}: NANs/Infs in {field_name} data.")
                print(f"{rank=}: NANs/Infs in {field_name} data.")

        if use_wall:
            if check_naninf_local(dcoll, dd_vol_wall, wall_temperature):
                health_error = True
                logger.info(f"{rank=}: NANs/Infs in wall temperature data.")
                print(f"{rank=}: NANs/Infs in wall temperature data.")

        # These range checking bits seem oblivious/impervious to NANs
        if global_range_check(dd_vol_fluid, dv.pressure,
                              health_pres_min, health_pres_max):
            health_error = True
            p_min = vol_min(dd_vol_fluid, dv.pressure)
            p_max = vol_max(dd_vol_fluid, dv.pressure)
            p_min_loc = vol_min_loc(dd_vol_fluid, dv.pressure)
            p_max_loc = vol_max_loc(dd_vol_fluid, dv.pressure)

            if rank == 0:
                logger.info("Pressure range violation:\n"
                             "\tSpecified Limits "
                            f"({health_pres_min=}, {health_pres_max=})\n"
                            f"\tGlobal Range     ({p_min:1.9e}, {p_max:1.9e})")
            logger.info(f"{rank=}: "
                        f"Local Range      ({p_min_loc:1.9e}, {p_max_loc:1.9e})")
            print(f"{rank=}: Local Pressure Range "
                  f"({p_min_loc:1.9e}, {p_max_loc:1.9e})")
            report_violators(dv.pressure, health_pres_min, health_pres_max)

        if global_range_check(dd_vol_fluid, dv.temperature,
                              health_temp_min, health_temp_max):
            health_error = True
            t_min = vol_min(dd_vol_fluid, dv.temperature)
            t_max = vol_max(dd_vol_fluid, dv.temperature)
            t_min_loc = vol_min_loc(dd_vol_fluid, dv.temperature)
            t_max_loc = vol_max_loc(dd_vol_fluid, dv.temperature)
            if rank == 0:
                logger.info("Temperature range violation:\n"
                             "\tSpecified Limits "
                            f"({health_temp_min=}, {health_temp_max=})\n"
                            f"\tGlobal Range     ({t_min:7g}, {t_max:7g})")
            logger.info(f"{rank=}: "
                        f"Local Range      ({t_min_loc:7g}, {t_max_loc:7g})")
            print(f"{rank=}: Local Temperature Range "
                  f"({t_min_loc:1.9e}, {t_max_loc:1.9e})")
            report_violators(dv.temperature, health_temp_min, health_temp_max)

        if use_wall:
            if global_range_check(dd_vol_wall, wall_temperature,
                                  health_temp_min, health_temp_max):
                health_error = True
                t_min = vol_min(dd_vol_wall, wall_temperature)
                t_max = vol_max(dd_vol_wall, wall_temperature)
                logger.info(
                    f"{rank=}:"
                    "Wall temperature range violation: "
                    f"Simulation Range ({t_min=}, {t_max=}) "
                    f"Specified Limits ({health_temp_min=}, {health_temp_max=})")
                t_min_loc = vol_min(dd_vol_wall, wall_temperature)
                t_max_loc = vol_max(dd_vol_wall, wall_temperature)
                print(f"{rank=}: Local Wall Temperature Range "
                      f"({t_min_loc:1.9e}, {t_max_loc:1.9e})")

        for i in range(nspecies):
            if global_range_check(dd_vol_fluid, cv.species_mass_fractions[i],
                                  health_mass_frac_min, health_mass_frac_max):
                health_error = True
                y_min = vol_min(dd_vol_fluid, cv.species_mass_fractions[i])
                y_max = vol_max(dd_vol_fluid, cv.species_mass_fractions[i])
                y_min_loc = vol_min_loc(dd_vol_fluid, cv.species_mass_fractions[i])
                y_max_loc = vol_max_loc(dd_vol_fluid, cv.species_mass_fractions[i])
                if rank == 0:
                    logger.info("Species mass fraction range violation:\n"
                                 "\tSpecified Limits "
                                f"({health_mass_frac_min=}, "
                                f"{health_mass_frac_max=})\n"
                                f"\tGlobal Range     {species_names[i]}:"
                                f"({y_min:1.3e}, {y_max:1.3e})")
                logger.info(f"{rank=}: "
                            f"Local Range      {species_names[i]}: "
                            f"({y_min_loc:1.3e}, {y_max_loc:1.3e})")
                print(f"{rank=}: "
                      f"Local Range      {species_names[i]}: "
                      f"({y_min_loc:1.3e}, {y_max_loc:1.3e})")
                report_violators(cv.species_mass_fractions[i],
                                 health_mass_frac_min, health_mass_frac_max)

        if eos_type == 1:
            # check the temperature convergence
            # a single call to get_temperature_update is like taking an additional
            # Newton iteration and gives us a residual
            temp_resid = get_temperature_update_compiled(
                cv, dv.temperature)/dv.temperature
            temp_err = vol_max(dd_vol_fluid, temp_resid)
            temp_err_loc = vol_max_loc(dd_vol_fluid, temp_resid)
            if temp_err > pyro_temp_tol:
                health_error = True
                logger.info(f"{rank=}:"
                             "Temperature is not converged "
                            f"{temp_err=} > {pyro_temp_tol}.")
                logger.info(f"{rank=}: Temperature is not converged."
                            f" Local Residual {temp_err_loc:7g} > {pyro_temp_tol}")
                print(f"{rank=}: Local Temperature Residual ({temp_err_loc:1.9e})")

        return health_error

    def my_get_viscous_timestep(dcoll, fluid_state):

        nu = 0
        d_alpha_max = 0

        if fluid_state.is_viscous:
            from mirgecom.viscous import get_local_max_species_diffusivity
            #nu = fluid_state.viscosity/fluid_state.mass_density
            nu = ((4./3.*fluid_state.viscosity + fluid_state.bulk_viscosity) /
                  fluid_state.mass_density)
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    fluid_state.array_context,
                    fluid_state.species_diffusivity
                )

        return (
            char_length_fluid / (fluid_state.wavespeed
            + ((nu + d_alpha_max) / char_length_fluid))
        )

    def _my_get_timestep(
            dcoll, fluid_state, t, dt, cfl, t_final, constant_cfl=False,
            fluid_dd=DD_VOLUME_ALL):

        mydt = dt
        if constant_cfl:
            from grudge.op import nodal_min
            ts_field = cfl*my_get_viscous_timestep(
                dcoll=dcoll, fluid_state=fluid_state)
            mydt = fluid_state.array_context.to_numpy(nodal_min(
                    dcoll, fluid_dd, ts_field, initial=np.inf))[()]
        else:
            from grudge.op import nodal_max
            ts_field = mydt/my_get_viscous_timestep(
                dcoll=dcoll, fluid_state=fluid_state)
            cfl = fluid_state.array_context.to_numpy(nodal_max(
                    dcoll, fluid_dd, ts_field, initial=0.))[()]

        return ts_field, cfl, mydt

    #my_get_timestep = actx.compile(_my_get_timestep)
    my_get_timestep = _my_get_timestep

    if use_wall:
        def wall_timestep(dcoll, wv, wall_kappa, wall_temperature):
            return (
                char_length_wall*char_length_wall
                / (
                    wall_time_scale
                    * actx.np.maximum(
                        wall_model.thermal_diffusivity(
                            wv.mass, wall_temperature, wall_kappa),
                        wall_model.oxygen_diffusivity)))

        def _my_get_timestep_wall(
                dcoll, wv, wall_kappa, wall_temperature, t, dt, cfl, t_final,
                constant_cfl=False, wall_dd=DD_VOLUME_ALL):

            actx = wall_kappa.array_context
            mydt = dt
            if constant_cfl:
                from grudge.op import nodal_min
                ts_field = cfl*wall_timestep(
                    dcoll=dcoll, wv=wv, wall_kappa=wall_kappa,
                    wall_temperature=wall_temperature)
                mydt = actx.to_numpy(
                    nodal_min(
                        dcoll, wall_dd, ts_field, initial=np.inf))[()]
            else:
                from grudge.op import nodal_max
                ts_field = mydt/wall_timestep(
                    dcoll=dcoll, wv=wv, wall_kappa=wall_kappa,
                    wall_temperature=wall_temperature)
                cfl = actx.to_numpy(
                    nodal_max(
                        dcoll, wall_dd, ts_field, initial=0.))[()]

            return ts_field, cfl, mydt

    #my_get_timestep = actx.compile(_my_get_timestep)
    if use_wall:
        my_get_timestep_wall = _my_get_timestep_wall

    def _check_time(time, dt, interval, interval_type):
        toler = 1.e-6
        status = False

        dumps_so_far = math.floor((time-t_start)/interval)

        # dump if we just passed a dump interval
        if interval_type == 2:
            time_till_next = (dumps_so_far + 1)*interval - time
            steps_till_next = math.floor(time_till_next/dt)

            # reduce the timestep going into a dump to avoid a big variation in dt
            if steps_till_next < 5:
                dt_new = dt
                extra_time = time_till_next - steps_till_next*dt
                #if actx.np.abs(extra_time/dt) > toler:
                if abs(extra_time/dt) > toler:
                    dt_new = time_till_next/(steps_till_next + 1)

                if steps_till_next < 1:
                    dt_new = time_till_next

                dt = dt_new

            time_from_last = time - t_start - (dumps_so_far)*interval
            if abs(time_from_last/dt) < toler:
                status = True
        else:
            time_from_last = time - t_start - (dumps_so_far)*interval
            if time_from_last < dt:
                status = True

        return status, dt, dumps_so_far + last_viz_interval

    #check_time = _check_time
    #from pytato.array import set_traceback_tag_enabled
    #set_traceback_tag_enabled(True)

    def my_pre_step(step, t, dt, state):

        # I don't think this should be needed, but shouldn't hurt anything
        #state = force_evaluation(actx, state)
        if logmgr:
            logmgr.tick_before()

        # This check reports when species mass fractions (Y) violate
        # the constraints of being in the range [0, 1] +/- 1e-16,
        # and sum(Y) = 1 +/- 1e-15. If the ranges are violated,
        # the snippet below will use *hammer_species* to force Y
        # back into the expected range.
        if nspecies > 0 and use_hammer_species is True:
            if check_step(step=step, interval=nspeccheck):
                spec_errors = global_reduce(
                    spec_check(state[0]), op="lor")
                if spec_errors:
                    if rank == 0:
                        logger.info("Solution failed species check.")
                    logger.info(f"{rank=}: Solution failed species check.")
                    print(f"{rank=}: Solution failed species check - limiting more.")
                    comm.Barrier()  # make msg before any rank raises
                    # Don't raise, just hammer Y back to [0, 1], sum(Y)=1
                    # raise MyRuntimeError("Failed simulation species check.")
                    state[0] = hammer_species(state[0])

        stepper_state = make_stepper_state_obj(state)

        if check_step(step=step, interval=ngarbage):
            with gc_timer:
                from warnings import warn
                warn("Running gc.collect() to work around memory growth issue "
                     "https://github.com/illinois-ceesd/mirgecom/issues/839")
                import gc
                gc.collect()

        # Filter *first* because this will be most straightfwd to
        # understand and move. For this to work, this routine
        # must pass back the filtered CV in the state.
        if check_step(step=step, interval=soln_nfilter):
            #cv, tseed, av_smu, av_sbeta, av_skappa, wv = state
            cv = filter_cv_compiled(stepper_state.cv)
            stepper_state = stepper_state.replace(cv=cv)

        if use_drop_order:
            # this limits the solution at the shock front,
            smoothness = smoothness_indicator(dcoll, stepper_state.cv.mass,
                                              dd=dd_vol_fluid,
                                              kappa=kappa_sc, s0=s0_sc)
            #smoothness = actx.np.zeros_like(stepper_state.cv.mass) + 1.0
            cv = drop_order_cv(stepper_state.cv, smoothness, drop_order_strength)
            stepper_state = stepper_state.replace(cv=cv)

        fluid_state = create_fluid_state(cv=stepper_state.cv,
                                         temperature_seed=stepper_state.tseed,
                                         smoothness_mu=stepper_state.av_smu,
                                         smoothness_beta=stepper_state.av_sbeta,
                                         smoothness_kappa=stepper_state.av_skappa,
                                         smoothness_d=stepper_state.av_sd,
                                         entropy_min=stepper_state.smin)

        print_stuff = False
        index = 6000
        if print_stuff is True:
            # initial state
            np.set_printoptions(threshold=sys.maxsize, precision=16)
            print("my_pre_step before limit")
            data_rho = actx.to_numpy(stepper_state.cv.mass)
            print(f"mass \n {data_rho[0][index]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(stepper_state.cv.species_mass)
                print(f"rhoY_[{i}] {data[i][0][index]} ")
                print(f"check Y_[{i}] {data[i][0][index]/data_rho[0][index]} ")
            for i in range(0, nspecies):
                data = actx.to_numpy(stepper_state.cv.species_mass_fractions)
                print(f"Y_[{i}] {data[i][0][index]} ")
            for i in range(0, nspecies):
                print(f"check Y_[{i}] {data[i][0][index]} ")

            print("my_pre_step after limit")
            data = actx.to_numpy(theta_rho)
            print(f"theta_rho {data[0][index]} ")
            data = actx.to_numpy(fluid_state.cv.mass)
            print(f"mass {data[0][index]} ")
            for i in range(0, nspecies):
                data = actx.to_numpy(fluid_state.cv.species_mass)
                print(f"rhoY_[{i}] {data[i][0][index]} ")
            for i in range(0, nspecies):
                data = actx.to_numpy(fluid_state.cv.species_mass_fractions)
                print(f"Y_[{i}] {data[i][0][index]} ")

        if use_wall:
            wdv = create_wall_dependent_vars_compiled(stepper_state.wv)
        cv = fluid_state.cv  # reset cv to limited version

        # update temperature seed and our limiter bounds for entropy
        tseed = fluid_state.temperature

        ones = 1. + actx.np.zeros_like(cv.mass)
        if constant_smin is True:
            smin_i = ones*limiter_smin
        else:
            gamma = gas_model.eos.gamma(cv, fluid_state.temperature)
            smin = actx.np.log(fluid_state.pressure/fluid_state.mass_density**gamma)

            from mirgecom.simutil import (
                inverse_element_connectivity,
                compute_vertex_averages,
                scatter_vertex_values_to_dofarray,
            )

            iconn = inverse_element_connectivity(
                mesh=dcoll.discr_from_dd(dd_vol_fluid).mesh)

            def nodal_average(field):
                # Compute cell averages of the state
                actx = field.array_context
                nodal_avg = compute_vertex_averages(actx, field, iconn)
                return scatter_vertex_values_to_dofarray(
                    actx, field, iconn, nodal_avg)

            element_average_smin = element_average(dcoll, dd_vol_fluid, smin)
            smin_i = nodal_average(element_average_smin)

            # fixed offset
            #smin_i = op.elementwise_min(dcoll, dd_vol_fluid, smin)
            # fixed offset
            smin_i = ones*(smin_i - 0.05)

        # This re-creation of the state resets *tseed* to current temp and forces the
        # limited cv into state
        stepper_state = stepper_state.replace(cv=cv, tseed=tseed, smin=smin_i)
        fluid_state = replace_fluid_state(fluid_state, gas_model, entropy_min=smin_i)

        try:
            # disable non-constant dt timestepping for now
            # re-enable when we're ready

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)
            next_dump_number = step

            dv = None
            ts_field_fluid = None
            ts_field_wall = None
            cfl_fluid = 0.
            cfl_wall = cfl_fluid
            if any([do_viz, do_restart, do_health, do_status]):
                wv = None
                if not force_eval:
                    fluid_state = force_evaluation(actx, fluid_state)
                    #state = force_evaluation(actx, state)
                    if use_wall:
                        wv = force_evaluation(actx, stepper_state.wv)
                elif use_wall:
                    wv = stepper_state.wv  # pylint: disable=no-member

                dv = fluid_state.dv

                ts_field_fluid, cfl_fluid, dt_fluid = my_get_timestep(
                    dcoll=dcoll, fluid_state=fluid_state,
                    t=t, dt=dt, cfl=current_cfl, t_final=t_final,
                    constant_cfl=constant_cfl, fluid_dd=dd_vol_fluid)

                if use_wall:
                    ts_field_wall, cfl_wall, dt_wall = my_get_timestep_wall(
                        dcoll=dcoll, wv=wv, wall_kappa=wdv.thermal_conductivity,
                        wall_temperature=wdv.temperature, t=t, dt=dt,
                        cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl,
                        wall_dd=dd_vol_wall)

            """
            # adjust time for constant cfl, use the smallest timescale
            dt_const_cfl = 100.
            if constant_cfl:
                dt_const_cfl = np.minimum(dt_fluid, dt_wall)

            # adjust time to hit the final requested time
            t_remaining = max(0, t_final - t)

            if viz_interval_type == 0:
                dt = np.minimum(t_remaining, current_dt)
            else:
                dt = np.minimum(t_remaining, dt_const_cfl)

            # update our I/O quantities
            cfl_fluid = dt*cfl_fluid/dt_fluid
            cfl_wall = dt*cfl_wall/dt_wall
            ts_field_fluid = dt*ts_field_fluid/dt_fluid
            ts_field_wall = dt*ts_field_wall/dt_wall

            if viz_interval_type == 1:
                do_viz, dt, next_dump_number = check_time(
                    time=t, dt=dt, interval=t_viz_interval,
                    interval_type=viz_interval_type)
            elif viz_interval_type == 2:
                dt_sav = dt
                do_viz, dt, next_dump_number = check_time(
                    time=t, dt=dt, interval=t_viz_interval,
                    interval_type=viz_interval_type)

                # adjust cfl by dt
                cfl_fluid = dt*cfl_fluid/dt_sav
                cfl_wall = dt*cfl_wall/dt_sav
            else:
                do_viz = check_step(step=step, interval=nviz)
                next_dump_number = step
            """

            t_wall = t_wall_start + (step - first_step)*dt*wall_time_scale
            my_write_status_lite(step=step, t=t, t_wall=t_wall)

            # these status updates require global reductions on state data
            if do_status:
                my_write_status_fluid(fluid_state, dt=dt, cfl_fluid=cfl_fluid)
                if use_wall:
                    my_write_status_wall(wall_temperature=wdv.temperature,
                                         dt=dt*wall_time_scale, cfl_wall=cfl_wall)

            if do_health:
                wall_temptr = wdv.temperature if use_wall else None
                health_errors = global_reduce(
                    my_health_check(fluid_state, wall_temperature=wall_temptr),
                    op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Solution failed health check.")
                    logger.info(f"{rank=}: Solution failed health check. Logger")
                    print(f"{rank=}:Solution failed health check. Print.")
                    comm.Barrier()  # make msg before any rank raises
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, t_wall=t_wall, state=stepper_state)

            if do_viz:
                # we can't get the limited viz data back from create_fluid_state
                # so call the limiter directly first, basically doing the limiting
                # twice
                theta_rho = actx.np.zeros_like(stepper_state.cv.mass)
                theta_Y = actx.np.zeros_like(stepper_state.cv.mass)
                theta_pres = actx.np.zeros_like(stepper_state.cv.mass)
                if viz_level >= 2 and use_species_limiter == 2:
                    cv_lim, theta_rho, theta_Y, theta_pres = \
                        limit_fluid_state_lv(
                            dcoll, cv=state[0], gas_model=gas_model,
                            temperature_seed=stepper_state.tseed,
                            entropy_min=stepper_state.smin,
                            dd=dd_vol_fluid, viz_theta=True)

                # pack things up
                if use_wall:
                    viz_state = make_obj_array([fluid_state, tseed, smin_i, wv])
                    viz_dv = make_obj_array([dv, wdv])
                else:
                    viz_state = make_obj_array([fluid_state, tseed, smin_i])
                    viz_dv = dv

                my_write_viz(
                    step=step, t=t, t_wall=t_wall,
                    viz_state=viz_state, viz_dv=viz_dv,
                    ts_field_fluid=ts_field_fluid,
                    ts_field_wall=ts_field_wall,
                    theta_rho=theta_rho,
                    theta_Y=theta_Y,
                    theta_pres=theta_pres,
                    dump_number=next_dump_number)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")

            # we can't get the limited viz data back from create_fluid_state
            # so call the limiter directly first, basically doing the limiting
            # twice
            theta_rho = actx.np.zeros_like(stepper_state.cv.mass)
            theta_Y = actx.np.zeros_like(stepper_state.cv.mass)
            theta_pres = actx.np.zeros_like(stepper_state.cv.mass)
            if viz_level >= 2 and use_species_limiter == 2:
                cv_lim, theta_rho, theta_Y, theta_pres = \
                    limit_fluid_state_lv(
                        dcoll, cv=state[0], gas_model=gas_model,
                        temperature_seed=stepper_state.tseed,
                        entropy_min=stepper_state.smin,
                        dd=dd_vol_fluid, viz_theta=True)

            if viz_interval_type == 0:
                dump_number = step
            else:
                dump_number = (math.floor((t-t_start)/t_viz_interval) +
                    last_viz_interval)

            # pack things up
            if use_wall:
                viz_state = make_obj_array([fluid_state, tseed, smin_i, wv])
                viz_dv = make_obj_array([dv, wdv])
            else:
                viz_state = make_obj_array([fluid_state, tseed, smin_i])
                viz_dv = dv

            my_write_viz(
                step=step, t=t, t_wall=t_wall,
                viz_state=viz_state, viz_dv=viz_dv,
                ts_field_fluid=ts_field_fluid,
                ts_field_wall=ts_field_wall,
                theta_rho=theta_rho,
                theta_Y=theta_Y,
                theta_pres=theta_pres,
                dump_number=dump_number)

            my_write_restart(step=step, t=t, t_wall=t_wall, state=stepper_state)
            comm.Barrier()  # cross and dot t's and i's (sync point)
            raise

        if step == first_profiling_step:
            MPI.Pcontrol(2)
            MPI.Pcontrol(1)

        return stepper_state.get_obj_array(), dt

    def my_post_step(step, t, dt, state):

        if step == last_profiling_step:
            MPI.Pcontrol(0)

        if step == first_step+2:
            with gc_timer:
                import gc
                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                logger.info("Freezing GC objects to reduce overhead of "
                            "future GC collections")
                gc.freeze()

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

        return state, dt

    from arraycontext import outer
    from grudge.trace_pair import interior_trace_pairs, tracepair_with_discr_tag
    from meshmode.discretization.connection import FACE_RESTR_ALL
    from mirgecom.flux import num_flux_central

    def my_derivative_function(dcoll, field, field_bounds, dd_vol,
                               bnd_cond, comm_tag):

        dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
        dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

        interp_to_surf_quad = partial(
            tracepair_with_discr_tag, dcoll, quadrature_tag)

        def interior_flux(field_tpair):
            dd_trace_quad = field_tpair.dd.with_discr_tag(quadrature_tag)
            #normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
            normal_quad = normal_vector(actx, dcoll, dd_trace_quad)
            bnd_tpair_quad = interp_to_surf_quad(field_tpair)
            flux_int = outer(
                num_flux_central(bnd_tpair_quad.int, bnd_tpair_quad.ext),
                normal_quad)

            return op.project(dcoll, dd_trace_quad, dd_allfaces_quad, flux_int)

        def boundary_flux(bdtag, bdry):
            if isinstance(bdtag, DOFDesc):
                bdtag = bdtag.domain_tag
            dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
            normal_quad = normal_vector(actx, dcoll, dd_bdry_quad)
            int_soln_quad = op.project(dcoll, dd_vol, dd_bdry_quad, field)

            # MJA, not sure about this
            if bnd_cond == "symmetry" and bdtag == "symmetry":
                ext_soln_quad = 0.0*int_soln_quad
            else:
                ext_soln_quad = 1.0*int_soln_quad

            bnd_tpair = TracePair(bdtag, interior=int_soln_quad,
                                  exterior=ext_soln_quad)
            flux_bnd = outer(
                num_flux_central(bnd_tpair.int, bnd_tpair.ext), normal_quad)

            return op.project(dcoll, dd_bdry_quad, dd_allfaces_quad, flux_bnd)

        field_quad = op.project(dcoll, dd_vol, dd_vol_quad, field)

        return -1.0*op.inverse_mass(
            dcoll, dd_vol_quad,
            op.weak_local_grad(dcoll, dd_vol_quad, field_quad)
            -  # noqa: W504
            op.face_mass(
                dcoll, dd_allfaces_quad,
                sum(
                    interior_flux(u_tpair) for u_tpair in interior_trace_pairs(
                        dcoll, field, volume_dd=dd_vol, comm_tag=comm_tag))
                + sum(
                     boundary_flux(bdtag, bdry)
                     for bdtag, bdry in field_bounds.items())
            )
        )

    def unfiltered_rhs(t, state):


        print("Start of unfiltered_rhs")
        stepper_state = make_stepper_state_obj(state)
        cv = stepper_state.cv
        tseed = stepper_state.tseed
        av_smu = stepper_state.av_smu
        av_sbeta = stepper_state.av_sbeta
        av_skappa = stepper_state.av_skappa
        av_sd = stepper_state.av_sd
        smin = stepper_state.smin

        print_stuff = False
        index = 6000
        if print_stuff is True:
            # initial state
            np.set_printoptions(threshold=sys.maxsize, precision=16)
            print("start of my_rhs")

            data_rho = actx.to_numpy(cv.mass)
            print(f"mass {data_rho[0][index]}")
            for i in range(0, nspecies):
                data_Y = actx.to_numpy(cv.species_mass_fractions)
                print(f"Y_[{i}] {data_Y[i][0][index]} ")

            for i in range(0, nspecies):
                data_rhoY = actx.to_numpy(cv.species_mass)
                print(f"rhoY_[{i}] {data_rhoY[i][0][index]} ")

            sum_Y = [0., 0., 0., 0.]
            sum_rhoY = [0., 0., 0., 0.]

            for i in range(0, nspecies):
                sum_Y = sum_Y + data_Y[i][0][index]
                sum_rhoY = sum_rhoY + data_rhoY[i][0][index]
            print(f"diff sum_Y = {1. - sum_Y}")
            print(f"diff sum_rhoY = {data_rho[0][index] - sum_rhoY}")

        # don't really want to do this twice
        if use_drop_order:
            smoothness = smoothness_indicator(dcoll, cv.mass, dd=dd_vol_fluid,
                                              kappa=kappa_sc, s0=s0_sc)
            #smoothness = actx.np.zeros_like(cv.mass) + 1.0
            cv = _drop_order_cv(cv, smoothness, drop_order_strength)

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed,
                                       smoothness_mu=av_smu,
                                       smoothness_beta=av_sbeta,
                                       smoothness_kappa=av_skappa,
                                       smoothness_d=av_sd,
                                       entropy_min=smin,
                                       limiter_func=limiter_func,
                                       limiter_dd=dd_vol_fluid)

        cv = fluid_state.cv  # reset cv to the limited version

        if print_stuff is True:
            # initial state
            np.set_printoptions(threshold=sys.maxsize, precision=16)
            print("start of my_rhs after make_fluid_state")

            data_rho = actx.to_numpy(cv.mass)
            print(f"mass {data_rho[0][index]}")
            for i in range(0, nspecies):
                data_Y = actx.to_numpy(cv.species_mass_fractions)
                print(f"Y_[{i}] {data_Y[i][0][index]} ")

            for i in range(0, nspecies):
                data_rhoY = actx.to_numpy(cv.species_mass)
                print(f"rhoY_[{i}] {data_rhoY[i][0][index]} ")

            sum_Y = [0., 0., 0., 0.]
            sum_rhoY = [0., 0., 0., 0.]

            for i in range(0, nspecies):
                sum_Y = sum_Y + data_Y[i][0][index]
                sum_rhoY = sum_rhoY + data_rhoY[i][0][index]
            print(f"diff sum_Y = {1. - sum_Y}")
            print(f"diff sum_rhoY = {data_rho[0][index] - sum_rhoY}")

        print("Before operator_fluid_states_quad")
        # update wall model
        if use_wall:
            wv = stepper_state.wv
            wdv = wall_model.dependent_vars(wv)

            fluid_operator_states_quad = make_coupled_operator_fluid_states(
                dcoll, fluid_state, gas_model, uncoupled_fluid_boundaries,
                dd_vol_fluid, dd_vol_wall, quadrature_tag=quadrature_tag,
                limiter_func=limiter_func, entropy_min=smin)

            grad_fluid_cv, grad_fluid_t, grad_wall_t = coupled_grad_operator(
                dcoll,
                gas_model,
                dd_vol_fluid, dd_vol_wall,
                uncoupled_fluid_boundaries,
                uncoupled_wall_boundaries,
                fluid_state, wdv.thermal_conductivity, wdv.temperature,
                time=t,
                interface_noslip=noslip,
                quadrature_tag=quadrature_tag,
                fluid_operator_states_quad=fluid_operator_states_quad,
                entropy_min=smin,
                limiter_func=limiter_func)

        else:
            # Get the operator fluid states
            fluid_operator_states_quad = make_operator_fluid_states(
                dcoll, fluid_state, gas_model, uncoupled_fluid_boundaries,
                quadrature_tag, dd=dd_vol_fluid, limiter_func=limiter_func,
                entropy_min=smin)

            grad_fluid_cv = grad_cv_operator(
                dcoll, gas_model, uncoupled_fluid_boundaries, fluid_state,
                dd=dd_vol_fluid, operator_states_quad=fluid_operator_states_quad,
                time=t, quadrature_tag=quadrature_tag, limiter_func=limiter_func,
                entropy_min=smin)

            grad_fluid_t = fluid_grad_t_operator(
                dcoll=dcoll, gas_model=gas_model,
                boundaries=uncoupled_fluid_boundaries, state=fluid_state,
                dd=dd_vol_fluid, operator_states_quad=fluid_operator_states_quad,
                time=t, quadrature_tag=quadrature_tag, limiter_func=limiter_func,
                entropy_min=smin)
        print("After operator_fluid_states_quad")

        smoothness_mu = actx.np.zeros_like(cv.mass)
        smoothness_beta = actx.np.zeros_like(cv.mass)
        smoothness_kappa = actx.np.zeros_like(cv.mass)
        smoothness_d = actx.np.zeros_like(cv.mass)
        if use_av == 1:
            smoothness_mu = compute_smoothness(
                cv=cv, dv=fluid_state.dv, grad_cv=grad_fluid_cv)
        elif use_av == 2:
            [smoothness_mu, smoothness_beta, smoothness_kappa] = \
                compute_smoothness_mbk(cv=cv, dv=fluid_state.dv,
                                       grad_cv=grad_fluid_cv,
                                       grad_t=grad_fluid_t)
        elif use_av == 3:
            [smoothness_mu, smoothness_beta, smoothness_kappa, smoothness_d] = \
                compute_smoothness_mbkd(cv=cv, dv=fluid_state.dv,
                                       grad_cv=grad_fluid_cv,
                                       grad_t=grad_fluid_t)

        tseed_rhs = actx.np.zeros_like(fluid_state.temperature)
        smin_rhs = actx.np.zeros_like(fluid_state.cv.mass)

        print("Before ns_operator")
        # have all the gradients and states, compute the rhs sources
        ns_operator = partial(
            general_ns_operator,
            inviscid_numerical_flux_func=inviscid_numerical_flux_func,
            viscous_numerical_flux_func=viscous_numerical_flux_func,
            use_esdg=use_esdg)


        wall_rhs = None
        if use_wall:
            fluid_rhs, wall_energy_rhs = coupled_ns_heat_operator(
                dcoll=dcoll,
                gas_model=gas_model,
                fluid_dd=dd_vol_fluid,
                wall_dd=dd_vol_wall,
                fluid_boundaries=uncoupled_fluid_boundaries,
                wall_boundaries=uncoupled_wall_boundaries,
                fluid_state=fluid_state,
                wall_kappa=wdv.thermal_conductivity,
                wall_temperature=wdv.temperature,
                fluid_grad_cv=grad_fluid_cv,
                fluid_grad_temperature=grad_fluid_t,
                wall_grad_temperature=grad_wall_t,
                time=t,
                interface_noslip=noslip,
                wall_penalty_amount=wall_penalty_amount,
                quadrature_tag=quadrature_tag,
                fluid_operator_states_quad=fluid_operator_states_quad,
                limiter_func=limiter_func,
                entropy_min=smin,
                ns_operator=ns_operator,
                axisymmetric=use_axisymmetric,
                fluid_nodes=fluid_nodes,
                wall_nodes=wall_nodes)

        else:
            fluid_rhs = ns_operator(
                dcoll=dcoll,
                gas_model=gas_model,
                dd=dd_vol_fluid,
                operator_states_quad=fluid_operator_states_quad,
                grad_cv=grad_fluid_cv,
                grad_t=grad_fluid_t,
                boundaries=uncoupled_fluid_boundaries,
                state=fluid_state,
                time=t,
                quadrature_tag=quadrature_tag,
                comm_tag=_FluidOperatorCommTag)

            if use_axisymmetric:
                fluid_rhs = fluid_rhs + \
                    axisym_source_fluid(dcoll=dcoll, fluid_state=fluid_state,
                                        fluid_nodes=fluid_nodes,
                                        gas_model=gas_model,
                                        quadrature_tag=quadrature_tag,
                                        dd_vol_fluid=dd_vol_fluid,
                                        boundaries=uncoupled_fluid_boundaries,
                                        grad_cv=grad_fluid_cv, grad_t=grad_fluid_t)

        print("After ns_operator")
        if print_stuff is True:
            # initial state
            np.set_printoptions(threshold=sys.maxsize, precision=16)
            print("before updating y_rhs")

            data_rho = actx.to_numpy(fluid_rhs.mass)
            print(f"rhs_mass {data_rho[0][index]}")
            data_rhoY = actx.to_numpy(fluid_rhs.species_mass)
            for i in range(0, nspecies):
                print(f"rhoY_[{i}] {data_rhoY[i][0][index]} ")

            sum_Y = [0., 0., 0., 0.]
            sum_rhoY = [0., 0., 0., 0.]

            data_Y = actx.to_numpy(fluid_rhs.species_mass_fractions)
            for i in range(0, nspecies):
                sum_rhoY = sum_rhoY + data_rhoY[i][0][index]
            print(f"diff sum_rhoY_rhs = {data_rho[0][index] - sum_rhoY}")

        #print(f"{actx.to_numpy(fluid_rhs.species_mass)=}")
        new_species_mass = actx.np.zeros_like(fluid_rhs.species_mass)
        #new_species_mass[0] = 0.39*fluid_rhs.mass
        #new_species_mass[1] = 0.35*fluid_rhs.mass
        #new_species_mass[2] = 0.26*fluid_rhs.mass
        new_species_mass[0] = cv.species_mass_fractions[0]*fluid_rhs.mass
        new_species_mass[1] = cv.species_mass_fractions[1]*fluid_rhs.mass
        new_species_mass[2] = cv.species_mass_fractions[2]*fluid_rhs.mass
        #fluid_rhs = fluid_rhs.replace(species_mass=new_species_mass)

        # reset the rho update to be the sum of the species mass fractions
        if eos_type == 1 or use_species_limiter > 0:
            new_mass_rhs = 0.
            for i in range(0, nspecies):
                new_mass_rhs = new_mass_rhs + fluid_rhs.species_mass[i]
            fluid_rhs = fluid_rhs.replace(mass=new_mass_rhs)

        if print_stuff is True:
            # initial state
            np.set_printoptions(threshold=sys.maxsize, precision=16)
            print("after updating y_rhs")

            data = actx.to_numpy(fluid_rhs.mass)
            print(f"rhs_mass {data[0][index]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(fluid_rhs.species_mass)
                print(f"rhoY_[{i}] {data[i][0][index]} ")

        #print(f"{actx.to_numpy(fluid_rhs.species_mass)=}")
        #print("After zero'ing species mass contribution")

        if use_combustion is True:
            fluid_rhs = fluid_rhs + \
                eos.get_species_source_terms(cv, temperature=fluid_state.temperature)

        if use_injection_source is True:
            fluid_rhs = fluid_rhs + \
                injection_source(x_vec=fluid_nodes, cv=cv,
                                 eos=gas_model.eos, time=t)

        if use_injection_source_comb is True:
            fluid_rhs = fluid_rhs + \
                injection_source_comb(x_vec=fluid_nodes, cv=cv,
                                      eos=gas_model.eos, time=t)

        if use_injection_source_3d is True:
            fluid_rhs = fluid_rhs + \
                injection_source_3d(x_vec=fluid_nodes, cv=cv,
                                    eos=gas_model.eos, time=t)

        if use_ignition > 0:
            fluid_rhs = fluid_rhs + \
                ignition_source(x_vec=fluid_nodes, state=fluid_state,
                                eos=gas_model.eos, time=t)/current_dt

        av_smu_rhs = actx.np.zeros_like(cv.mass)
        av_sbeta_rhs = actx.np.zeros_like(cv.mass)
        av_skappa_rhs = actx.np.zeros_like(cv.mass)
        av_sd_rhs = actx.np.zeros_like(cv.mass)
        # work good for shock 1d

        tau = current_dt/smoothness_tau
        epsilon_diff = smoothness_alpha*smoothed_char_length_fluid**2/current_dt

        if use_av > 0:
            # regular boundaries for smoothness mu
            smooth_neumann = NeumannDiffusionBoundary(0)
            fluid_av_boundaries = {}
            for bnd_name in bndry_config:
                if bndry_config[bnd_name] != "none":
                    fluid_av_boundaries[bndry_elements[bnd_name]] = smooth_neumann

            if use_wall:
                from grudge.discretization import filter_part_boundaries
                fluid_av_boundaries.update({
                     dd_bdry.domain_tag: NeumannDiffusionBoundary(0)
                     for dd_bdry in filter_part_boundaries(
                         dcoll, volume_dd=dd_vol_fluid,
                         neighbor_volume_dd=dd_vol_wall)})

            # av mu
            av_smu_rhs = (
                diffusion_operator(
                    dcoll, epsilon_diff, fluid_av_boundaries, av_smu,
                    quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
                    comm_tag=_MuDiffFluidCommTag
                ) + 1/tau * (smoothness_mu - av_smu)
            )

            if use_av >= 2:
                av_sbeta_rhs = (
                    diffusion_operator(
                        dcoll, epsilon_diff, fluid_av_boundaries, av_sbeta,
                        quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
                        comm_tag=_BetaDiffFluidCommTag
                    ) + 1/tau * (smoothness_beta - av_sbeta)
                )

                av_skappa_rhs = (
                    diffusion_operator(
                        dcoll, epsilon_diff, fluid_av_boundaries, av_skappa,
                        quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
                        comm_tag=_KappaDiffFluidCommTag
                    ) + 1/tau * (smoothness_kappa - av_skappa)
                )

            if use_av == 3:
                av_sd_rhs = (
                    diffusion_operator(
                        dcoll, epsilon_diff, fluid_av_boundaries, av_sd,
                        quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
                        comm_tag=_DDiffFluidCommTag
                    ) + 1/tau * (smoothness_d - av_sd)
                )

        if use_sponge is True:
            sponge_cv = cv
            if use_time_dependent_sponge is True:
                # as long as these pieces only operate on a non-overlapping subset
                # of the domain, we don't need to call make_fluid_state
                # in between each additive call to recompute temperature/pressure
                sponge_cv = bulk_init.add_inlet(cv=sponge_cv,
                                                pressure=fluid_state.pressure,
                                                temperature=fluid_state.temperature,
                                                x_vec=fluid_nodes,
                                                eos=eos, time=t)
                sponge_cv = bulk_init.add_outlet(cv=sponge_cv,
                                                pressure=fluid_state.pressure,
                                                temperature=fluid_state.temperature,
                                                x_vec=fluid_nodes,
                                                eos=eos, time=t)

                if use_injection is True:
                    sponge_cv = bulk_init.add_injection(
                        cv=sponge_cv, pressure=fluid_state.pressure,
                        temperature=fluid_state.temperature, eos=eos_init,
                        x_vec=fluid_nodes)

                if use_upstream_injection is True:
                    sponge_cv = bulk_init.add_injection_upstream(
                        cv=sponge_cv, pressure=fluid_state.pressure,
                        temperature=fluid_state.temperature,
                        eos=eos_init, x_vec=fluid_nodes)
            else:
                sponge_cv = target_fluid_state.cv

            fluid_rhs = fluid_rhs + _sponge_source(sigma=sponge_sigma,
                                                   cv=cv,
                                                   sponge_cv=sponge_cv)

        if use_wall:
            # wall mass loss
            wall_mass_rhs = actx.np.zeros_like(wv.mass)
            if use_wall_mass:
                wall_mass_rhs = -wall_model.mass_loss_rate(
                    mass=wv.mass, ox_mass=wv.ox_mass,
                    temperature=wdv.temperature)

            # wall oxygen diffusion
            wall_ox_mass_rhs = actx.np.zeros_like(wv.mass)
            if use_wall_ox:
                if nspecies == 0:
                    fluid_ox_mass = mf_o2 + actx.np.zeros_like(cv.mass)
                elif nspecies > 3:
                    fluid_ox_mass = cv.species_mass[i_ox]
                else:
                    fluid_ox_mass = mf_o2*cv.species_mass[0]
                pairwise_ox = {
                    (dd_vol_fluid, dd_vol_wall):
                        (fluid_ox_mass, wv.ox_mass)}
                pairwise_ox_tpairs = inter_volume_trace_pairs(
                    dcoll, pairwise_ox, comm_tag=_OxCommTag)
                ox_tpairs = pairwise_ox_tpairs[dd_vol_fluid, dd_vol_wall]
                wall_ox_boundaries = {
                    wall_ffld_bnd.domain_tag:  # pylint: disable=no-member
                    DirichletDiffusionBoundary(0)}

                wall_ox_boundaries.update({
                    tpair.dd.domain_tag:
                    DirichletDiffusionBoundary(
                        op.project(dcoll, tpair.dd,
                                   tpair.dd.with_discr_tag(quadrature_tag),
                                   tpair.ext))
                    for tpair in ox_tpairs})

                wall_ox_mass_rhs = diffusion_operator(
                    dcoll, wall_model.oxygen_diffusivity,
                    wall_ox_boundaries, wv.ox_mass,
                    penalty_amount=wall_penalty_amount,
                    quadrature_tag=quadrature_tag, dd=dd_vol_wall,
                    comm_tag=_WallOxDiffCommTag)

            wall_rhs = wall_time_scale * WallVars(
                mass=wall_mass_rhs,
                energy=wall_energy_rhs,
                ox_mass=wall_ox_mass_rhs)

            if use_wall_ox:
                # Solve a diffusion equation in the fluid too just to ensure all MPI
                # sends/recvs from inter_volume_trace_pairs are in DAG
                # FIXME: this is dumb
                reverse_ox_tpairs = pairwise_ox_tpairs[dd_vol_wall, dd_vol_fluid]
                fluid_ox_boundaries = {
                    bdtag: DirichletDiffusionBoundary(0)
                    for bdtag in uncoupled_fluid_boundaries}
                fluid_ox_boundaries.update({
                    tpair.dd.domain_tag:
                    DirichletDiffusionBoundary(
                        op.project(dcoll, tpair.dd,
                                   tpair.dd.with_discr_tag(quadrature_tag),
                                   tpair.ext))
                    for tpair in reverse_ox_tpairs})

                fluid_dummy_ox_mass_rhs = diffusion_operator(
                    dcoll, 0, fluid_ox_boundaries, fluid_ox_mass,
                    quadrature_tag=quadrature_tag, dd=dd_vol_fluid,
                    comm_tag=_FluidOxDiffCommTag)

                fluid_rhs = fluid_rhs + 0*fluid_dummy_ox_mass_rhs

        print_stuff = False
        index = 6000
        if print_stuff is True:
            # initial state
            np.set_printoptions(threshold=sys.maxsize, precision=16)
            print("end of my_rhs")

            # the update to cv if this were an euler step
            cvrhs = cv + fluid_rhs*1.e-7

            data = actx.to_numpy(cvrhs.mass)
            print(f"mass {data[0][index]}")
            for i in range(0, nspecies):
                data = actx.to_numpy(cvrhs.species_mass_fractions)
                print(f"Y_[{i}] {data[i][0][index]} ")

            for i in range(0, nspecies):
                data = actx.to_numpy(cvrhs.species_mass)
                print(f"rhoY_[{i}] {data[i][0][index]} ")

        rhs_stepper_state = make_stepper_state(
            cv=fluid_rhs,
            tseed=tseed_rhs,
            wv=wall_rhs,
            av_smu=av_smu_rhs,
            av_sbeta=av_sbeta_rhs,
            av_skappa=av_skappa_rhs,
            av_sd=av_sd_rhs,
            smin=smin_rhs)

        #print(f"{actx.to_numpy(fluid_rhs.species_mass)=}")
        print("End of unfiltered_rhs")

        return rhs_stepper_state.get_obj_array()

    unfiltered_rhs_compiled = actx.compile(unfiltered_rhs)

    def my_rhs(t, state):

        # precludes a pre-compiled timestepper
        # don't know if we should do this
        #state = force_evaluation(actx, state)

        # Work around long compile issue by computing and filtering RHS in separate
        # compiled functions
        rhs_state = unfiltered_rhs_compiled(t, state)
        #rhs_data_precompute = precompute_rhs_compiled(t, state)
        #rhs_state = compute_rhs_compiled(t, rhs_data_precompute)

        # Use a spectral filter on the RHS
        if use_rhs_filter:
            rhs_state_filtered = make_stepper_state_obj(rhs_state)
            rhs_state_filtered = rhs_state_filtered.replace(
                cv=filter_rhs_fluid_compiled(rhs_state_filtered.cv))
            if use_wall:
                # pylint: disable=no-member
                rhs_state_filtered = rhs_state_filtered.replace(
                    wv=filter_rhs_wall_compiled(rhs_state_filtered.wv))
                # pylint: enable=no-member

            rhs_state = rhs_state_filtered.get_obj_array()

        return rhs_state

    """
    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)
    """

    pre_step_callback = my_pre_step if use_callbacks else None
    post_step_callback = my_post_step if use_callbacks else None
    if advance_time:
        current_step, current_t, current_stepper_state_obj = \
            advance_state(rhs=my_rhs, timestepper=timestepper,
                          pre_step_callback=pre_step_callback,
                          post_step_callback=post_step_callback,
                          istep=current_step, dt=current_dt,
                          t=current_t, t_final=t_final,
                          force_eval=force_eval,
                          state=stepper_state.get_obj_array(),
                          compile_rhs=False)
        current_stepper_state = make_stepper_state_obj(current_stepper_state_obj)
    else:
        current_stepper_state = stepper_state

    current_cv = current_stepper_state.cv
    tseed = current_stepper_state.tseed
    current_av_smu = current_stepper_state.av_smu
    current_av_sbeta = current_stepper_state.av_sbeta
    current_av_skappa = current_stepper_state.av_skappa
    current_av_sd = current_stepper_state.av_sd
    current_smin = current_stepper_state.smin

    # we can't get the limited viz data back from create_fluid_state
    # so call the limiter directly first, basically doing the limiting twice
    theta_rho = actx.np.zeros_like(current_cv.mass)
    theta_Y = actx.np.zeros_like(current_cv.mass)
    theta_pres = actx.np.zeros_like(current_cv.mass)
    if viz_level >= 2 and use_species_limiter == 2:
        cv_lim, theta_rho, theta_Y, theta_pres = \
            limit_fluid_state_lv(
                dcoll, cv=current_cv, gas_model=gas_model,
                temperature_seed=tseed, entropy_min=current_smin,
                dd=dd_vol_fluid, viz_theta=True)

    current_fluid_state = create_fluid_state(current_cv, tseed,
                                             smoothness_mu=current_av_smu,
                                             smoothness_beta=current_av_sbeta,
                                             smoothness_kappa=current_av_skappa,
                                             smoothness_d=current_av_sd,
                                             entropy_min=current_smin)

    if last_profiling_step < 0:
        MPI.Pcontrol(0)

    if use_wall:
        current_wv = current_stepper_state.wv
        current_wdv = create_wall_dependent_vars_compiled(current_wv)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    ts_field_fluid, cfl, dt = my_get_timestep(dcoll=dcoll,
        fluid_state=current_fluid_state,
        t=current_t, dt=current_dt, cfl=current_cfl,
        t_final=t_final, constant_cfl=constant_cfl, fluid_dd=dd_vol_fluid)

    ts_field_wall = None
    if use_wall:
        ts_field_wall, cfl_wall, dt_wall = my_get_timestep_wall(dcoll=dcoll,
            wv=current_wv, wall_kappa=current_wdv.thermal_conductivity,
            wall_temperature=current_wdv.temperature, t=current_t, dt=current_dt,
            cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl,
            wall_dd=dd_vol_wall)
    current_t_wall = t_wall_start + (current_step - first_step)*dt*wall_time_scale

    my_write_status_lite(step=current_step, t=current_t,
                         t_wall=current_t_wall)

    my_write_status_fluid(fluid_state=current_fluid_state, dt=dt, cfl_fluid=cfl)
    if use_wall:
        my_write_status_wall(wall_temperature=current_wdv.temperature,
                             dt=dt*wall_time_scale, cfl_wall=cfl_wall)

    if viz_interval_type == 0:
        dump_number = current_step
    else:
        dump_number = (math.floor((current_t - t_start)/t_viz_interval) +
            last_viz_interval)

    if nviz > 0:
        # pack things up
        if use_wall:
            viz_state = make_obj_array([current_fluid_state, tseed,
                                        current_smin, current_wv])
            viz_dv = make_obj_array([current_fluid_state.dv, current_wdv])
        else:
            viz_state = make_obj_array([current_fluid_state, tseed,
                                        current_smin])
            viz_dv = current_fluid_state.dv

        my_write_viz(
            step=current_step, t=current_t, t_wall=current_t_wall,
            viz_state=viz_state, viz_dv=viz_dv,
            ts_field_fluid=ts_field_fluid,
            ts_field_wall=ts_field_wall,
            theta_rho=theta_rho,
            theta_Y=theta_Y,
            theta_pres=theta_pres,
            dump_number=dump_number)

    if nrestart > 0:
        my_write_restart(step=current_step, t=current_t, t_wall=current_t_wall,
                         state=current_stepper_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 2*current_dt
    assert np.abs(current_t - t_final) < finish_tol

# vim: foldmethod=marker

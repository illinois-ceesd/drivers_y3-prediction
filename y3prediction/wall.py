import numpy as np
from meshmode.dof_array import DOFArray
from dataclasses import dataclass, fields
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively,
    tag_axes,
)
from meshmode.transform_metadata import (
    DiscretizationElementAxisTag,
    DiscretizationDOFAxisTag)


def mask_from_elements(vol_discr, actx, elements):
    mesh = vol_discr.mesh
    zeros = vol_discr.zeros(actx)

    group_arrays = []

    for igrp in range(len(mesh.groups)):
        start_elem_nr = mesh.base_element_nrs[igrp]
        end_elem_nr = start_elem_nr + mesh.groups[igrp].nelements
        grp_elems = elements[
            (elements >= start_elem_nr)
            & (elements < end_elem_nr)] - start_elem_nr
        grp_ary_np = actx.to_numpy(zeros[igrp])
        grp_ary_np[grp_elems] = 1
        group_arrays.append(actx.from_numpy(grp_ary_np))

    return tag_axes(actx, {
                0: DiscretizationElementAxisTag(),
                1: DiscretizationDOFAxisTag()
            }, DOFArray(actx, tuple(group_arrays)))


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallVars:
    mass: DOFArray
    energy: DOFArray
    ox_mass: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.mass)

    def __reduce__(self):
        """Return a tuple reproduction of self for pickling."""
        return (WallVars, tuple(getattr(self, f.name)
                                    for f in fields(WallVars)))


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    thermal_conductivity: DOFArray
    temperature: DOFArray


class WallModel:
    """Model for calculating wall quantities."""
    def __init__(
            self,
            heat_capacity,
            thermal_conductivity_func,
            *,
            effective_surface_area_func=None,
            mass_loss_func=None,
            oxygen_diffusivity=0.):
        self._heat_capacity = heat_capacity
        self._thermal_conductivity_func = thermal_conductivity_func
        self._effective_surface_area_func = effective_surface_area_func
        self._mass_loss_func = mass_loss_func
        self._oxygen_diffusivity = oxygen_diffusivity

    @property
    def heat_capacity(self):
        return self._heat_capacity

    def thermal_conductivity(self, mass, temperature):
        return self._thermal_conductivity_func(mass, temperature)

    def thermal_diffusivity(self, mass, temperature, thermal_conductivity=None):
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity(mass, temperature)
        return thermal_conductivity/(mass * self.heat_capacity)

    def mass_loss_rate(self, mass, ox_mass, temperature):
        dm = mass*0.
        if self._effective_surface_area_func is not None:
            eff_surf_area = self._effective_surface_area_func(mass)
            if self._mass_loss_func is not None:
                dm = self._mass_loss_func(mass, ox_mass, temperature, eff_surf_area)
        return dm

    @property
    def oxygen_diffusivity(self):
        return self._oxygen_diffusivity

    def temperature(self, wv):
        return wv.energy/(wv.mass * self.heat_capacity)

    def dependent_vars(self, wv):
        temperature = self.temperature(wv)
        kappa = self.thermal_conductivity(wv.mass, temperature)
        return WallDependentVars(
            thermal_conductivity=kappa,
            temperature=temperature)

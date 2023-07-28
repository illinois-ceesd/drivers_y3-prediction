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

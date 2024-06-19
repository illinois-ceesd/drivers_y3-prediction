"""
.. autoclass:: Thermochemistry
"""


from warnings import warn
import numpy as np


class Thermochemistry:
    """
    .. attribute:: model_name
    .. attribute:: num_elements
    .. attribute:: num_species
    .. attribute:: num_reactions
    .. attribute:: num_falloff
    .. attribute:: one_atm

        Returns 1 atm in SI units of pressure (Pa).

    .. attribute:: gas_constant
    .. attribute:: species_names
    .. attribute:: species_indices

    .. automethod:: get_specific_gas_constant
    .. automethod:: get_density
    .. automethod:: get_pressure
    .. automethod:: get_mix_molecular_weight
    .. automethod:: get_concentrations
    .. automethod:: get_mole_fractions
    .. automethod:: get_mass_average_property
    .. automethod:: get_mixture_specific_heat_cp_mass
    .. automethod:: get_mixture_specific_heat_cv_mass
    .. automethod:: get_mixture_enthalpy_mass
    .. automethod:: get_mixture_internal_energy_mass
    .. automethod:: get_species_viscosities
    .. automethod:: get_mixture_viscosity_mixavg
    .. automethod:: get_species_thermal_conductivities
    .. automethod:: get_mixture_thermal_conductivity_mixavg
    .. automethod:: get_species_binary_mass_diffusivities
    .. automethod:: get_species_mass_diffusivities_mixavg
    .. automethod:: get_species_specific_heats_r
    .. automethod:: get_species_enthalpies_rt
    .. automethod:: get_species_entropies_r
    .. automethod:: get_species_gibbs_rt
    .. automethod:: get_equilibrium_constants
    .. automethod:: get_temperature
    .. automethod:: __init__
    """

    def __init__(self, usr_np=np):
        """Initialize thermochemistry object for a mechanism.

        Parameters
        ----------
        usr_np
            :mod:`numpy`-like namespace providing at least the following functions,
            for any array ``X`` of the bulk array type:

            - ``usr_np.log(X)`` (like :data:`numpy.log`)
            - ``usr_np.log10(X)`` (like :data:`numpy.log10`)
            - ``usr_np.exp(X)`` (like :data:`numpy.exp`)
            - ``usr_np.where(X > 0, X_yes, X_no)`` (like :func:`numpy.where`)
            - ``usr_np.linalg.norm(X, np.inf)`` (like :func:`numpy.linalg.norm`)

            where the "bulk array type" is a type that offers arithmetic analogous
            to :class:`numpy.ndarray` and is used to hold all types of (potentialy
            volumetric) "bulk data", such as temperature, pressure, mass fractions,
            etc. This parameter defaults to *actual numpy*, so it can be ignored
            unless it is needed by the user (e.g. for purposes of
            GPU processing or automatic differentiation).

        """

        self.usr_np = usr_np
        self.model_name = 'sandiego_ec.yaml'
        self.num_elements = 3
        self.num_species = 9
        self.num_reactions = 24
        self.num_falloff = 0

        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['H2', 'H', 'O2', 'O', 'OH', 'HO2', 'H2O2', 'H2O', 'N2']
        self.species_indices = {'H2': 0, 'H': 1, 'O2': 2, 'O': 3, 'OH': 4, 'HO2': 5, 'H2O2': 6, 'H2O': 7, 'N2': 8}

        self.molecular_weights = np.array([2.016, 1.008, 31.998, 15.999, 17.007, 33.006, 34.014, 18.015, 28.014])
        self.inv_molecular_weights = 1/self.molecular_weights

    @property
    def wts(self):
        warn("Thermochemistry.wts is deprecated and will go away in 2024. "
             "Use molecular_weights instead.", DeprecationWarning, stacklevel=2)

        return self.molecular_weights

    @property
    def iwts(self):
        warn("Thermochemistry.iwts is deprecated and will go away in 2024. "
             "Use inv_molecular_weights instead.", DeprecationWarning, stacklevel=2)

        return self.inv_molecular_weights

    def _pyro_zeros_like(self, argument):
        # FIXME: This is imperfect, as a NaN will stay a NaN.
        return 0 * argument

    def _pyro_make_array(self, res_list):
        """This works around (e.g.) numpy.exp not working with object
        arrays of numpy scalars. It defaults to making object arrays, however
        if an array consists of all scalars, it makes a "plain old"
        :class:`numpy.ndarray`.

        See ``this numpy bug <https://github.com/numpy/numpy/issues/18004>`__
        for more context.
        """

        from numbers import Number
        all_numbers = all(isinstance(e, Number) for e in res_list)

        dtype = np.float64 if all_numbers else object
        result = np.empty((len(res_list),), dtype=dtype)

        # 'result[:] = res_list' may look tempting, however:
        # https://github.com/numpy/numpy/issues/16564
        for idx in range(len(res_list)):
            result[idx] = res_list[idx]

        return result

    def _pyro_norm(self, argument, normord):
        """This works around numpy.linalg norm not working with scalars.

        If the argument is a regular ole number, it uses :func:`numpy.abs`,
        otherwise it uses ``usr_np.linalg.norm``.
        """
        # Wrap norm for scalars

        from numbers import Number

        if isinstance(argument, Number):
            return np.abs(argument)
        return self.usr_np.linalg.norm(argument, normord)

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, mass_fractions):
        return self.gas_constant * (
            + self.inv_molecular_weights[0]*mass_fractions[0]
            + self.inv_molecular_weights[1]*mass_fractions[1]
            + self.inv_molecular_weights[2]*mass_fractions[2]
            + self.inv_molecular_weights[3]*mass_fractions[3]
            + self.inv_molecular_weights[4]*mass_fractions[4]
            + self.inv_molecular_weights[5]*mass_fractions[5]
            + self.inv_molecular_weights[6]*mass_fractions[6]
            + self.inv_molecular_weights[7]*mass_fractions[7]
            + self.inv_molecular_weights[8]*mass_fractions[8]
            )

    def get_density(self, pressure, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return pressure * mmw / rt

    def get_pressure(self, rho, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return rho * rt / mmw

    def get_mix_molecular_weight(self, mass_fractions):
        return 1/(
            + self.inv_molecular_weights[0]*mass_fractions[0]
            + self.inv_molecular_weights[1]*mass_fractions[1]
            + self.inv_molecular_weights[2]*mass_fractions[2]
            + self.inv_molecular_weights[3]*mass_fractions[3]
            + self.inv_molecular_weights[4]*mass_fractions[4]
            + self.inv_molecular_weights[5]*mass_fractions[5]
            + self.inv_molecular_weights[6]*mass_fractions[6]
            + self.inv_molecular_weights[7]*mass_fractions[7]
            + self.inv_molecular_weights[8]*mass_fractions[8]
            )

    def get_concentrations(self, rho, mass_fractions):
        return self._pyro_make_array([
            self.inv_molecular_weights[0] * rho * mass_fractions[0],
            self.inv_molecular_weights[1] * rho * mass_fractions[1],
            self.inv_molecular_weights[2] * rho * mass_fractions[2],
            self.inv_molecular_weights[3] * rho * mass_fractions[3],
            self.inv_molecular_weights[4] * rho * mass_fractions[4],
            self.inv_molecular_weights[5] * rho * mass_fractions[5],
            self.inv_molecular_weights[6] * rho * mass_fractions[6],
            self.inv_molecular_weights[7] * rho * mass_fractions[7],
            self.inv_molecular_weights[8] * rho * mass_fractions[8],
        ])

    def get_mole_fractions(self, mix_mol_weight, mass_fractions):
        return self._pyro_make_array([
            self.inv_molecular_weights[0] * mass_fractions[0] * mix_mol_weight,
            self.inv_molecular_weights[1] * mass_fractions[1] * mix_mol_weight,
            self.inv_molecular_weights[2] * mass_fractions[2] * mix_mol_weight,
            self.inv_molecular_weights[3] * mass_fractions[3] * mix_mol_weight,
            self.inv_molecular_weights[4] * mass_fractions[4] * mix_mol_weight,
            self.inv_molecular_weights[5] * mass_fractions[5] * mix_mol_weight,
            self.inv_molecular_weights[6] * mass_fractions[6] * mix_mol_weight,
            self.inv_molecular_weights[7] * mass_fractions[7] * mix_mol_weight,
            self.inv_molecular_weights[8] * mass_fractions[8] * mix_mol_weight,
            ])

    def get_mass_average_property(self, mass_fractions, spec_property):
        return sum([
            mass_fractions[i] * spec_property[i] * self.inv_molecular_weights[i]
            for i in range(self.num_species)])

    def get_mixture_specific_heat_cp_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature)
        cp_mix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cp_mix

    def get_mixture_specific_heat_cv_mass(self, temperature, mass_fractions):
        cv0_r = self.get_species_specific_heats_r(temperature) - 1.0
        cv_mix = self.get_mass_average_property(mass_fractions, cv0_r)
        return self.gas_constant * cv_mix

    def get_mixture_enthalpy_mass(self, temperature, mass_fractions):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        h_mix = self.get_mass_average_property(mass_fractions, h0_rt)
        return self.gas_constant * temperature * h_mix

    def get_mixture_internal_energy_mass(self, temperature, mass_fractions):
        e0_rt = self.get_species_enthalpies_rt(temperature) - 1.0
        e_mix = self.get_mass_average_property(mass_fractions, e0_rt)
        return self.gas_constant * temperature * e_mix

    def get_species_specific_heats_r(self, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -4.94024731e-05*temperature + 4.99456778e-07*temperature**2 + -1.79566394e-10*temperature**3 + 2.00255376e-14*temperature**4, 2.34433112 + 0.00798052075*temperature + -1.9478151e-05*temperature**2 + 2.01572094e-08*temperature**3 + -7.37611761e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.50000001 + -2.30842973e-11*temperature + 1.61561948e-14*temperature**2 + -4.73515235e-18*temperature**3 + 4.98197357e-22*temperature**4, 2.5 + 7.05332819e-13*temperature + -1.99591964e-15*temperature**2 + 2.30081632e-18*temperature**3 + -9.27732332e-22*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00148308754*temperature + -7.57966669e-07*temperature**2 + 2.09470555e-10*temperature**3 + -2.16717794e-14*temperature**4, 3.78245636 + -0.00299673416*temperature + 9.84730201e-06*temperature**2 + -9.68129509e-09*temperature**3 + 3.24372837e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.56942078 + -8.59741137e-05*temperature + 4.19484589e-08*temperature**2 + -1.00177799e-11*temperature**3 + 1.22833691e-15*temperature**4, 3.1682671 + -0.00327931884*temperature + 6.64306396e-06*temperature**2 + -6.12806624e-09*temperature**3 + 2.11265971e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.86472886 + 0.00105650448*temperature + -2.59082758e-07*temperature**2 + 3.05218674e-11*temperature**3 + -1.33195876e-15*temperature**4, 4.12530561 + -0.00322544939*temperature + 6.52764691e-06*temperature**2 + -5.79853643e-09*temperature**3 + 2.06237379e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.0172109 + 0.00223982013*temperature + -6.3365815e-07*temperature**2 + 1.1424637e-10*temperature**3 + -1.07908535e-14*temperature**4, 4.30179801 + -0.00474912051*temperature + 2.11582891e-05*temperature**2 + -2.42763894e-08*temperature**3 + 9.29225124e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.16500285 + 0.00490831694*temperature + -1.90139225e-06*temperature**2 + 3.71185986e-10*temperature**3 + -2.87908305e-14*temperature**4, 4.27611269 + -0.000542822417*temperature + 1.67335701e-05*temperature**2 + -2.15770813e-08*temperature**3 + 8.62454363e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00217691804*temperature + -1.64072518e-07*temperature**2 + -9.7041987e-11*temperature**3 + 1.68200992e-14*temperature**4, 4.19864056 + -0.0020364341*temperature + 6.52040211e-06*temperature**2 + -5.48797062e-09*temperature**3 + 1.77197817e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0014879768*temperature + -5.68476e-07*temperature**2 + 1.0097038e-10*temperature**3 + -6.753351e-15*temperature**4, 3.298677 + 0.0014082404*temperature + -3.963222e-06*temperature**2 + 5.641515e-09*temperature**3 + -2.444854e-12*temperature**4),
            ])

    def get_species_enthalpies_rt(self, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -2.470123655e-05*temperature + 1.6648559266666665e-07*temperature**2 + -4.48915985e-11*temperature**3 + 4.00510752e-15*temperature**4 + -950.158922 / temperature, 2.34433112 + 0.003990260375*temperature + -6.4927169999999995e-06*temperature**2 + 5.03930235e-09*temperature**3 + -1.4752235220000002e-12*temperature**4 + -917.935173 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.50000001 + -1.154214865e-11*temperature + 5.385398266666667e-15*temperature**2 + -1.1837880875e-18*temperature**3 + 9.96394714e-23*temperature**4 + 25473.6599 / temperature, 2.5 + 3.526664095e-13*temperature + -6.653065466666667e-16*temperature**2 + 5.7520408e-19*temperature**3 + -1.855464664e-22*temperature**4 + 25473.6599 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00074154377*temperature + -2.526555563333333e-07*temperature**2 + 5.236763875e-11*temperature**3 + -4.33435588e-15*temperature**4 + -1088.45772 / temperature, 3.78245636 + -0.00149836708*temperature + 3.282434003333333e-06*temperature**2 + -2.4203237725e-09*temperature**3 + 6.48745674e-13*temperature**4 + -1063.94356 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.56942078 + -4.298705685e-05*temperature + 1.3982819633333334e-08*temperature**2 + -2.504444975e-12*temperature**3 + 2.4566738199999997e-16*temperature**4 + 29217.5791 / temperature, 3.1682671 + -0.00163965942*temperature + 2.2143546533333334e-06*temperature**2 + -1.53201656e-09*temperature**3 + 4.22531942e-13*temperature**4 + 29122.2592 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.86472886 + 0.00052825224*temperature + -8.636091933333334e-08*temperature**2 + 7.63046685e-12*temperature**3 + -2.66391752e-16*temperature**4 + 3718.85774 / temperature, 4.12530561 + -0.001612724695*temperature + 2.1758823033333334e-06*temperature**2 + -1.4496341075e-09*temperature**3 + 4.1247475799999997e-13*temperature**4 + 3381.53812 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.0172109 + 0.001119910065*temperature + -2.1121938333333332e-07*temperature**2 + 2.85615925e-11*temperature**3 + -2.1581707e-15*temperature**4 + 111.856713 / temperature, 4.30179801 + -0.002374560255*temperature + 7.0527630333333326e-06*temperature**2 + -6.06909735e-09*temperature**3 + 1.8584502480000002e-12*temperature**4 + 294.80804 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.16500285 + 0.00245415847*temperature + -6.337974166666666e-07*temperature**2 + 9.27964965e-11*temperature**3 + -5.7581661e-15*temperature**4 + -17861.7877 / temperature, 4.27611269 + -0.0002714112085*temperature + 5.5778567000000005e-06*temperature**2 + -5.394270325e-09*temperature**3 + 1.724908726e-12*temperature**4 + -17702.5821 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00108845902*temperature + -5.469083933333333e-08*temperature**2 + -2.426049675e-11*temperature**3 + 3.36401984e-15*temperature**4 + -30004.2971 / temperature, 4.19864056 + -0.00101821705*temperature + 2.17346737e-06*temperature**2 + -1.371992655e-09*temperature**3 + 3.54395634e-13*temperature**4 + -30293.7267 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0007439884*temperature + -1.8949200000000001e-07*temperature**2 + 2.5242595e-11*temperature**3 + -1.3506701999999999e-15*temperature**4 + -922.7977 / temperature, 3.298677 + 0.0007041202*temperature + -1.3210739999999999e-06*temperature**2 + 1.41037875e-09*temperature**3 + -4.889707999999999e-13*temperature**4 + -1020.8999 / temperature),
            ])

    def get_species_entropies_r(self, pressure, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792*self.usr_np.log(temperature) + -4.94024731e-05*temperature + 2.49728389e-07*temperature**2 + -5.985546466666667e-11*temperature**3 + 5.0063844e-15*temperature**4 + -3.20502331, 2.34433112*self.usr_np.log(temperature) + 0.00798052075*temperature + -9.7390755e-06*temperature**2 + 6.7190698e-09*temperature**3 + -1.8440294025e-12*temperature**4 + 0.683010238)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.50000001*self.usr_np.log(temperature) + -2.30842973e-11*temperature + 8.0780974e-15*temperature**2 + -1.5783841166666668e-18*temperature**3 + 1.2454933925e-22*temperature**4 + -0.446682914, 2.5*self.usr_np.log(temperature) + 7.05332819e-13*temperature + -9.9795982e-16*temperature**2 + 7.669387733333333e-19*temperature**3 + -2.31933083e-22*temperature**4 + -0.446682853)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784*self.usr_np.log(temperature) + 0.00148308754*temperature + -3.789833345e-07*temperature**2 + 6.982351833333333e-11*temperature**3 + -5.41794485e-15*temperature**4 + 5.45323129, 3.78245636*self.usr_np.log(temperature) + -0.00299673416*temperature + 4.923651005e-06*temperature**2 + -3.2270983633333334e-09*temperature**3 + 8.109320925e-13*temperature**4 + 3.65767573)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.56942078*self.usr_np.log(temperature) + -8.59741137e-05*temperature + 2.097422945e-08*temperature**2 + -3.3392599666666663e-12*temperature**3 + 3.070842275e-16*temperature**4 + 4.78433864, 3.1682671*self.usr_np.log(temperature) + -0.00327931884*temperature + 3.32153198e-06*temperature**2 + -2.0426887466666666e-09*temperature**3 + 5.281649275e-13*temperature**4 + 2.05193346)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.86472886*self.usr_np.log(temperature) + 0.00105650448*temperature + -1.29541379e-07*temperature**2 + 1.01739558e-11*temperature**3 + -3.3298969e-16*temperature**4 + 5.70164073, 4.12530561*self.usr_np.log(temperature) + -0.00322544939*temperature + 3.263823455e-06*temperature**2 + -1.9328454766666666e-09*temperature**3 + 5.155934475e-13*temperature**4 + -0.69043296)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.0172109*self.usr_np.log(temperature) + 0.00223982013*temperature + -3.16829075e-07*temperature**2 + 3.808212333333334e-11*temperature**3 + -2.697713375e-15*temperature**4 + 3.78510215, 4.30179801*self.usr_np.log(temperature) + -0.00474912051*temperature + 1.057914455e-05*temperature**2 + -8.0921298e-09*temperature**3 + 2.32306281e-12*temperature**4 + 3.71666245)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.16500285*self.usr_np.log(temperature) + 0.00490831694*temperature + -9.50696125e-07*temperature**2 + 1.2372866199999999e-10*temperature**3 + -7.197707625e-15*temperature**4 + 2.91615662, 4.27611269*self.usr_np.log(temperature) + -0.000542822417*temperature + 8.36678505e-06*temperature**2 + -7.192360433333333e-09*temperature**3 + 2.1561359075e-12*temperature**4 + 3.43505074)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249*self.usr_np.log(temperature) + 0.00217691804*temperature + -8.2036259e-08*temperature**2 + -3.2347329e-11*temperature**3 + 4.2050248e-15*temperature**4 + 4.9667701, 4.19864056*self.usr_np.log(temperature) + -0.0020364341*temperature + 3.260201055e-06*temperature**2 + -1.82932354e-09*temperature**3 + 4.429945425e-13*temperature**4 + -0.849032208)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664*self.usr_np.log(temperature) + 0.0014879768*temperature + -2.84238e-07*temperature**2 + 3.3656793333333334e-11*temperature**3 + -1.68833775e-15*temperature**4 + 5.980528, 3.298677*self.usr_np.log(temperature) + 0.0014082404*temperature + -1.981611e-06*temperature**2 + 1.8805050000000002e-09*temperature**3 + -6.112135e-13*temperature**4 + 3.950372)
            - self.usr_np.log(pressure/self.one_atm),
            ])

    def get_species_gibbs_rt(self, pressure, temperature):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        s0_r = self.get_species_entropies_r(pressure, temperature)
        return h0_rt - s0_r

    def get_equilibrium_constants(self, pressure, temperature):
        rt = self.gas_constant * temperature
        c0 = self.usr_np.log(pressure / rt)

        g0_rt = self.get_species_gibbs_rt(pressure, temperature)
        return self._pyro_make_array([
            g0_rt[3] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[2]),
            g0_rt[1] + g0_rt[4] + -1*(g0_rt[0] + g0_rt[3]),
            g0_rt[1] + g0_rt[7] + -1*(g0_rt[0] + g0_rt[4]),
            2.0*g0_rt[4] + -1*(g0_rt[7] + g0_rt[3]),
            g0_rt[0] + -1*2.0*g0_rt[1] + -1*-1.0*c0,
            g0_rt[7] + -1*(g0_rt[1] + g0_rt[4]) + -1*-1.0*c0,
            g0_rt[2] + -1*2.0*g0_rt[3] + -1*-1.0*c0,
            g0_rt[4] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
            g0_rt[5] + -1*(g0_rt[3] + g0_rt[4]) + -1*-1.0*c0,
            g0_rt[5] + -1*(g0_rt[1] + g0_rt[2]) + -1*-1.0*c0,
            2.0*g0_rt[4] + -1*(g0_rt[1] + g0_rt[5]),
            g0_rt[0] + g0_rt[2] + -1*(g0_rt[1] + g0_rt[5]),
            g0_rt[7] + g0_rt[3] + -1*(g0_rt[1] + g0_rt[5]),
            g0_rt[2] + g0_rt[4] + -1*(g0_rt[5] + g0_rt[3]),
            g0_rt[7] + g0_rt[2] + -1*(g0_rt[5] + g0_rt[4]),
            g0_rt[7] + g0_rt[2] + -1*(g0_rt[5] + g0_rt[4]),
            g0_rt[6] + -1*2.0*g0_rt[4] + -1*-1.0*c0,
            g0_rt[6] + g0_rt[2] + -1*2.0*g0_rt[5],
            g0_rt[6] + g0_rt[2] + -1*2.0*g0_rt[5],
            g0_rt[0] + g0_rt[5] + -1*(g0_rt[1] + g0_rt[6]),
            g0_rt[7] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[6]),
            g0_rt[7] + g0_rt[5] + -1*(g0_rt[6] + g0_rt[4]),
            g0_rt[7] + g0_rt[5] + -1*(g0_rt[6] + g0_rt[4]),
            g0_rt[5] + g0_rt[4] + -1*(g0_rt[6] + g0_rt[3]),
            ])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):
        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        ones = self._pyro_zeros_like(enthalpy_or_energy) + 1.0
        t_i = t_guess * ones

        for _ in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i += dt
            if self._pyro_norm(dt, np.inf) < tol:
                return t_i

        raise RuntimeError("Temperature iteration failed to converge")

    def get_falloff_rates(self, temperature, concentrations):
        k_high = self._pyro_make_array([
            4650000000.000001*temperature**0.44,
            95500000000.00002*temperature**-0.27,
        ])

        k_low = self._pyro_make_array([
            57500000000000.01*temperature**-1.4,
            2.760000000000001e+19*temperature**-3.2,
        ])

        reduced_pressure = self._pyro_make_array([
            (2.5*concentrations[0] + 16.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8])*k_low[0]/k_high[0],
            (2.5*concentrations[0] + 6.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8])*k_low[1]/k_high[1],
        ])

        falloff_center = self._pyro_make_array([
            self.usr_np.log10(0.5*self.usr_np.exp((-1*temperature) / 1e-30) + 0.5*self.usr_np.exp((-1*temperature) / 1.0000000000000002e+30)),
            self.usr_np.log10(0.43000000000000005*self.usr_np.exp((-1*temperature) / 1.0000000000000002e+30) + 0.57*self.usr_np.exp((-1*temperature) / 1e-30)),
        ])

        falloff_function = self._pyro_make_array([
            10**(falloff_center[0] / (1 + ((self.usr_np.log10(reduced_pressure[0]) + -0.4 + -1*0.67*falloff_center[0]) / (0.75 + -1*1.27*falloff_center[0] + -1*0.14*(self.usr_np.log10(reduced_pressure[0]) + -0.4 + -1*0.67*falloff_center[0])))**2)),
            10**(falloff_center[1] / (1 + ((self.usr_np.log10(reduced_pressure[1]) + -0.4 + -1*0.67*falloff_center[1]) / (0.75 + -1*1.27*falloff_center[1] + -1*0.14*(self.usr_np.log10(reduced_pressure[1]) + -0.4 + -1*0.67*falloff_center[1])))**2)),
        ])*reduced_pressure/(1+reduced_pressure)

        return k_high*falloff_function

    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_falloff = self.get_falloff_rates(temperature, concentrations)
        k_fwd = [
            self.usr_np.exp(31.192067198532598 + -0.7*self.usr_np.log(temperature) + -1*(8589.851597151493 / temperature)) * ones,
            self.usr_np.exp(3.92395157629342 + 2.67*self.usr_np.log(temperature) + -1*(3165.568384724549 / temperature)) * ones,
            self.usr_np.exp(13.97251430677394 + 1.3*self.usr_np.log(temperature) + -1*(1829.342520199863 / temperature)) * ones,
            self.usr_np.exp(6.551080335043405 + 2.33*self.usr_np.log(temperature) + -1*(7320.978251450734 / temperature)) * ones,
            1300000000000.0002*temperature**-1.0 * ones,
            4.000000000000001e+16*temperature**-2.0 * ones,
            6170000000.000001*temperature**-0.5 * ones,
            4710000000000.001*temperature**-1.0 * ones,
            8000000000.000002 * ones,
            k_falloff[0]*ones,
            self.usr_np.exp(24.983124837646084 + -1*(148.41608612272393 / temperature)) * ones,
            self.usr_np.exp(23.532668532308907 + -1*(414.09771841210573 / temperature)) * ones,
            self.usr_np.exp(24.157253041431556 + -1*(865.9609563076275 / temperature)) * ones,
            20000000000.000004 * ones,
            self.usr_np.exp(26.832513419710775 + -1*(5500.054796103862 / temperature)) * ones,
            self.usr_np.exp(24.11777423045777 + -1*(-250.16649848887016 / temperature)) * ones,
            k_falloff[1]*ones,
            self.usr_np.exp(19.083368717027604 + -1*(-709.00553297687 / temperature)) * ones,
            self.usr_np.exp(25.357994825176046 + -1*(5556.582802973943 / temperature)) * ones,
            self.usr_np.exp(23.85876005287556 + -1*(4000.619345786196 / temperature)) * ones,
            self.usr_np.exp(23.025850929940457 + -1*(1804.0853256408905 / temperature)) * ones,
            self.usr_np.exp(25.052682521347997 + -1*(3659.8877639501534 / temperature)) * ones,
            self.usr_np.exp(21.27715095017285 + -1*(159.96223220682563 / temperature)) * ones,
            self.usr_np.exp(9.172638504792172 + 2.0*self.usr_np.log(temperature) + -1*(2008.5483292135248 / temperature)) * ones,
        ]
        return self._pyro_make_array(k_fwd)

    def get_rev_rate_coefficients(self, pressure, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(pressure, temperature)
        return self._pyro_make_array(k_fwd * self.usr_np.exp(log_k_eq))

    def get_net_rates_of_progress(self, pressure, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(pressure, temperature)
        return self._pyro_make_array([
            k_fwd[0]*(concentrations[1]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[0])*concentrations[3]*concentrations[4]),
            k_fwd[1]*(concentrations[0]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[1])*concentrations[1]*concentrations[4]),
            k_fwd[2]*(concentrations[0]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[2])*concentrations[1]*concentrations[7]),
            k_fwd[3]*(concentrations[7]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[3])*concentrations[4]**2.0),
            k_fwd[4]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[4])*concentrations[0])*(2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8]),
            k_fwd[5]*(concentrations[1]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[5])*concentrations[7])*(2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8]),
            k_fwd[6]*(concentrations[3]**2.0 + -1*self.usr_np.exp(log_k_eq[6])*concentrations[2])*(2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8]),
            k_fwd[7]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[7])*concentrations[4])*(2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8]),
            k_fwd[8]*(concentrations[3]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[8])*concentrations[5])*(2.5*concentrations[0] + 12.0*concentrations[7] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[8]),
            k_fwd[9]*(concentrations[1]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[9])*concentrations[5]),
            k_fwd[10]*(concentrations[1]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[10])*concentrations[4]**2.0),
            k_fwd[11]*(concentrations[1]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[11])*concentrations[0]*concentrations[2]),
            k_fwd[12]*(concentrations[1]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[12])*concentrations[7]*concentrations[3]),
            k_fwd[13]*(concentrations[5]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[13])*concentrations[2]*concentrations[4]),
            k_fwd[14]*(concentrations[5]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[14])*concentrations[7]*concentrations[2]),
            k_fwd[15]*(concentrations[5]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[15])*concentrations[7]*concentrations[2]),
            k_fwd[16]*(concentrations[4]**2.0 + -1*self.usr_np.exp(log_k_eq[16])*concentrations[6]),
            k_fwd[17]*(concentrations[5]**2.0 + -1*self.usr_np.exp(log_k_eq[17])*concentrations[6]*concentrations[2]),
            k_fwd[18]*(concentrations[5]**2.0 + -1*self.usr_np.exp(log_k_eq[18])*concentrations[6]*concentrations[2]),
            k_fwd[19]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[19])*concentrations[0]*concentrations[5]),
            k_fwd[20]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[20])*concentrations[7]*concentrations[4]),
            k_fwd[21]*(concentrations[6]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[21])*concentrations[7]*concentrations[5]),
            k_fwd[22]*(concentrations[6]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[22])*concentrations[7]*concentrations[5]),
            k_fwd[23]*(concentrations[6]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[23])*concentrations[5]*concentrations[4]),
            ])

    def get_net_production_rates(self, rho, temperature, mass_fractions):
        pressure = self.get_pressure(rho, temperature, mass_fractions)
        c = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(pressure, temperature, c)
        ones = self._pyro_zeros_like(r_net[0]) + 1.0
        return self._pyro_make_array([
            r_net[4] + r_net[11] + r_net[19] + -1*(r_net[1] + r_net[2]) * ones,
            r_net[1] + r_net[2] + -1*(r_net[0] + 2.0*r_net[4] + r_net[5] + r_net[7] + r_net[9] + r_net[10] + r_net[11] + r_net[12] + r_net[19] + r_net[20]) * ones,
            r_net[6] + r_net[11] + r_net[13] + r_net[14] + r_net[15] + r_net[17] + r_net[18] + -1*(r_net[0] + r_net[9]) * ones,
            r_net[0] + r_net[12] + -1*(r_net[1] + r_net[3] + 2.0*r_net[6] + r_net[7] + r_net[8] + r_net[13] + r_net[23]) * ones,
            r_net[0] + r_net[1] + 2.0*r_net[3] + r_net[7] + 2.0*r_net[10] + r_net[13] + r_net[20] + r_net[23] + -1*(r_net[2] + r_net[5] + r_net[8] + r_net[14] + r_net[15] + 2.0*r_net[16] + r_net[21] + r_net[22]) * ones,
            r_net[8] + r_net[9] + r_net[19] + r_net[21] + r_net[22] + r_net[23] + -1*(r_net[10] + r_net[11] + r_net[12] + r_net[13] + r_net[14] + r_net[15] + 2.0*r_net[17] + 2.0*r_net[18]) * ones,
            r_net[16] + r_net[17] + r_net[18] + -1*(r_net[19] + r_net[20] + r_net[21] + r_net[22] + r_net[23]) * ones,
            r_net[2] + r_net[5] + r_net[12] + r_net[14] + r_net[15] + r_net[20] + r_net[21] + r_net[22] + -1*r_net[3] * ones,
            0.0 * ones,
            ])

    def get_species_viscosities(self, temperature):
        return self._pyro_make_array([
            self.usr_np.sqrt(temperature)*(-0.00018517475885498877 + 0.0003891974465431562*self.usr_np.log(temperature) + -6.981604708795209e-05*self.usr_np.log(temperature)**2 + 6.364547895891601e-06*self.usr_np.log(temperature)**3 + -2.112164423759375e-07*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(-0.005034401953285114 + 0.0027769431000660974*self.usr_np.log(temperature) + -0.000505779280871413*self.usr_np.log(temperature)**2 + 4.1797031261689865e-05*self.usr_np.log(temperature)**3 + -1.2898934783544183e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(-0.005801490698399093 + 0.003392952010391191*self.usr_np.log(temperature) + -0.0006367679782768594*self.usr_np.log(temperature)**2 + 5.437791327369793e-05*self.usr_np.log(temperature)**3 + -1.7323741579628686e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(-0.0042543922809776605 + 0.0026940981513205097*self.usr_np.log(temperature) + -0.0005124612565785647*self.usr_np.log(temperature)**2 + 4.4691911762473806e-05*self.usr_np.log(temperature)**3 + -1.4505298562275118e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(-0.004319875697335926 + 0.0027355655899804474*self.usr_np.log(temperature) + -0.0005203490373977456*self.usr_np.log(temperature)**2 + 4.537980767625896e-05*self.usr_np.log(temperature)**3 + -1.4728563471198529e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(-0.005846650259658423 + 0.0034193631919541157*self.usr_np.log(temperature) + -0.0006417246633802151*self.usr_np.log(temperature)**2 + 5.4801198052224935e-05*self.usr_np.log(temperature)**3 + -1.745859184650765e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(-0.005890786984573198 + 0.0034451761764641756*self.usr_np.log(temperature) + -0.0006465690826084607*self.usr_np.log(temperature)**2 + 5.521489569036132e-05*self.usr_np.log(temperature)**3 + -1.7590387837631472e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(0.014933228200044495 + -0.008193670414138366*self.usr_np.log(temperature) + 0.0016828272719709029*self.usr_np.log(temperature)**2 + -0.00014573852839895512*self.usr_np.log(temperature)**3 + 4.601166862340898e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(-0.0048688367635371915 + 0.0029108156329475674*self.usr_np.log(temperature) + -0.0005498402785074613*self.usr_np.log(temperature)**2 + 4.733955628229572e-05*self.usr_np.log(temperature)**3 + -1.5195759686983762e-06*self.usr_np.log(temperature)**4)**2,
            ])

    def get_species_thermal_conductivities(self, temperature):
        return self._pyro_make_array([
            self.usr_np.sqrt(temperature)*(0.9102936191111852 + -0.5309733215378966*self.usr_np.log(temperature) + 0.11697403601306908*self.usr_np.log(temperature)**2 + -0.011438089184321772*self.usr_np.log(temperature)**3 + 0.00042154652985483326*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.20516989434210456 + 0.10357993398009493*self.usr_np.log(temperature) + -0.01830280712649368*self.usr_np.log(temperature)**2 + 0.0014706843635638467*self.usr_np.log(temperature)**3 + -4.3565965915042504e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.22525706395543346 + 0.13195200219245115*self.usr_np.log(temperature) + -0.028797611711561522*self.usr_np.log(temperature)**2 + 0.002788535188549312*self.usr_np.log(temperature)**3 + -0.00010055703119723457*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(0.1271646965919318 + -0.07463336580700054*self.usr_np.log(temperature) + 0.016614227493876883*self.usr_np.log(temperature)**2 + -0.0016289880457345798*self.usr_np.log(temperature)**3 + 5.974310354536575e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.1278913120936802 + 0.08092719038705067*self.usr_np.log(temperature) + -0.01861178369306037*self.usr_np.log(temperature)**2 + 0.0018791563647526774*self.usr_np.log(temperature)**3 + -6.941158279594658e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.2814566848273447 + 0.16718681427736323*self.usr_np.log(temperature) + -0.03708019193185602*self.usr_np.log(temperature)**2 + 0.003649463556476896*self.usr_np.log(temperature)**3 + -0.00013352470954930407*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.08106927212613932 + 0.05019829068848131*self.usr_np.log(temperature) + -0.011718717624866674*self.usr_np.log(temperature)**2 + 0.0012313582331194906*self.usr_np.log(temperature)**3 + -4.763346805346269e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(0.04807838518765636 + -0.01822008919117064*self.usr_np.log(temperature) + 0.0018052743926564715*self.usr_np.log(temperature)**2 + 2.4733683869826875e-05*self.usr_np.log(temperature)**3 + -5.612207534116106e-06*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(0.047708693508663615 + -0.02518622216031755*self.usr_np.log(temperature) + 0.004947388165965617*self.usr_np.log(temperature)**2 + -0.00041578747402100006*self.usr_np.log(temperature)**3 + 1.2922051284570714e-05*self.usr_np.log(temperature)**4),
            ])

    def get_species_binary_mass_diffusivities(self, temperature):
        return self._pyro_make_array([
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.006098424587778366 + 0.004074726320134811*self.usr_np.log(temperature) + -0.0007655079217732526*self.usr_np.log(temperature)**2 + 7.019454149192331e-05*self.usr_np.log(temperature)**3 + -2.283978048238303e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.023318952783947036 + 0.012988306044059204*self.usr_np.log(temperature) + -0.002390997476497533*self.usr_np.log(temperature)**2 + 0.0002048164504882058*self.usr_np.log(temperature)**3 + -6.393432695486336e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.007361772346024044 + 0.004218710865312339*self.usr_np.log(temperature) + -0.0007832424157226389*self.usr_np.log(temperature)**2 + 6.805076239811621e-05*self.usr_np.log(temperature)**3 + -2.1468141590217994e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.008203316666561931 + 0.004854681011220871*self.usr_np.log(temperature) + -0.0009069544030357264*self.usr_np.log(temperature)**2 + 7.989721260227312e-05*self.usr_np.log(temperature)**3 + -2.544060256277509e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.008176066416372073 + 0.004838554451984105*self.usr_np.log(temperature) + -0.0009039416296176856*self.usr_np.log(temperature)**2 + 7.963180543573052e-05*self.usr_np.log(temperature)**3 + -2.5356092502652206e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.007355106588560635 + 0.004214891010240474*self.usr_np.log(temperature) + -0.0007825332245477616*self.usr_np.log(temperature)**2 + 6.798914545912692e-05*self.usr_np.log(temperature)**3 + -2.144870314273726e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0073488303893876564 + 0.004211294394045626*self.usr_np.log(temperature) + -0.0007818654797209279*self.usr_np.log(temperature)**2 + 6.793112951970256e-05*self.usr_np.log(temperature)**3 + -2.143040071146609e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.012714368881648919 + 0.005800464307553849*self.usr_np.log(temperature) + -0.0008628173793011402*self.usr_np.log(temperature)**2 + 5.7978658190212e-05*self.usr_np.log(temperature)**3 + -1.31486649900594e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.006631024777879834 + 0.003839737084698172*self.usr_np.log(temperature) + -0.0007137696048370785*self.usr_np.log(temperature)**2 + 6.226499117220601e-05*self.usr_np.log(temperature)**3 + -1.968810850186332e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.023318952783947036 + 0.012988306044059204*self.usr_np.log(temperature) + -0.002390997476497533*self.usr_np.log(temperature)**2 + 0.0002048164504882058*self.usr_np.log(temperature)**3 + -6.393432695486336e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.04954737163988291 + 0.024207866731390578*self.usr_np.log(temperature) + -0.00400398535250643*self.usr_np.log(temperature)**2 + 0.0003046592732154171*self.usr_np.log(temperature)**3 + -8.373772281600414e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.020655585733532118 + 0.010499328175680164*self.usr_np.log(temperature) + -0.0018228205967797553*self.usr_np.log(temperature)**2 + 0.00014586133392893895*self.usr_np.log(temperature)**3 + -4.26526643465218e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.02701875467937279 + 0.014131423473428176*self.usr_np.log(temperature) + -0.002519893219301516*self.usr_np.log(temperature)**2 + 0.00020732039220444529*self.usr_np.log(temperature)**3 + -6.247265305087795e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.02697125585226038 + 0.014106580506072478*self.usr_np.log(temperature) + -0.002515463260414055*self.usr_np.log(temperature)**2 + 0.00020695592405666112*self.usr_np.log(temperature)**3 + -6.236282645875639e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.02064595090271774 + 0.010494430747355524*self.usr_np.log(temperature) + -0.0018219703392136429*self.usr_np.log(temperature)**2 + 0.000145793296677777*self.usr_np.log(temperature)**3 + -4.2632768943412685e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.020636883019349548 + 0.010489821500998121*self.usr_np.log(temperature) + -0.001821170113804225*self.usr_np.log(temperature)**2 + 0.00014572926297870152*self.usr_np.log(temperature)**3 + -4.261404425608901e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.015491646067670905 + -0.012297762819736099*self.usr_np.log(temperature) + 0.003322023627052121*self.usr_np.log(temperature)**2 + -0.00034981378874157794*self.usr_np.log(temperature)**3 + 1.3233960398106484e-05*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.01943052686236605 + 0.009973597438006639*self.usr_np.log(temperature) + -0.0017481642351558954*self.usr_np.log(temperature)**2 + 0.0001412844368085822*self.usr_np.log(temperature)**3 + -4.177030598880499e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.007361772346024044 + 0.004218710865312339*self.usr_np.log(temperature) + -0.0007832424157226389*self.usr_np.log(temperature)**2 + 6.805076239811621e-05*self.usr_np.log(temperature)**3 + -2.1468141590217994e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.020655585733532118 + 0.010499328175680164*self.usr_np.log(temperature) + -0.0018228205967797553*self.usr_np.log(temperature)**2 + 0.00014586133392893895*self.usr_np.log(temperature)**3 + -4.26526643465218e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0031656225297819014 + 0.0016565295961653466*self.usr_np.log(temperature) + -0.00029551028674573*self.usr_np.log(temperature)**2 + 2.4323268961005906e-05*self.usr_np.log(temperature)**3 + -7.332687938297444e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004477615809772843 + 0.0024017439135353828*self.usr_np.log(temperature) + -0.00043555046112226373*self.usr_np.log(temperature)**2 + 3.652320160561789e-05*self.usr_np.log(temperature)**3 + -1.1207561496122619e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004388261916822556 + 0.002353815467312997*self.usr_np.log(temperature) + -0.00042685875309501843*self.usr_np.log(temperature)**2 + 3.579435608045406e-05*self.usr_np.log(temperature)**3 + -1.0983906923544268e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0031413601016297026 + 0.0016438333792505788*self.usr_np.log(temperature) + -0.0002932453934956317*self.usr_np.log(temperature)**2 + 2.4136846998186314e-05*self.usr_np.log(temperature)**3 + -7.276487676714953e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0031183633828317267 + 0.0016317994917782956*self.usr_np.log(temperature) + -0.0002910986539833047*self.usr_np.log(temperature)**2 + 2.396015019645834e-05*self.usr_np.log(temperature)**3 + -7.223219240273291e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0017962300955805912 + -0.0016252040565704286*self.usr_np.log(temperature) + 0.00047039226523126206*self.usr_np.log(temperature)**2 + -5.1041731991841654e-05*self.usr_np.log(temperature)**3 + 1.970740354557153e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0030766151054803736 + 0.0016239301376489812*self.usr_np.log(temperature) + -0.0002916902121622096*self.usr_np.log(temperature)**2 + 2.4185952795912036e-05*self.usr_np.log(temperature)**3 + -7.345781692661095e-07*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.008203316666561931 + 0.004854681011220871*self.usr_np.log(temperature) + -0.0009069544030357264*self.usr_np.log(temperature)**2 + 7.989721260227312e-05*self.usr_np.log(temperature)**3 + -2.544060256277509e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.02701875467937279 + 0.014131423473428176*self.usr_np.log(temperature) + -0.002519893219301516*self.usr_np.log(temperature)**2 + 0.00020732039220444529*self.usr_np.log(temperature)**3 + -6.247265305087795e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004477615809772843 + 0.0024017439135353828*self.usr_np.log(temperature) + -0.00043555046112226373*self.usr_np.log(temperature)**2 + 3.652320160561789e-05*self.usr_np.log(temperature)**3 + -1.1207561496122619e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.005938290490168799 + 0.0032649962117399824*self.usr_np.log(temperature) + -0.0005991311922261666*self.usr_np.log(temperature)**2 + 5.101403674075905e-05*self.usr_np.log(temperature)**3 + -1.5859623241789838e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.005849638561391066 + 0.003216253528622326*self.usr_np.log(temperature) + -0.0005901868443753489*self.usr_np.log(temperature)**2 + 5.025245514425346e-05*self.usr_np.log(temperature)**3 + -1.562285708956096e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004454766518546262 + 0.002389487804824088*self.usr_np.log(temperature) + -0.0004333278453926392*self.usr_np.log(temperature)**2 + 3.63368235630474e-05*self.usr_np.log(temperature)**3 + -1.115036926538166e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.00443316399372883 + 0.0023779004479133324*self.usr_np.log(temperature) + -0.0004312265061878023*self.usr_np.log(temperature)**2 + 3.616061519623593e-05*self.usr_np.log(temperature)**3 + -1.1096297715783332e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003140903435972856 + 0.0009642154990053999*self.usr_np.log(temperature) + -1.5256977763046791e-05*self.usr_np.log(temperature)**2 + -1.0009613901224984e-05*self.usr_np.log(temperature)**3 + 6.865263141570081e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004199722246741017 + 0.0022703026942462834*self.usr_np.log(temperature) + -0.00041316736654270595*self.usr_np.log(temperature)**2 + 3.480896229542892e-05*self.usr_np.log(temperature)**3 + -1.0723125191057284e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.008176066416372073 + 0.004838554451984105*self.usr_np.log(temperature) + -0.0009039416296176856*self.usr_np.log(temperature)**2 + 7.963180543573052e-05*self.usr_np.log(temperature)**3 + -2.5356092502652206e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.02697125585226038 + 0.014106580506072478*self.usr_np.log(temperature) + -0.002515463260414055*self.usr_np.log(temperature)**2 + 0.00020695592405666112*self.usr_np.log(temperature)**3 + -6.236282645875639e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004388261916822556 + 0.002353815467312997*self.usr_np.log(temperature) + -0.00042685875309501843*self.usr_np.log(temperature)**2 + 3.579435608045406e-05*self.usr_np.log(temperature)**3 + -1.0983906923544268e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.005849638561391066 + 0.003216253528622326*self.usr_np.log(temperature) + -0.0005901868443753489*self.usr_np.log(temperature)**2 + 5.025245514425346e-05*self.usr_np.log(temperature)**3 + -1.562285708956096e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.005759622266468185 + 0.003166760688485024*self.usr_np.log(temperature) + -0.0005811048417035733*self.usr_np.log(temperature)**2 + 4.9479152695654574e-05*self.usr_np.log(temperature)**3 + -1.5382447071644631e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004364944907493674 + 0.002341308479751633*self.usr_np.log(temperature) + -0.000424590641091519*self.usr_np.log(temperature)**2 + 3.560416293554359e-05*self.usr_np.log(temperature)**3 + -1.0925543985082616e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004342895613060662 + 0.002329481480528768*self.usr_np.log(temperature) + -0.0004224458433317861*self.usr_np.log(temperature)**2 + 3.542431033986447e-05*self.usr_np.log(temperature)**3 + -1.0870354162238588e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003091211825554596 + 0.0009489608368045379*self.usr_np.log(temperature) + -1.50156001453656e-05*self.usr_np.log(temperature)**2 + -9.851253784347894e-06*self.usr_np.log(temperature)**3 + 6.756649174683129e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004119743857367522 + 0.0022270676557820525*self.usr_np.log(temperature) + -0.0004052991175068533*self.usr_np.log(temperature)**2 + 3.414606970950956e-05*self.usr_np.log(temperature)**3 + -1.0518916857384942e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.007355106588560635 + 0.004214891010240474*self.usr_np.log(temperature) + -0.0007825332245477616*self.usr_np.log(temperature)**2 + 6.798914545912692e-05*self.usr_np.log(temperature)**3 + -2.144870314273726e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.02064595090271774 + 0.010494430747355524*self.usr_np.log(temperature) + -0.0018219703392136429*self.usr_np.log(temperature)**2 + 0.000145793296677777*self.usr_np.log(temperature)**3 + -4.2632768943412685e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0031413601016297026 + 0.0016438333792505788*self.usr_np.log(temperature) + -0.0002932453934956317*self.usr_np.log(temperature)**2 + 2.4136846998186314e-05*self.usr_np.log(temperature)**3 + -7.276487676714953e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004454766518546262 + 0.002389487804824088*self.usr_np.log(temperature) + -0.0004333278453926392*self.usr_np.log(temperature)**2 + 3.63368235630474e-05*self.usr_np.log(temperature)**3 + -1.115036926538166e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004364944907493674 + 0.002341308479751633*self.usr_np.log(temperature) + -0.000424590641091519*self.usr_np.log(temperature)**2 + 3.560416293554359e-05*self.usr_np.log(temperature)**3 + -1.0925543985082616e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0031169088172688236 + 0.001631038336308339*self.usr_np.log(temperature) + -0.0002909628705528547*self.usr_np.log(temperature)**2 + 2.3948973946265063e-05*self.usr_np.log(temperature)**3 + -7.219849958162259e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003093730349327849 + 0.001618909341844277*self.usr_np.log(temperature) + -0.0002887991647908075*self.usr_np.log(temperature)**2 + 2.3770880662998656e-05*self.usr_np.log(temperature)**3 + -7.166160527178143e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-9.806367281211873e-05 + -0.0005259982739891355*self.usr_np.log(temperature) + 0.00023777270488714111*self.usr_np.log(temperature)**2 + -2.961075865641527e-05*self.usr_np.log(temperature)**3 + 1.240955619245755e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003054605917266221 + 0.0016123130250687598*self.usr_np.log(temperature) + -0.0002896035472529688*self.usr_np.log(temperature)**2 + 2.40129336924516e-05*self.usr_np.log(temperature)**3 + -7.293232158085036e-07*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.0073488303893876564 + 0.004211294394045626*self.usr_np.log(temperature) + -0.0007818654797209279*self.usr_np.log(temperature)**2 + 6.793112951970256e-05*self.usr_np.log(temperature)**3 + -2.143040071146609e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.020636883019349548 + 0.010489821500998121*self.usr_np.log(temperature) + -0.001821170113804225*self.usr_np.log(temperature)**2 + 0.00014572926297870152*self.usr_np.log(temperature)**3 + -4.261404425608901e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0031183633828317267 + 0.0016317994917782956*self.usr_np.log(temperature) + -0.0002910986539833047*self.usr_np.log(temperature)**2 + 2.396015019645834e-05*self.usr_np.log(temperature)**3 + -7.223219240273291e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.00443316399372883 + 0.0023779004479133324*self.usr_np.log(temperature) + -0.0004312265061878023*self.usr_np.log(temperature)**2 + 3.616061519623593e-05*self.usr_np.log(temperature)**3 + -1.1096297715783332e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004342895613060662 + 0.002329481480528768*self.usr_np.log(temperature) + -0.0004224458433317861*self.usr_np.log(temperature)**2 + 3.542431033986447e-05*self.usr_np.log(temperature)**3 + -1.0870354162238588e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003093730349327849 + 0.001618909341844277*self.usr_np.log(temperature) + -0.0002887991647908075*self.usr_np.log(temperature)**2 + 2.3770880662998656e-05*self.usr_np.log(temperature)**3 + -7.166160527178143e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003070376910663961 + 0.0016066887874492553*self.usr_np.log(temperature) + -0.00028661912554372776*self.usr_np.log(temperature)**2 + 2.3591442980684572e-05*self.usr_np.log(temperature)**3 + -7.112065802865576e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-9.754926582450284e-05 + -0.0005232390750063918*self.usr_np.log(temperature) + 0.00023652543424376865*self.usr_np.log(temperature)**2 + -2.945543119769758e-05*self.usr_np.log(temperature)**3 + 1.2344460095136146e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0030337554543624842 + 0.0016013075226149597*self.usr_np.log(temperature) + -0.0002876267397098679*self.usr_np.log(temperature)**2 + 2.3849023585328153e-05*self.usr_np.log(temperature)**3 + -7.243449216959913e-07*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.012714368881648919 + 0.005800464307553849*self.usr_np.log(temperature) + -0.0008628173793011402*self.usr_np.log(temperature)**2 + 5.7978658190212e-05*self.usr_np.log(temperature)**3 + -1.31486649900594e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.015491646067670905 + -0.012297762819736099*self.usr_np.log(temperature) + 0.003322023627052121*self.usr_np.log(temperature)**2 + -0.00034981378874157794*self.usr_np.log(temperature)**3 + 1.3233960398106484e-05*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0017962300955805912 + -0.0016252040565704286*self.usr_np.log(temperature) + 0.00047039226523126206*self.usr_np.log(temperature)**2 + -5.1041731991841654e-05*self.usr_np.log(temperature)**3 + 1.970740354557153e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003140903435972856 + 0.0009642154990053999*self.usr_np.log(temperature) + -1.5256977763046791e-05*self.usr_np.log(temperature)**2 + -1.0009613901224984e-05*self.usr_np.log(temperature)**3 + 6.865263141570081e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003091211825554596 + 0.0009489608368045379*self.usr_np.log(temperature) + -1.50156001453656e-05*self.usr_np.log(temperature)**2 + -9.851253784347894e-06*self.usr_np.log(temperature)**3 + 6.756649174683129e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-9.806367281211873e-05 + -0.0005259982739891355*self.usr_np.log(temperature) + 0.00023777270488714111*self.usr_np.log(temperature)**2 + -2.961075865641527e-05*self.usr_np.log(temperature)**3 + 1.240955619245755e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-9.754926582450284e-05 + -0.0005232390750063918*self.usr_np.log(temperature) + 0.00023652543424376865*self.usr_np.log(temperature)**2 + -2.945543119769758e-05*self.usr_np.log(temperature)**3 + 1.2344460095136146e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.015418678757900513 + -0.008276025270357978*self.usr_np.log(temperature) + 0.0016022988850514965*self.usr_np.log(temperature)**2 + -0.00012924880751471977*self.usr_np.log(temperature)**3 + 3.778041611680929e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0009009636432401638 + -0.0010975268417082404*self.usr_np.log(temperature) + 0.0003568805024623022*self.usr_np.log(temperature)**2 + -4.047906132941797e-05*self.usr_np.log(temperature)**3 + 1.6078992712038646e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.006631024777879834 + 0.003839737084698172*self.usr_np.log(temperature) + -0.0007137696048370785*self.usr_np.log(temperature)**2 + 6.226499117220601e-05*self.usr_np.log(temperature)**3 + -1.968810850186332e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.01943052686236605 + 0.009973597438006639*self.usr_np.log(temperature) + -0.0017481642351558954*self.usr_np.log(temperature)**2 + 0.0001412844368085822*self.usr_np.log(temperature)**3 + -4.177030598880499e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0030766151054803736 + 0.0016239301376489812*self.usr_np.log(temperature) + -0.0002916902121622096*self.usr_np.log(temperature)**2 + 2.4185952795912036e-05*self.usr_np.log(temperature)**3 + -7.345781692661095e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004199722246741017 + 0.0022703026942462834*self.usr_np.log(temperature) + -0.00041316736654270595*self.usr_np.log(temperature)**2 + 3.480896229542892e-05*self.usr_np.log(temperature)**3 + -1.0723125191057284e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004119743857367522 + 0.0022270676557820525*self.usr_np.log(temperature) + -0.0004052991175068533*self.usr_np.log(temperature)**2 + 3.414606970950956e-05*self.usr_np.log(temperature)**3 + -1.0518916857384942e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003054605917266221 + 0.0016123130250687598*self.usr_np.log(temperature) + -0.0002896035472529688*self.usr_np.log(temperature)**2 + 2.40129336924516e-05*self.usr_np.log(temperature)**3 + -7.293232158085036e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0030337554543624842 + 0.0016013075226149597*self.usr_np.log(temperature) + -0.0002876267397098679*self.usr_np.log(temperature)**2 + 2.3849023585328153e-05*self.usr_np.log(temperature)**3 + -7.243449216959913e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0009009636432401638 + -0.0010975268417082404*self.usr_np.log(temperature) + 0.0003568805024623022*self.usr_np.log(temperature)**2 + -4.047906132941797e-05*self.usr_np.log(temperature)**3 + 1.6078992712038646e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0029800170762017252 + 0.001585969492462742*self.usr_np.log(temperature) + -0.0002866086744964936*self.usr_np.log(temperature)**2 + 2.3921390316491723e-05*self.usr_np.log(temperature)**3 + -7.312450753174033e-07*self.usr_np.log(temperature)**4),
            ]),
        ])

    def get_mixture_viscosity_mixavg(self, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fractions = self.get_mole_fractions(mmw, mass_fractions)
        viscosities = self.get_species_viscosities(temperature)
        mix_rule_f = self._pyro_make_array([
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[0])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[1])*self.usr_np.sqrt(0.5)))**2) / self.usr_np.sqrt(24.0) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[2])*self.usr_np.sqrt(15.87202380952381)))**2) / self.usr_np.sqrt(8.504031501968873) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[3])*self.usr_np.sqrt(7.936011904761905)))**2) / self.usr_np.sqrt(9.008063003937746) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[4])*self.usr_np.sqrt(8.436011904761905)))**2) / self.usr_np.sqrt(8.948315399541364) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[5])*self.usr_np.sqrt(16.37202380952381)))**2) / self.usr_np.sqrt(8.488638429376477) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[6])*self.usr_np.sqrt(16.87202380952381)))**2) / self.usr_np.sqrt(8.474157699770682) + (mole_fractions[7]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[7])*self.usr_np.sqrt(8.936011904761905)))**2) / self.usr_np.sqrt(8.895253955037468) + (mole_fractions[8]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[8])*self.usr_np.sqrt(13.895833333333332)))**2) / self.usr_np.sqrt(8.575712143928037),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[0])*self.usr_np.sqrt(2.0)))**2) / self.usr_np.sqrt(12.0) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[1])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[2])*self.usr_np.sqrt(31.74404761904762)))**2) / self.usr_np.sqrt(8.252015750984437) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[3])*self.usr_np.sqrt(15.87202380952381)))**2) / self.usr_np.sqrt(8.504031501968873) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[4])*self.usr_np.sqrt(16.87202380952381)))**2) / self.usr_np.sqrt(8.474157699770682) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[5])*self.usr_np.sqrt(32.74404761904762)))**2) / self.usr_np.sqrt(8.244319214688238) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[6])*self.usr_np.sqrt(33.74404761904762)))**2) / self.usr_np.sqrt(8.237078849885341) + (mole_fractions[7]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[7])*self.usr_np.sqrt(17.87202380952381)))**2) / self.usr_np.sqrt(8.447626977518734) + (mole_fractions[8]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[8])*self.usr_np.sqrt(27.791666666666664)))**2) / self.usr_np.sqrt(8.287856071964018),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[0])*self.usr_np.sqrt(0.06300393774610913)))**2) / self.usr_np.sqrt(134.97619047619048) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[1])*self.usr_np.sqrt(0.031501968873054564)))**2) / self.usr_np.sqrt(261.95238095238096) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[2])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[3])*self.usr_np.sqrt(0.5)))**2) / self.usr_np.sqrt(24.0) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[4])*self.usr_np.sqrt(0.5315019688730546)))**2) / self.usr_np.sqrt(23.051684600458636) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[5])*self.usr_np.sqrt(1.0315019688730545)))**2) / self.usr_np.sqrt(15.75568078531176) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[6])*self.usr_np.sqrt(1.0630039377461091)))**2) / self.usr_np.sqrt(15.525842300229318) + (mole_fractions[7]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[7])*self.usr_np.sqrt(0.5630039377461091)))**2) / self.usr_np.sqrt(22.209492089925064) + (mole_fractions[8]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[8])*self.usr_np.sqrt(0.8754922182636414)))**2) / self.usr_np.sqrt(17.137716855857786),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[0])*self.usr_np.sqrt(0.12600787549221826)))**2) / self.usr_np.sqrt(71.48809523809524) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[1])*self.usr_np.sqrt(0.06300393774610913)))**2) / self.usr_np.sqrt(134.97619047619048) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[2])*self.usr_np.sqrt(2.0)))**2) / self.usr_np.sqrt(12.0) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[3])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[4])*self.usr_np.sqrt(1.0630039377461091)))**2) / self.usr_np.sqrt(15.525842300229318) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[5])*self.usr_np.sqrt(2.063003937746109)))**2) / self.usr_np.sqrt(11.87784039265588) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[6])*self.usr_np.sqrt(2.1260078754922183)))**2) / self.usr_np.sqrt(11.762921150114659) + (mole_fractions[7]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[7])*self.usr_np.sqrt(1.1260078754922183)))**2) / self.usr_np.sqrt(15.104746044962532) + (mole_fractions[8]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[8])*self.usr_np.sqrt(1.7509844365272829)))**2) / self.usr_np.sqrt(12.568858427928893),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[0])*self.usr_np.sqrt(0.11853942494267065)))**2) / self.usr_np.sqrt(75.48809523809524) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[1])*self.usr_np.sqrt(0.05926971247133533)))**2) / self.usr_np.sqrt(142.97619047619048) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[2])*self.usr_np.sqrt(1.8814605750573292)))**2) / self.usr_np.sqrt(12.252015750984437) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[3])*self.usr_np.sqrt(0.9407302875286646)))**2) / self.usr_np.sqrt(16.504031501968875) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[4])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[5])*self.usr_np.sqrt(1.9407302875286645)))**2) / self.usr_np.sqrt(12.12215960734412) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[6])*self.usr_np.sqrt(2.0)))**2) / self.usr_np.sqrt(12.0) + (mole_fractions[7]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[7])*self.usr_np.sqrt(1.0592697124713353)))**2) / self.usr_np.sqrt(15.552373022481266) + (mole_fractions[8]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[8])*self.usr_np.sqrt(1.6472040924325275)))**2) / self.usr_np.sqrt(12.856714499892911),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[0])*self.usr_np.sqrt(0.061079803672059625)))**2) / self.usr_np.sqrt(138.97619047619048) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[1])*self.usr_np.sqrt(0.030539901836029813)))**2) / self.usr_np.sqrt(269.95238095238096) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[2])*self.usr_np.sqrt(0.9694600981639702)))**2) / self.usr_np.sqrt(16.252015750984434) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[3])*self.usr_np.sqrt(0.4847300490819851)))**2) / self.usr_np.sqrt(24.50403150196887) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[4])*self.usr_np.sqrt(0.515269950918015)))**2) / self.usr_np.sqrt(23.525842300229314) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[5])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[6])*self.usr_np.sqrt(1.03053990183603)))**2) / self.usr_np.sqrt(15.762921150114657) + (mole_fractions[7]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[7])*self.usr_np.sqrt(0.5458098527540447)))**2) / self.usr_np.sqrt(22.657119067443794) + (mole_fractions[8]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[8])*self.usr_np.sqrt(0.8487547718596619)))**2) / self.usr_np.sqrt(17.425572927821804),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[0])*self.usr_np.sqrt(0.05926971247133533)))**2) / self.usr_np.sqrt(142.97619047619048) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[1])*self.usr_np.sqrt(0.029634856235667664)))**2) / self.usr_np.sqrt(277.95238095238096) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[2])*self.usr_np.sqrt(0.9407302875286646)))**2) / self.usr_np.sqrt(16.504031501968875) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[3])*self.usr_np.sqrt(0.4703651437643323)))**2) / self.usr_np.sqrt(25.008063003937746) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[4])*self.usr_np.sqrt(0.5)))**2) / self.usr_np.sqrt(24.0) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[5])*self.usr_np.sqrt(0.9703651437643322)))**2) / self.usr_np.sqrt(16.24431921468824) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[6])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[7]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[7])*self.usr_np.sqrt(0.5296348562356676)))**2) / self.usr_np.sqrt(23.104746044962532) + (mole_fractions[8]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[8])*self.usr_np.sqrt(0.8236020462162638)))**2) / self.usr_np.sqrt(17.713428999785823),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[7] / viscosities[0])*self.usr_np.sqrt(0.11190674437968359)))**2) / self.usr_np.sqrt(79.48809523809524) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[7] / viscosities[1])*self.usr_np.sqrt(0.055953372189841796)))**2) / self.usr_np.sqrt(150.97619047619048) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[7] / viscosities[2])*self.usr_np.sqrt(1.7761865112406328)))**2) / self.usr_np.sqrt(12.504031501968873) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[7] / viscosities[3])*self.usr_np.sqrt(0.8880932556203164)))**2) / self.usr_np.sqrt(17.008063003937746) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[7] / viscosities[4])*self.usr_np.sqrt(0.9440466278101582)))**2) / self.usr_np.sqrt(16.474157699770682) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[7] / viscosities[5])*self.usr_np.sqrt(1.8321398834304745)))**2) / self.usr_np.sqrt(12.366478822032358) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[7] / viscosities[6])*self.usr_np.sqrt(1.8880932556203165)))**2) / self.usr_np.sqrt(12.237078849885341) + (mole_fractions[7]*(1 + self.usr_np.sqrt((viscosities[7] / viscosities[7])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[8]*(1 + self.usr_np.sqrt((viscosities[7] / viscosities[8])*self.usr_np.sqrt(1.55503746877602)))**2) / self.usr_np.sqrt(13.144570571856928),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[8] / viscosities[0])*self.usr_np.sqrt(0.0719640179910045)))**2) / self.usr_np.sqrt(119.16666666666666) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[8] / viscosities[1])*self.usr_np.sqrt(0.03598200899550225)))**2) / self.usr_np.sqrt(230.33333333333331) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[8] / viscosities[2])*self.usr_np.sqrt(1.1422146069822232)))**2) / self.usr_np.sqrt(15.00393774610913) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[8] / viscosities[3])*self.usr_np.sqrt(0.5711073034911116)))**2) / self.usr_np.sqrt(22.00787549221826) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[8] / viscosities[4])*self.usr_np.sqrt(0.6070893124866139)))**2) / self.usr_np.sqrt(21.177632739460222) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[8] / viscosities[5])*self.usr_np.sqrt(1.1781966159777255)))**2) / self.usr_np.sqrt(14.790038174877296) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[8] / viscosities[6])*self.usr_np.sqrt(1.2141786249732278)))**2) / self.usr_np.sqrt(14.588816369730111) + (mole_fractions[7]*(1 + self.usr_np.sqrt((viscosities[8] / viscosities[7])*self.usr_np.sqrt(0.6430713214821161)))**2) / self.usr_np.sqrt(20.440299750208162) + (mole_fractions[8]*(1 + self.usr_np.sqrt((viscosities[8] / viscosities[8])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0),
            ])
        return sum(mole_fractions*viscosities/mix_rule_f)

    def get_mixture_thermal_conductivity_mixavg(self, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fractions = self.get_mole_fractions(mmw, mass_fractions)
        conductivities = self.get_species_thermal_conductivities(temperature)
        return 0.5*(sum(mole_fractions*conductivities)
            + 1/sum(mole_fractions/conductivities))

    def get_species_mass_diffusivities_mixavg(self, pressure, temperature,
            mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fractions = self.get_mole_fractions(mmw, mass_fractions)
        bdiff_ij = self.get_species_binary_mass_diffusivities(temperature)
        zeros = self._pyro_zeros_like(temperature)

        x_sum = self._pyro_make_array([
            mole_fractions[0] / bdiff_ij[0][0] + mole_fractions[1] / bdiff_ij[1][0] + mole_fractions[2] / bdiff_ij[2][0] + mole_fractions[3] / bdiff_ij[3][0] + mole_fractions[4] / bdiff_ij[4][0] + mole_fractions[5] / bdiff_ij[5][0] + mole_fractions[6] / bdiff_ij[6][0] + mole_fractions[7] / bdiff_ij[7][0] + mole_fractions[8] / bdiff_ij[8][0],
            mole_fractions[0] / bdiff_ij[0][1] + mole_fractions[1] / bdiff_ij[1][1] + mole_fractions[2] / bdiff_ij[2][1] + mole_fractions[3] / bdiff_ij[3][1] + mole_fractions[4] / bdiff_ij[4][1] + mole_fractions[5] / bdiff_ij[5][1] + mole_fractions[6] / bdiff_ij[6][1] + mole_fractions[7] / bdiff_ij[7][1] + mole_fractions[8] / bdiff_ij[8][1],
            mole_fractions[0] / bdiff_ij[0][2] + mole_fractions[1] / bdiff_ij[1][2] + mole_fractions[2] / bdiff_ij[2][2] + mole_fractions[3] / bdiff_ij[3][2] + mole_fractions[4] / bdiff_ij[4][2] + mole_fractions[5] / bdiff_ij[5][2] + mole_fractions[6] / bdiff_ij[6][2] + mole_fractions[7] / bdiff_ij[7][2] + mole_fractions[8] / bdiff_ij[8][2],
            mole_fractions[0] / bdiff_ij[0][3] + mole_fractions[1] / bdiff_ij[1][3] + mole_fractions[2] / bdiff_ij[2][3] + mole_fractions[3] / bdiff_ij[3][3] + mole_fractions[4] / bdiff_ij[4][3] + mole_fractions[5] / bdiff_ij[5][3] + mole_fractions[6] / bdiff_ij[6][3] + mole_fractions[7] / bdiff_ij[7][3] + mole_fractions[8] / bdiff_ij[8][3],
            mole_fractions[0] / bdiff_ij[0][4] + mole_fractions[1] / bdiff_ij[1][4] + mole_fractions[2] / bdiff_ij[2][4] + mole_fractions[3] / bdiff_ij[3][4] + mole_fractions[4] / bdiff_ij[4][4] + mole_fractions[5] / bdiff_ij[5][4] + mole_fractions[6] / bdiff_ij[6][4] + mole_fractions[7] / bdiff_ij[7][4] + mole_fractions[8] / bdiff_ij[8][4],
            mole_fractions[0] / bdiff_ij[0][5] + mole_fractions[1] / bdiff_ij[1][5] + mole_fractions[2] / bdiff_ij[2][5] + mole_fractions[3] / bdiff_ij[3][5] + mole_fractions[4] / bdiff_ij[4][5] + mole_fractions[5] / bdiff_ij[5][5] + mole_fractions[6] / bdiff_ij[6][5] + mole_fractions[7] / bdiff_ij[7][5] + mole_fractions[8] / bdiff_ij[8][5],
            mole_fractions[0] / bdiff_ij[0][6] + mole_fractions[1] / bdiff_ij[1][6] + mole_fractions[2] / bdiff_ij[2][6] + mole_fractions[3] / bdiff_ij[3][6] + mole_fractions[4] / bdiff_ij[4][6] + mole_fractions[5] / bdiff_ij[5][6] + mole_fractions[6] / bdiff_ij[6][6] + mole_fractions[7] / bdiff_ij[7][6] + mole_fractions[8] / bdiff_ij[8][6],
            mole_fractions[0] / bdiff_ij[0][7] + mole_fractions[1] / bdiff_ij[1][7] + mole_fractions[2] / bdiff_ij[2][7] + mole_fractions[3] / bdiff_ij[3][7] + mole_fractions[4] / bdiff_ij[4][7] + mole_fractions[5] / bdiff_ij[5][7] + mole_fractions[6] / bdiff_ij[6][7] + mole_fractions[7] / bdiff_ij[7][7] + mole_fractions[8] / bdiff_ij[8][7],
            mole_fractions[0] / bdiff_ij[0][8] + mole_fractions[1] / bdiff_ij[1][8] + mole_fractions[2] / bdiff_ij[2][8] + mole_fractions[3] / bdiff_ij[3][8] + mole_fractions[4] / bdiff_ij[4][8] + mole_fractions[5] / bdiff_ij[5][8] + mole_fractions[6] / bdiff_ij[6][8] + mole_fractions[7] / bdiff_ij[7][8] + mole_fractions[8] / bdiff_ij[8][8],
            ])
        denom = self._pyro_make_array([
            x_sum[0] - mole_fractions[0]/bdiff_ij[0][0],
            x_sum[1] - mole_fractions[1]/bdiff_ij[1][1],
            x_sum[2] - mole_fractions[2]/bdiff_ij[2][2],
            x_sum[3] - mole_fractions[3]/bdiff_ij[3][3],
            x_sum[4] - mole_fractions[4]/bdiff_ij[4][4],
            x_sum[5] - mole_fractions[5]/bdiff_ij[5][5],
            x_sum[6] - mole_fractions[6]/bdiff_ij[6][6],
            x_sum[7] - mole_fractions[7]/bdiff_ij[7][7],
            x_sum[8] - mole_fractions[8]/bdiff_ij[8][8],
            ])

        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(denom[0], zeros), (mmw - mole_fractions[0] * self.molecular_weights[0])/(pressure * mmw * denom[0]), bdiff_ij[0][0] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[1], zeros), (mmw - mole_fractions[1] * self.molecular_weights[1])/(pressure * mmw * denom[1]), bdiff_ij[1][1] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[2], zeros), (mmw - mole_fractions[2] * self.molecular_weights[2])/(pressure * mmw * denom[2]), bdiff_ij[2][2] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[3], zeros), (mmw - mole_fractions[3] * self.molecular_weights[3])/(pressure * mmw * denom[3]), bdiff_ij[3][3] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[4], zeros), (mmw - mole_fractions[4] * self.molecular_weights[4])/(pressure * mmw * denom[4]), bdiff_ij[4][4] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[5], zeros), (mmw - mole_fractions[5] * self.molecular_weights[5])/(pressure * mmw * denom[5]), bdiff_ij[5][5] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[6], zeros), (mmw - mole_fractions[6] * self.molecular_weights[6])/(pressure * mmw * denom[6]), bdiff_ij[6][6] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[7], zeros), (mmw - mole_fractions[7] * self.molecular_weights[7])/(pressure * mmw * denom[7]), bdiff_ij[7][7] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[8], zeros), (mmw - mole_fractions[8] * self.molecular_weights[8])/(pressure * mmw * denom[8]), bdiff_ij[8][8] / pressure),
            ])

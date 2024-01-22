"""
.. autoclass:: Thermochemistry
"""


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
        self.model_name = 'uiuc_const_gamma.yaml'
        self.num_elements = 4
        self.num_species = 7
        self.num_reactions = 3
        self.num_falloff = 0

        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['C2H4', 'O2', 'CO2', 'CO', 'H2O', 'H2', 'N2']
        self.species_indices = {'C2H4': 0, 'O2': 1, 'CO2': 2, 'CO': 3, 'H2O': 4, 'H2': 5, 'N2': 6}

        self.wts = np.array([28.054, 31.998, 44.009, 28.009999999999998, 18.015, 2.016, 28.014])
        self.iwts = 1/self.wts

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
            + self.iwts[0]*mass_fractions[0]
            + self.iwts[1]*mass_fractions[1]
            + self.iwts[2]*mass_fractions[2]
            + self.iwts[3]*mass_fractions[3]
            + self.iwts[4]*mass_fractions[4]
            + self.iwts[5]*mass_fractions[5]
            + self.iwts[6]*mass_fractions[6]
        )

    def get_density(self, p, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return p * mmw / rt

    def get_pressure(self, rho, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return rho * rt / mmw

    def get_mix_molecular_weight(self, mass_fractions):
        return 1/(
            + self.iwts[0]*mass_fractions[0]
            + self.iwts[1]*mass_fractions[1]
            + self.iwts[2]*mass_fractions[2]
            + self.iwts[3]*mass_fractions[3]
            + self.iwts[4]*mass_fractions[4]
            + self.iwts[5]*mass_fractions[5]
            + self.iwts[6]*mass_fractions[6]
        )

    def get_concentrations(self, rho, mass_fractions):
        return self._pyro_make_array([
            self.iwts[0] * rho * mass_fractions[0],
            self.iwts[1] * rho * mass_fractions[1],
            self.iwts[2] * rho * mass_fractions[2],
            self.iwts[3] * rho * mass_fractions[3],
            self.iwts[4] * rho * mass_fractions[4],
            self.iwts[5] * rho * mass_fractions[5],
            self.iwts[6] * rho * mass_fractions[6],
        ])

    def get_mole_fractions(self, mix_mol_weight, mass_fractions):
        return self._pyro_make_array([
            self.iwts[0] * mass_fractions[0] * mix_mol_weight,
            self.iwts[1] * mass_fractions[1] * mix_mol_weight,
            self.iwts[2] * mass_fractions[2] * mix_mol_weight,
            self.iwts[3] * mass_fractions[3] * mix_mol_weight,
            self.iwts[4] * mass_fractions[4] * mix_mol_weight,
            self.iwts[5] * mass_fractions[5] * mix_mol_weight,
            self.iwts[6] * mass_fractions[6] * mix_mol_weight,
        ])

    def get_mass_average_property(self, mass_fractions, spec_property):
        return sum([mass_fractions[i] * spec_property[i] * self.iwts[i]
                    for i in range(self.num_species)])

    def get_mixture_specific_heat_cp_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature)
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_specific_heat_cv_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature) - 1.0
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_enthalpy_mass(self, temperature, mass_fractions):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        hmix = self.get_mass_average_property(mass_fractions, h0_rt)
        return self.gas_constant * temperature * hmix

    def get_mixture_internal_energy_mass(self, temperature, mass_fractions):
        e0_rt = self.get_species_enthalpies_rt(temperature) - 1.0
        emix = self.get_mass_average_property(mass_fractions, e0_rt)
        return self.gas_constant * temperature * emix

    def get_species_viscosities(self, temperature):
        return self._pyro_make_array([
                self.usr_np.sqrt(temperature)*(0.00056437200093386 + -0.0008536047562476543*self.usr_np.log(temperature) + 0.0003406553640740371*self.usr_np.log(temperature)**2 + -4.2648038931529444e-05*self.usr_np.log(temperature)**3 + 1.7934368018117115e-06*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.006186428071747617 + 0.0036188245122442887*self.usr_np.log(temperature) + -0.0006861983404475868*self.usr_np.log(temperature)**2 + 5.91601270960357e-05*self.usr_np.log(temperature)**3 + -1.9049771784291988e-06*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.002706042231744705 + 0.0009960891348943866*self.usr_np.log(temperature) + -3.24575928842047e-05*self.usr_np.log(temperature)**2 + -9.152079934518157e-06*self.usr_np.log(temperature)**3 + 6.716178143188336e-07*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.005216081450795922 + 0.0031111996741948764*self.usr_np.log(temperature) + -0.000593952979220727*self.usr_np.log(temperature)**2 + 5.162313384558182e-05*self.usr_np.log(temperature)**3 + -1.6749040550312975e-06*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(0.009495196334902445 + -0.004974400618415688*self.usr_np.log(temperature) + 0.0009719845681883221*self.usr_np.log(temperature)**2 + -7.63468726043692e-05*self.usr_np.log(temperature)**3 + 2.074120177526231e-06*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.00032862351581375954 + 0.00047402944328358616*self.usr_np.log(temperature) + -8.852339013572501e-05*self.usr_np.log(temperature)**2 + 8.188000383358501e-06*self.usr_np.log(temperature)**3 + -2.775116846638978e-07*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.005232503458857763 + 0.0031249811202461084*self.usr_np.log(temperature) + -0.0005968857242794387*self.usr_np.log(temperature)**2 + 5.190839695047226e-05*self.usr_np.log(temperature)**3 + -1.6850989392760908e-06*self.usr_np.log(temperature)**4)**2,
                ])

    def get_mixture_viscosity_mixavg(self, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fracs = self.get_mole_fractions(mmw, mass_fractions)
        viscosities = self.get_species_viscosities(temperature)
        mix_rule_f = self._pyro_make_array([
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[0])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[1])*self.usr_np.sqrt(1.1405860126898126)))**2) / self.usr_np.sqrt(15.013938371148196) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[2])*self.usr_np.sqrt(1.5687246025522208)))**2) / self.usr_np.sqrt(13.099684155513645) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[3])*self.usr_np.sqrt(0.9984315962073145)))**2) / self.usr_np.sqrt(16.012566940378434) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[4])*self.usr_np.sqrt(0.642154416482498)))**2) / self.usr_np.sqrt(20.458062725506522) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[5])*self.usr_np.sqrt(0.07186141013759179)))**2) / self.usr_np.sqrt(119.32539682539682) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[6])*self.usr_np.sqrt(0.998574178370286)))**2) / self.usr_np.sqrt(16.01142285999857),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[0])*self.usr_np.sqrt(0.8767422963935245)))**2) / self.usr_np.sqrt(17.1246881015185) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[1])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[2])*self.usr_np.sqrt(1.3753672104506531)))**2) / self.usr_np.sqrt(13.816628416914721) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[3])*self.usr_np.sqrt(0.875367210450653)))**2) / self.usr_np.sqrt(17.139021777936453) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[4])*self.usr_np.sqrt(0.5630039377461091)))**2) / self.usr_np.sqrt(22.209492089925064) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[5])*self.usr_np.sqrt(0.06300393774610913)))**2) / self.usr_np.sqrt(134.97619047619048) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[6])*self.usr_np.sqrt(0.8754922182636414)))**2) / self.usr_np.sqrt(17.137716855857786),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[0])*self.usr_np.sqrt(0.6374605194392056)))**2) / self.usr_np.sqrt(20.549796820417768) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[1])*self.usr_np.sqrt(0.7270785521143402)))**2) / self.usr_np.sqrt(19.002937683605225) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[2])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[3])*self.usr_np.sqrt(0.6364607239428298)))**2) / self.usr_np.sqrt(20.569510888968225) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[4])*self.usr_np.sqrt(0.4093480878911132)))**2) / self.usr_np.sqrt(27.543269497640853) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[5])*self.usr_np.sqrt(0.04580881183394306)))**2) / self.usr_np.sqrt(182.63888888888889) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[6])*self.usr_np.sqrt(0.6365516144425004)))**2) / self.usr_np.sqrt(20.567716141929036),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[0])*self.usr_np.sqrt(1.0015708675473045)))**2) / self.usr_np.sqrt(15.987452769658516) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[1])*self.usr_np.sqrt(1.1423777222420566)))**2) / self.usr_np.sqrt(15.002937683605225) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[2])*self.usr_np.sqrt(1.5711888611210283)))**2) / self.usr_np.sqrt(13.091685791542638) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[3])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[4])*self.usr_np.sqrt(0.6431631560157087)))**2) / self.usr_np.sqrt(20.43852345267832) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[5])*self.usr_np.sqrt(0.07197429489468048)))**2) / self.usr_np.sqrt(119.15079365079364) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[6])*self.usr_np.sqrt(1.000142806140664)))**2) / self.usr_np.sqrt(15.998857714000142),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[0])*self.usr_np.sqrt(1.557257840688315)))**2) / self.usr_np.sqrt(13.137235331859983) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[1])*self.usr_np.sqrt(1.7761865112406328)))**2) / self.usr_np.sqrt(12.504031501968873) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[2])*self.usr_np.sqrt(2.4429086872051067)))**2) / self.usr_np.sqrt(11.274784703128905) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[3])*self.usr_np.sqrt(1.5548154315847902)))**2) / self.usr_np.sqrt(13.14530524812567) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[4])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[5])*self.usr_np.sqrt(0.11190674437968359)))**2) / self.usr_np.sqrt(79.48809523809524) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[6])*self.usr_np.sqrt(1.55503746877602)))**2) / self.usr_np.sqrt(13.144570571856928),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[0])*self.usr_np.sqrt(13.915674603174603)))**2) / self.usr_np.sqrt(8.574891281100735) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[1])*self.usr_np.sqrt(15.87202380952381)))**2) / self.usr_np.sqrt(8.504031501968873) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[2])*self.usr_np.sqrt(21.82986111111111)))**2) / self.usr_np.sqrt(8.366470494671544) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[3])*self.usr_np.sqrt(13.893849206349206)))**2) / self.usr_np.sqrt(8.575794359157443) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[4])*self.usr_np.sqrt(8.936011904761905)))**2) / self.usr_np.sqrt(8.895253955037468) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[5])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[6])*self.usr_np.sqrt(13.895833333333332)))**2) / self.usr_np.sqrt(8.575712143928037),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[0])*self.usr_np.sqrt(1.0014278574998214)))**2) / self.usr_np.sqrt(15.988593426962288) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[1])*self.usr_np.sqrt(1.1422146069822232)))**2) / self.usr_np.sqrt(15.00393774610913) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[2])*self.usr_np.sqrt(1.5709645177411296)))**2) / self.usr_np.sqrt(13.092412915540002) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[3])*self.usr_np.sqrt(0.9998572142500178)))**2) / self.usr_np.sqrt(16.00114244912531) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[4])*self.usr_np.sqrt(0.6430713214821161)))**2) / self.usr_np.sqrt(20.440299750208162) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[5])*self.usr_np.sqrt(0.0719640179910045)))**2) / self.usr_np.sqrt(119.16666666666666) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[6])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0),
            ])
        return sum(mole_fracs*viscosities/mix_rule_f)

    def get_species_thermal_conductivities(self, temperature):
        return self._pyro_make_array([
                self.usr_np.sqrt(temperature)*(0.0247682201963315 + -0.016958321557725024*self.usr_np.log(temperature) + 0.004182784447798461*self.usr_np.log(temperature)**2 + -0.00042688086846824847*self.usr_np.log(temperature)**3 + 1.5897516098414554e-05*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(-0.015481158007398325 + 0.00810774415832125*self.usr_np.log(temperature) + -0.0014727237664290268*self.usr_np.log(temperature)**2 + 0.00012258649045074455*self.usr_np.log(temperature)**3 + -3.7620170290222617e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(0.007084443377523098 + -0.005933304595613126*self.usr_np.log(temperature) + 0.0016646988442737785*self.usr_np.log(temperature)**2 + -0.00018167323848062043*self.usr_np.log(temperature)**3 + 7.099116964975215e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(-0.015877101278221794 + 0.00841461544097927*self.usr_np.log(temperature) + -0.0015331347580769235*self.usr_np.log(temperature)**2 + 0.00012861516067246787*self.usr_np.log(temperature)**3 + -3.971062243407754e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(0.017574727960951043 + -0.006060019898969308*self.usr_np.log(temperature) + 0.00021480200784464333*self.usr_np.log(temperature)**2 + 0.00011121870189995938*self.usr_np.log(temperature)**3 + -8.543309853978141e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(-0.021491312355572993 + 0.014941063472170163*self.usr_np.log(temperature) + -0.0028494128257026343*self.usr_np.log(temperature)**2 + 0.00026520446240242094*self.usr_np.log(temperature)**3 + -8.793827992991172e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(-0.014113069056641473 + 0.007522128253079056*self.usr_np.log(temperature) + -0.0013882229396441801*self.usr_np.log(temperature)**2 + 0.00011748749772469808*self.usr_np.log(temperature)**3 + -3.6691089537792376e-06*self.usr_np.log(temperature)**4),
                ])

    def get_mixture_thermal_conductivity_mixavg(self, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fracs = self.get_mole_fractions(mmw, mass_fractions)
        conductivities = self.get_species_thermal_conductivities(temperature)
        return 0.5*(sum(mole_fracs*conductivities)
            + 1/sum(mole_fracs/conductivities))

    def get_species_binary_mass_diffusivities(self, temperature):
        return self._pyro_make_array([
                0.002514025167404718 + -0.0017759952625726505*self.usr_np.log(temperature) + 0.00044874581158256305*self.usr_np.log(temperature)**2 + -4.63252893024973e-05*self.usr_np.log(temperature)**3 + 1.7399023936012455e-06*self.usr_np.log(temperature)**4,
                -0.001732870278755321 + 0.0006970694562179823*self.usr_np.log(temperature) + -7.498493379425965e-05*self.usr_np.log(temperature)**2 + 2.3790313073441086e-06*self.usr_np.log(temperature)**3 + 6.258149853836124e-08*self.usr_np.log(temperature)**4,
                0.0016279660308488376 + -0.0012518611559844848*self.usr_np.log(temperature) + 0.0003350193750336406*self.usr_np.log(temperature)**2 + -3.560522825082911e-05*self.usr_np.log(temperature)**3 + 1.3658158187700505e-06*self.usr_np.log(temperature)**4,
                -0.0019075474076651014 + 0.0008156938751059555*self.usr_np.log(temperature) + -0.0001033057352502026*self.usr_np.log(temperature)**2 + 5.235151668430256e-06*self.usr_np.log(temperature)**3 + -4.243764677111869e-08*self.usr_np.log(temperature)**4,
                0.011422240725275307 + -0.006973463254020516*self.usr_np.log(temperature) + 0.0015647848002851477*self.usr_np.log(temperature)**2 + -0.00015019828712517556*self.usr_np.log(temperature)**3 + 5.320779247518453e-06*self.usr_np.log(temperature)**4,
                -0.00905166214758596 + 0.004761002619202597*self.usr_np.log(temperature) + -0.0008507988153643708*self.usr_np.log(temperature)**2 + 7.018609346634951e-05*self.usr_np.log(temperature)**3 + -2.118223235603562e-06*self.usr_np.log(temperature)**4,
                -0.0019364269556070043 + 0.0008309848230287911*self.usr_np.log(temperature) + -0.00010613902008003644*self.usr_np.log(temperature)**2 + 5.474741525862431e-06*self.usr_np.log(temperature)**3 + -4.991875443999132e-08*self.usr_np.log(temperature)**4,
                -0.001732870278755321 + 0.0006970694562179823*self.usr_np.log(temperature) + -7.498493379425965e-05*self.usr_np.log(temperature)**2 + 2.3790313073441086e-06*self.usr_np.log(temperature)**3 + 6.258149853836124e-08*self.usr_np.log(temperature)**4,
                -0.003127289937114834 + 0.001633232188528128*self.usr_np.log(temperature) + -0.00029023244731756897*self.usr_np.log(temperature)**2 + 2.3795154190948053e-05*self.usr_np.log(temperature)**3 + -7.135754459081666e-07*self.usr_np.log(temperature)**4,
                -0.0019538762606351678 + 0.0008581597928151936*self.usr_np.log(temperature) + -0.00011552373820595307*self.usr_np.log(temperature)**2 + 6.588031407432268e-06*self.usr_np.log(temperature)**3 + -9.570226239447783e-08*self.usr_np.log(temperature)**4,
                -0.003009672869056897 + 0.0015847679011338385*self.usr_np.log(temperature) + -0.00028341591661372364*self.usr_np.log(temperature)**2 + 2.3400497455366053e-05*self.usr_np.log(temperature)**3 + -7.068324598269804e-07*self.usr_np.log(temperature)**4,
                0.0044168550074850646 + -0.0031775743855656643*self.usr_np.log(temperature) + 0.0008133764546254836*self.usr_np.log(temperature)**2 + -8.454231010204543e-05*self.usr_np.log(temperature)**3 + 3.1913607821347743e-06*self.usr_np.log(temperature)**4,
                -0.008078545001672823 + 0.004640788076930059*self.usr_np.log(temperature) + -0.0008759648030254944*self.usr_np.log(temperature)**2 + 7.705769878335738e-05*self.usr_np.log(temperature)**3 + -2.473269089069558e-06*self.usr_np.log(temperature)**4,
                -0.0030300534762843713 + 0.0015963119455998603*self.usr_np.log(temperature) + -0.0002855827123818112*self.usr_np.log(temperature)**2 + 2.3589003022680055e-05*self.usr_np.log(temperature)**3 + -7.128139183189626e-07*self.usr_np.log(temperature)**4,
                0.0016279660308488376 + -0.0012518611559844848*self.usr_np.log(temperature) + 0.0003350193750336406*self.usr_np.log(temperature)**2 + -3.560522825082911e-05*self.usr_np.log(temperature)**3 + 1.3658158187700505e-06*self.usr_np.log(temperature)**4,
                -0.0019538762606351678 + 0.0008581597928151936*self.usr_np.log(temperature) + -0.00011552373820595307*self.usr_np.log(temperature)**2 + 6.588031407432268e-06*self.usr_np.log(temperature)**3 + -9.570226239447783e-08*self.usr_np.log(temperature)**4,
                0.0008536776054310564 + -0.0007825552849291938*self.usr_np.log(temperature) + 0.00023088784386152654*self.usr_np.log(temperature)**2 + -2.5636905937189747e-05*self.usr_np.log(temperature)**3 + 1.0133819725200464e-06*self.usr_np.log(temperature)**4,
                -0.002093331054537941 + 0.0009551433118481891*self.usr_np.log(temperature) + -0.00013894323923985315*self.usr_np.log(temperature)**2 + 8.974603118087413e-06*self.usr_np.log(temperature)**3 + -1.8416952591081097e-07*self.usr_np.log(temperature)**4,
                0.011799896789255535 + -0.007176547139722343*self.usr_np.log(temperature) + 0.0016046798538743986*self.usr_np.log(temperature)**2 + -0.00015361512650166273*self.usr_np.log(temperature)**3 + 5.428853321357422e-06*self.usr_np.log(temperature)**4,
                -0.009335286151697049 + 0.0049774080579552*self.usr_np.log(temperature) + -0.0009000585925378521*self.usr_np.log(temperature)**2 + 7.519297228955668e-05*self.usr_np.log(temperature)**3 + -2.3000420567826175e-06*self.usr_np.log(temperature)**4,
                -0.0021194343807509125 + 0.0009691103714398469*self.usr_np.log(temperature) + -0.0001415437410209868*self.usr_np.log(temperature)**2 + 9.196204490081458e-06*self.usr_np.log(temperature)**3 + -1.9113978374469097e-07*self.usr_np.log(temperature)**4,
                -0.0019075474076651014 + 0.0008156938751059555*self.usr_np.log(temperature) + -0.0001033057352502026*self.usr_np.log(temperature)**2 + 5.235151668430256e-06*self.usr_np.log(temperature)**3 + -4.243764677111869e-08*self.usr_np.log(temperature)**4,
                -0.003009672869056897 + 0.0015847679011338385*self.usr_np.log(temperature) + -0.00028341591661372364*self.usr_np.log(temperature)**2 + 2.3400497455366053e-05*self.usr_np.log(temperature)**3 + -7.068324598269804e-07*self.usr_np.log(temperature)**4,
                -0.002093331054537941 + 0.0009551433118481891*self.usr_np.log(temperature) + -0.00013894323923985315*self.usr_np.log(temperature)**2 + 8.974603118087413e-06*self.usr_np.log(temperature)**3 + -1.8416952591081097e-07*self.usr_np.log(temperature)**4,
                -0.002913105750739417 + 0.0015474779150351247*self.usr_np.log(temperature) + -0.000278885486786419*self.usr_np.log(temperature)**2 + 2.3216200335791097e-05*self.usr_np.log(temperature)**3 + -7.074354736489869e-07*self.usr_np.log(temperature)**4,
                0.0034801095778208327 + -0.00261654567161524*self.usr_np.log(temperature) + 0.0006901368404514886*self.usr_np.log(temperature)**2 + -7.282721342074574e-05*self.usr_np.log(temperature)**3 + 2.779507883753783e-06*self.usr_np.log(temperature)**4,
                -0.0072859272712669865 + 0.004223298411087991*self.usr_np.log(temperature) + -0.0007987477271429515*self.usr_np.log(temperature)**2 + 7.054617033869104e-05*self.usr_np.log(temperature)**3 + -2.2704437485901175e-06*self.usr_np.log(temperature)**4,
                -0.0029348181666410227 + 0.0015599234311142202*self.usr_np.log(temperature) + -0.0002812808486235039*self.usr_np.log(temperature)**2 + 2.3428922377638197e-05*self.usr_np.log(temperature)**3 + -7.143582738165882e-07*self.usr_np.log(temperature)**4,
                0.011422240725275307 + -0.006973463254020516*self.usr_np.log(temperature) + 0.0015647848002851477*self.usr_np.log(temperature)**2 + -0.00015019828712517556*self.usr_np.log(temperature)**3 + 5.320779247518453e-06*self.usr_np.log(temperature)**4,
                0.0044168550074850646 + -0.0031775743855656643*self.usr_np.log(temperature) + 0.0008133764546254836*self.usr_np.log(temperature)**2 + -8.454231010204543e-05*self.usr_np.log(temperature)**3 + 3.1913607821347743e-06*self.usr_np.log(temperature)**4,
                0.011799896789255535 + -0.007176547139722343*self.usr_np.log(temperature) + 0.0016046798538743986*self.usr_np.log(temperature)**2 + -0.00015361512650166273*self.usr_np.log(temperature)**3 + 5.428853321357422e-06*self.usr_np.log(temperature)**4,
                0.0034801095778208327 + -0.00261654567161524*self.usr_np.log(temperature) + 0.0006901368404514886*self.usr_np.log(temperature)**2 + -7.282721342074574e-05*self.usr_np.log(temperature)**3 + 2.779507883753783e-06*self.usr_np.log(temperature)**4,
                0.008153691915940649 + -0.003950287262275618*self.usr_np.log(temperature) + 0.0006415133652182064*self.usr_np.log(temperature)**2 + -3.490182562127687e-05*self.usr_np.log(temperature)**3 + 3.2184246744576303e-07*self.usr_np.log(temperature)**4,
                -0.009701326278444037 + 0.004014323899384872*self.usr_np.log(temperature) + -0.0004679109587932634*self.usr_np.log(temperature)**2 + 1.9380852658722403e-05*self.usr_np.log(temperature)**3 + 9.241023547807315e-08*self.usr_np.log(temperature)**4,
                0.003278860518160964 + -0.002505478451489052*self.usr_np.log(temperature) + 0.0006678165711086637*self.usr_np.log(temperature)**2 + -7.083588486066636e-05*self.usr_np.log(temperature)**3 + 2.713492698249412e-06*self.usr_np.log(temperature)**4,
                -0.00905166214758596 + 0.004761002619202597*self.usr_np.log(temperature) + -0.0008507988153643708*self.usr_np.log(temperature)**2 + 7.018609346634951e-05*self.usr_np.log(temperature)**3 + -2.118223235603562e-06*self.usr_np.log(temperature)**4,
                -0.008078545001672823 + 0.004640788076930059*self.usr_np.log(temperature) + -0.0008759648030254944*self.usr_np.log(temperature)**2 + 7.705769878335738e-05*self.usr_np.log(temperature)**3 + -2.473269089069558e-06*self.usr_np.log(temperature)**4,
                -0.009335286151697049 + 0.0049774080579552*self.usr_np.log(temperature) + -0.0009000585925378521*self.usr_np.log(temperature)**2 + 7.519297228955668e-05*self.usr_np.log(temperature)**3 + -2.3000420567826175e-06*self.usr_np.log(temperature)**4,
                -0.0072859272712669865 + 0.004223298411087991*self.usr_np.log(temperature) + -0.0007987477271429515*self.usr_np.log(temperature)**2 + 7.054617033869104e-05*self.usr_np.log(temperature)**3 + -2.2704437485901175e-06*self.usr_np.log(temperature)**4,
                -0.009701326278444037 + 0.004014323899384872*self.usr_np.log(temperature) + -0.0004679109587932634*self.usr_np.log(temperature)**2 + 1.9380852658722403e-05*self.usr_np.log(temperature)**3 + 9.241023547807315e-08*self.usr_np.log(temperature)**4,
                -0.006865233575642219 + 0.0045279883627338076*self.usr_np.log(temperature) + -0.0008654211810261634*self.usr_np.log(temperature)**2 + 7.992970847127505e-05*self.usr_np.log(temperature)**3 + -2.6377961782764532e-06*self.usr_np.log(temperature)**4,
                -0.007323280548496199 + 0.004247486239692243*self.usr_np.log(temperature) + -0.0008033594542376295*self.usr_np.log(temperature)**2 + 7.096837773031612e-05*self.usr_np.log(temperature)**3 + -2.2842690699174716e-06*self.usr_np.log(temperature)**4,
                -0.0019364269556070043 + 0.0008309848230287911*self.usr_np.log(temperature) + -0.00010613902008003644*self.usr_np.log(temperature)**2 + 5.474741525862431e-06*self.usr_np.log(temperature)**3 + -4.991875443999132e-08*self.usr_np.log(temperature)**4,
                -0.0030300534762843713 + 0.0015963119455998603*self.usr_np.log(temperature) + -0.0002855827123818112*self.usr_np.log(temperature)**2 + 2.3589003022680055e-05*self.usr_np.log(temperature)**3 + -7.128139183189626e-07*self.usr_np.log(temperature)**4,
                -0.0021194343807509125 + 0.0009691103714398469*self.usr_np.log(temperature) + -0.0001415437410209868*self.usr_np.log(temperature)**2 + 9.196204490081458e-06*self.usr_np.log(temperature)**3 + -1.9113978374469097e-07*self.usr_np.log(temperature)**4,
                -0.0029348181666410227 + 0.0015599234311142202*self.usr_np.log(temperature) + -0.0002812808486235039*self.usr_np.log(temperature)**2 + 2.3428922377638197e-05*self.usr_np.log(temperature)**3 + -7.143582738165882e-07*self.usr_np.log(temperature)**4,
                0.003278860518160964 + -0.002505478451489052*self.usr_np.log(temperature) + 0.0006678165711086637*self.usr_np.log(temperature)**2 + -7.083588486066636e-05*self.usr_np.log(temperature)**3 + 2.713492698249412e-06*self.usr_np.log(temperature)**4,
                -0.007323280548496199 + 0.004247486239692243*self.usr_np.log(temperature) + -0.0008033594542376295*self.usr_np.log(temperature)**2 + 7.096837773031612e-05*self.usr_np.log(temperature)**3 + -2.2842690699174716e-06*self.usr_np.log(temperature)**4,
                -0.0029567385229400263 + 0.0015724917419100686*self.usr_np.log(temperature) + -0.000283699905787975*self.usr_np.log(temperature)**2 + 2.364377574619869e-05*self.usr_np.log(temperature)**3 + -7.213510385737903e-07*self.usr_np.log(temperature)**4,
                ]).reshape((self.num_species, self.num_species))

    def get_species_mass_diffusivities_mixavg(self, pressure, temperature,
                                              mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fracs = self.get_mole_fractions(mmw, mass_fractions)
        bdiff_ij = self.get_species_binary_mass_diffusivities(temperature)
        temp_pres = temperature**(3/2)/pressure
        zeros = self._pyro_zeros_like(temperature)

        x_sum = self._pyro_make_array([
            mole_fracs[0] / bdiff_ij[0, 0] + mole_fracs[1] / bdiff_ij[1, 0] + mole_fracs[2] / bdiff_ij[2, 0] + mole_fracs[3] / bdiff_ij[3, 0] + mole_fracs[4] / bdiff_ij[4, 0] + mole_fracs[5] / bdiff_ij[5, 0] + mole_fracs[6] / bdiff_ij[6, 0],
            mole_fracs[0] / bdiff_ij[0, 1] + mole_fracs[1] / bdiff_ij[1, 1] + mole_fracs[2] / bdiff_ij[2, 1] + mole_fracs[3] / bdiff_ij[3, 1] + mole_fracs[4] / bdiff_ij[4, 1] + mole_fracs[5] / bdiff_ij[5, 1] + mole_fracs[6] / bdiff_ij[6, 1],
            mole_fracs[0] / bdiff_ij[0, 2] + mole_fracs[1] / bdiff_ij[1, 2] + mole_fracs[2] / bdiff_ij[2, 2] + mole_fracs[3] / bdiff_ij[3, 2] + mole_fracs[4] / bdiff_ij[4, 2] + mole_fracs[5] / bdiff_ij[5, 2] + mole_fracs[6] / bdiff_ij[6, 2],
            mole_fracs[0] / bdiff_ij[0, 3] + mole_fracs[1] / bdiff_ij[1, 3] + mole_fracs[2] / bdiff_ij[2, 3] + mole_fracs[3] / bdiff_ij[3, 3] + mole_fracs[4] / bdiff_ij[4, 3] + mole_fracs[5] / bdiff_ij[5, 3] + mole_fracs[6] / bdiff_ij[6, 3],
            mole_fracs[0] / bdiff_ij[0, 4] + mole_fracs[1] / bdiff_ij[1, 4] + mole_fracs[2] / bdiff_ij[2, 4] + mole_fracs[3] / bdiff_ij[3, 4] + mole_fracs[4] / bdiff_ij[4, 4] + mole_fracs[5] / bdiff_ij[5, 4] + mole_fracs[6] / bdiff_ij[6, 4],
            mole_fracs[0] / bdiff_ij[0, 5] + mole_fracs[1] / bdiff_ij[1, 5] + mole_fracs[2] / bdiff_ij[2, 5] + mole_fracs[3] / bdiff_ij[3, 5] + mole_fracs[4] / bdiff_ij[4, 5] + mole_fracs[5] / bdiff_ij[5, 5] + mole_fracs[6] / bdiff_ij[6, 5],
            mole_fracs[0] / bdiff_ij[0, 6] + mole_fracs[1] / bdiff_ij[1, 6] + mole_fracs[2] / bdiff_ij[2, 6] + mole_fracs[3] / bdiff_ij[3, 6] + mole_fracs[4] / bdiff_ij[4, 6] + mole_fracs[5] / bdiff_ij[5, 6] + mole_fracs[6] / bdiff_ij[6, 6],
            ])
        denom = self._pyro_make_array([
            x_sum[0] - mole_fracs[0]/bdiff_ij[0, 0],
            x_sum[1] - mole_fracs[1]/bdiff_ij[1, 1],
            x_sum[2] - mole_fracs[2]/bdiff_ij[2, 2],
            x_sum[3] - mole_fracs[3]/bdiff_ij[3, 3],
            x_sum[4] - mole_fracs[4]/bdiff_ij[4, 4],
            x_sum[5] - mole_fracs[5]/bdiff_ij[5, 5],
            x_sum[6] - mole_fracs[6]/bdiff_ij[6, 6],
            ])
        return self._pyro_make_array([
              temp_pres*self.usr_np.where(self.usr_np.greater(denom[0], zeros),
                  (mmw - mole_fracs[0] * self.wts[0])/(mmw * denom[0]),
                  bdiff_ij[0, 0]
              ),
              temp_pres*self.usr_np.where(self.usr_np.greater(denom[1], zeros),
                  (mmw - mole_fracs[1] * self.wts[1])/(mmw * denom[1]),
                  bdiff_ij[1, 1]
              ),
              temp_pres*self.usr_np.where(self.usr_np.greater(denom[2], zeros),
                  (mmw - mole_fracs[2] * self.wts[2])/(mmw * denom[2]),
                  bdiff_ij[2, 2]
              ),
              temp_pres*self.usr_np.where(self.usr_np.greater(denom[3], zeros),
                  (mmw - mole_fracs[3] * self.wts[3])/(mmw * denom[3]),
                  bdiff_ij[3, 3]
              ),
              temp_pres*self.usr_np.where(self.usr_np.greater(denom[4], zeros),
                  (mmw - mole_fracs[4] * self.wts[4])/(mmw * denom[4]),
                  bdiff_ij[4, 4]
              ),
              temp_pres*self.usr_np.where(self.usr_np.greater(denom[5], zeros),
                  (mmw - mole_fracs[5] * self.wts[5])/(mmw * denom[5]),
                  bdiff_ij[5, 5]
              ),
              temp_pres*self.usr_np.where(self.usr_np.greater(denom[6], zeros),
                  (mmw - mole_fracs[6] * self.wts[6])/(mmw * denom[6]),
                  bdiff_ij[6, 6]
              ),
              ])

    def get_species_specific_heats_r(self, temperature):
        """ Get individual species Cp/R."""
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.51, 7.51),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.74, 3.74),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.37, 5.37),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.59, 4.59),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.24, 4.24),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.52, 3.52),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.56, 3.56),
                ])

    def get_species_enthalpies_rt(self, temperature):
        """ Get individual species h/RT."""
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.51 + 5089.77593 / temperature, 7.51 + 5089.77593 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.74 + -1063.94356 / temperature, 3.74 + -1063.94356 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.37 + -48371.9697 / temperature, 5.37 + -48371.9697 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.59 + -14344.086 / temperature, 4.59 + -14344.086 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.24 + -30293.7267 / temperature, 4.24 + -30293.7267 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.52 + -917.935173 / temperature, 3.52 + -917.935173 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.56 + -1020.8999 / temperature, 3.56 + -1020.8999 / temperature),
                ])

    def get_species_entropies_r(self, temperature):
        """ Get individual species s/R."""
        return self._pyro_make_array([
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.51*self.usr_np.log(temperature) + 4.09733096, 7.51*self.usr_np.log(temperature) + 4.09733096),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.74*self.usr_np.log(temperature) + 3.65767573, 3.74*self.usr_np.log(temperature) + 3.65767573),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.37*self.usr_np.log(temperature) + 9.90105222, 5.37*self.usr_np.log(temperature) + 9.90105222),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.59*self.usr_np.log(temperature) + 3.50840928, 4.59*self.usr_np.log(temperature) + 3.50840928),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.24*self.usr_np.log(temperature) + -0.849032208, 4.24*self.usr_np.log(temperature) + -0.849032208),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.52*self.usr_np.log(temperature) + 0.683010238, 3.52*self.usr_np.log(temperature) + 0.683010238),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.56*self.usr_np.log(temperature) + 3.950372, 3.56*self.usr_np.log(temperature) + 3.950372),
                ])

    def get_species_gibbs_rt(self, temperature):
        """ Get individual species G/RT."""
        h0_rt = self.get_species_enthalpies_rt(temperature)
        s0_r = self.get_species_entropies_r(temperature)
        return h0_rt - s0_r

    def get_equilibrium_constants(self, temperature):
        rt = self.gas_constant * temperature
        c0 = self.usr_np.log(self.one_atm / rt)

        g0_rt = self.get_species_gibbs_rt(temperature)
        return self._pyro_make_array([
                    -0.17364695002734*temperature,
                    g0_rt[2] + -1*(g0_rt[3] + 0.5*g0_rt[1]) + -1*-0.5*c0,
                    g0_rt[4] + -1*(g0_rt[5] + 0.5*g0_rt[1]) + -1*-0.5*c0,
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

    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_fwd = [
            self.usr_np.exp(21.989687638093137 + 0.75*self.usr_np.log(temperature) + -1*(20128.78133995063 / temperature)) * ones,
            self.usr_np.exp(12.693776813708796 + 0.7*self.usr_np.log(temperature) + -1*(5535.414868486423 / temperature)) * ones,
            self.usr_np.exp(16.91271325351661 + -1*(4528.975801488891 / temperature)) * ones,
                ]

        return self._pyro_make_array(k_fwd)

    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(temperature)
        return self._pyro_make_array([
                    k_fwd[0]*concentrations[0]**0.5*concentrations[1]**0.65,
                    k_fwd[1]*(concentrations[3]*concentrations[1]**0.5 + -1*self.usr_np.exp(log_k_eq[1])*concentrations[2]),
                    k_fwd[2]*(concentrations[5]*concentrations[1]**0.5 + -1*self.usr_np.exp(log_k_eq[2])*concentrations[4]),
               ])

    def get_net_production_rates(self, rho, temperature, mass_fractions):
        c = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(temperature, c)
        ones = self._pyro_zeros_like(r_net[0]) + 1.0
        return self._pyro_make_array([
                -1*r_net[0] * ones,
                -1*(r_net[0] + 0.5*r_net[1] + 0.5*r_net[2]) * ones,
                r_net[1] * ones,
                2.0*r_net[0] + -1*r_net[1] * ones,
                r_net[2] * ones,
                2.0*r_net[0] + -1*r_net[2] * ones,
                0.0 * ones,
               ])

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
        self.model_name = 'uiuc_updated.yaml'
        self.num_elements = 4
        self.num_species = 7
        self.num_reactions = 3
        self.num_falloff = 0

        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['C2H4', 'O2', 'CO2', 'CO', 'H2O', 'H2', 'N2']
        self.species_indices = {'C2H4': 0, 'O2': 1, 'CO2': 2, 'CO': 3, 'H2O': 4, 'H2': 5, 'N2': 6}

        self.molecular_weights = np.array([28.054, 31.998, 44.009, 28.009999999999998, 18.015, 2.016, 28.014])
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
            self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 11.3144488322845 + 0.00199411450797709*temperature + -2.99633189643295e-07*temperature**2 + 1.89518716793814e-11*temperature**3 + -3.37402443785122e-16*temperature**4, 3.37402443785122 + 0.00843506109462805*temperature + 2.95775949418238e-07*temperature**2 + -1.43695501015373e-09*temperature**3 + 2.69921955028098e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.62121821265236 + 0.000698658481530188*temperature + -1.36762472758879e-07*temperature**2 + 1.26896313105827e-11*temperature**3 + -3.46362734255368e-16*temperature**4, 3.36741547192719 + 3.84847482505965e-05*temperature + 1.94533957871007e-06*temperature**2 + -1.15578230888786e-09*temperature**3),
            self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 6.94194752652209 + 0.000158791982538956*temperature, 3.44049295501071 + 0.00105861321692637*temperature + 4.37280273656205e-06*temperature**2 + -3.29694137659692e-09*temperature**3 + 6.61633260578982e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 4.1910836707949 + 8.42207168025508e-05*temperature, 3.36882867210203 + 3.36882867210203e-06*temperature + 9.67049430918262e-07*temperature**2 + -4.95283015473991e-10*temperature**3 + 6.73765734420406e-14*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 3000.0), 6.5698766493359 + 6.67523678681837e-05*temperature + 6.22387502882128e-09*temperature**2 + 1.57026148153926e-13*temperature**3, 3.90007166548563 + 1.33066761477293e-06*temperature**2 + -4.65028013451077e-10*temperature**3 + 4.3334129616507e-14*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.09157199385194 + 0.000589592539473733*temperature + -5.54854640011679e-08*temperature**2 + 2.13783758232463e-12*temperature**3 + -2.42469068295526e-17*temperature**4, 3.45518422321125 + 5.22846992388397e-08*temperature**2 + 1.00926212086268e-10*temperature**3 + 1.93975254636421e-14*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 3000.0), 4.2513066433086 + 6.7386195230465e-05*temperature, 3.36930976152325 + 7.17970438920127e-07*temperature**2 + -3.25518689494322e-10*temperature**3 + 4.21163720190407e-14*temperature**4),
            ])

    def get_species_enthalpies_rt(self, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 11.3144488322845 + 0.000997057253988545*temperature + -9.987772988109834e-08*temperature**2 + 4.73796791984535e-12*temperature**3 + -6.74804887570244e-17*temperature**4 + -415.389620912947 / temperature, 3.37402443785122 + 0.004217530547314025*temperature + 9.859198313941267e-08*temperature**2 + -3.592387525384325e-10*temperature**3 + 5.3984391005619604e-14*temperature**4 + 5089.77593 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.62121821265236 + 0.000349329240765094*temperature + -4.558749091962634e-08*temperature**2 + 3.172407827645675e-12*temperature**3 + -6.92725468510736e-17*temperature**4 + -1245.84786271141 / temperature, 3.36741547192719 + 1.924237412529825e-05*temperature + 6.484465262366899e-07*temperature**2 + -2.88945577221965e-10*temperature**3 + -1063.94356 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 6.94194752652209 + 7.9395991269478e-05*temperature + -50867.7417154313 / temperature, 3.44049295501071 + 0.000529306608463185*temperature + 1.45760091218735e-06*temperature**2 + -8.2423534414923e-10*temperature**3 + 1.323266521157964e-13*temperature**4 + -48371.9697 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 4.1910836707949 + 4.21103584012754e-05*temperature + -15121.4232830648 / temperature, 3.36882867210203 + 1.684414336051015e-06*temperature + 3.2234981030608734e-07*temperature**2 + -1.2382075386849775e-10*temperature**3 + 1.3475314688408119e-14*temperature**4 + -14344.086 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 3000.0), 6.5698766493359 + 3.337618393409185e-05*temperature + 2.0746250096070936e-09*temperature**2 + 3.92565370384815e-14*temperature**3 + -33997.4920017828 / temperature, 3.90007166548563 + 4.435558715909766e-07*temperature**2 + -1.1625700336276925e-10*temperature**3 + 8.6668259233014e-15*temperature**4 + -30293.7267 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.09157199385194 + 0.0002947962697368665*temperature + -1.8495154667055966e-08*temperature**2 + 5.344593955811575e-13*temperature**3 + -4.84938136591052e-18*temperature**4 + -784.614377530807 / temperature, 3.45518422321125 + 1.7428233079613233e-08*temperature**2 + 2.5231553021567e-11*temperature**3 + 3.87950509272842e-15*temperature**4 + -917.935173 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 3000.0), 4.2513066433086 + 3.36930976152325e-05*temperature + -2053.29225574664 / temperature, 3.36930976152325 + 2.393234796400423e-07*temperature**2 + -8.13796723735805e-11*temperature**3 + 8.42327440380814e-15*temperature**4 + -1020.8999 / temperature),
            ])

    def get_species_entropies_r(self, pressure, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 11.3144488322845*self.usr_np.log(temperature) + 0.00199411450797709*temperature + -1.498165948216475e-07*temperature**2 + 6.3172905597938e-12*temperature**3 + -8.43506109462805e-17*temperature**4 + -44.9857298196983, 3.37402443785122*self.usr_np.log(temperature) + 0.00843506109462805*temperature + 1.47887974709119e-07*temperature**2 + -4.789850033845766e-10*temperature**3 + 6.74804887570245e-14*temperature**4 + 4.09733096)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.62121821265236*self.usr_np.log(temperature) + 0.000698658481530188*temperature + -6.83812363794395e-08*temperature**2 + 4.229877103527567e-12*temperature**3 + -8.6590683563842e-17*temperature**4 + 1.89594174434121, 3.36741547192719*self.usr_np.log(temperature) + 3.84847482505965e-05*temperature + 9.72669789355035e-07*temperature**2 + -3.852607696292867e-10*temperature**3 + 3.65767573)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 6.94194752652209*self.usr_np.log(temperature) + 0.000158791982538956*temperature + -12.313225131286, 3.44049295501071*self.usr_np.log(temperature) + 0.00105861321692637*temperature + 2.186401368281025e-06*temperature**2 + -1.0989804588656401e-09*temperature**3 + 1.654083151447455e-13*temperature**4 + 9.90105222)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 4.1910836707949*self.usr_np.log(temperature) + 8.42207168025508e-05*temperature + -2.02032409052224, 3.36882867210203*self.usr_np.log(temperature) + 3.36882867210203e-06*temperature + 4.83524715459131e-07*temperature**2 + -1.6509433849133032e-10*temperature**3 + 1.684414336051015e-14*temperature**4 + 3.50840928)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 3000.0), 6.5698766493359*self.usr_np.log(temperature) + 6.67523678681837e-05*temperature + 3.11193751441064e-09*temperature**2 + 5.2342049384642e-14*temperature**3 + -19.7738817490647, 3.90007166548563*self.usr_np.log(temperature) + 6.65333807386465e-07*temperature**2 + -1.5500933781702567e-10*temperature**3 + 1.083353240412675e-14*temperature**4 + -0.849032208)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.09157199385194*self.usr_np.log(temperature) + 0.000589592539473733*temperature + -2.774273200058395e-08*temperature**2 + 7.126125274415434e-13*temperature**3 + -6.06172670738815e-18*temperature**4 + 2.69683197826611, 3.45518422321125*self.usr_np.log(temperature) + 2.614234961941985e-08*temperature**2 + 3.364207069542267e-11*temperature**3 + 4.849381365910525e-15*temperature**4 + 0.683010238)
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 3000.0), 4.2513066433086*self.usr_np.log(temperature) + 6.7386195230465e-05*temperature + -2.15932251170904, 3.36930976152325*self.usr_np.log(temperature) + 3.589852194600635e-07*temperature**2 + -1.0850622983144067e-10*temperature**3 + 1.0529093004760176e-14*temperature**4 + 3.950372)
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
            -0.17364695002734*temperature,
            g0_rt[2] + -1*(g0_rt[3] + 0.5*g0_rt[1]) + -1*-0.5*c0,
            -0.17364695002734*temperature,
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
            self.usr_np.exp(21.989687638093137 + -1*(18115.903205955565 / temperature)) * ones,
            self.usr_np.exp(12.759528191271576 + 0.7*self.usr_np.log(temperature) + -1*(5535.414868486423 / temperature)) * ones,
            self.usr_np.exp(18.639652073262145 + -1*(6038.634401985189 / temperature)) * ones,
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
            k_fwd[0]*concentrations[0]**0.5*concentrations[1]**0.65,
            k_fwd[1]*(concentrations[3]*concentrations[1]**0.5 + -1*self.usr_np.exp(log_k_eq[1])*concentrations[2]),
            k_fwd[2]*concentrations[5]**0.75*concentrations[1]**0.5,
            ])

    def get_net_production_rates(self, rho, temperature, mass_fractions):
        pressure = self.get_pressure(rho, temperature, mass_fractions)
        c = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(pressure, temperature, c)
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

    def get_species_thermal_conductivities(self, temperature):
        return self._pyro_make_array([
            self.usr_np.sqrt(temperature)*(-0.03384964882584151 + 0.03628284134171194*self.usr_np.log(temperature) + -0.012028594568370318*self.usr_np.log(temperature)**2 + 0.001599965478175202*self.usr_np.log(temperature)**3 + -7.30235910593391e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(0.06206613954447124 + -0.03670910998089459*self.usr_np.log(temperature) + 0.00812289006659686*self.usr_np.log(temperature)**2 + -0.0007837740699417001*self.usr_np.log(temperature)**3 + 2.8362480404370246e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(0.16823521613660328 + -0.09283372873292427*self.usr_np.log(temperature) + 0.018783485464601103*self.usr_np.log(temperature)**2 + -0.0016422572395808478*self.usr_np.log(temperature)**3 + 5.2773949244949305e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.02648324564420295 + 0.01891809648149079*self.usr_np.log(temperature) + -0.0048526870156499965*self.usr_np.log(temperature)**2 + 0.0005480021813354761*self.usr_np.log(temperature)**3 + -2.247169798394823e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.688020040129694 + 0.4192507580367533*self.usr_np.log(temperature) + -0.09513300073513713*self.usr_np.log(temperature)**2 + 0.00951630373084075*self.usr_np.log(temperature)**3 + -0.00035212520788100143*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.3485012486745646 + 0.20936008520134688*self.usr_np.log(temperature) + -0.0454714339304075*self.usr_np.log(temperature)**2 + 0.00432347479083708*self.usr_np.log(temperature)**3 + -0.0001491060879673804*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.13628873302333658 + 0.08308994511418351*self.usr_np.log(temperature) + -0.018799674928916505*self.usr_np.log(temperature)**2 + 0.0018842880075559487*self.usr_np.log(temperature)**3 + -7.009402853518108e-05*self.usr_np.log(temperature)**4),
            ])

    def get_species_binary_mass_diffusivities(self, temperature):
        return self._pyro_make_array([
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.002514025167404718 + -0.0017759952625726505*self.usr_np.log(temperature) + 0.00044874581158256305*self.usr_np.log(temperature)**2 + -4.63252893024973e-05*self.usr_np.log(temperature)**3 + 1.7399023936012455e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.001732870278755321 + 0.0006970694562179823*self.usr_np.log(temperature) + -7.498493379425965e-05*self.usr_np.log(temperature)**2 + 2.3790313073441086e-06*self.usr_np.log(temperature)**3 + 6.258149853836124e-08*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0016279660308488376 + -0.0012518611559844848*self.usr_np.log(temperature) + 0.0003350193750336406*self.usr_np.log(temperature)**2 + -3.560522825082911e-05*self.usr_np.log(temperature)**3 + 1.3658158187700505e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0019075474076651014 + 0.0008156938751059555*self.usr_np.log(temperature) + -0.0001033057352502026*self.usr_np.log(temperature)**2 + 5.235151668430256e-06*self.usr_np.log(temperature)**3 + -4.243764677111869e-08*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.011422240725275307 + -0.006973463254020516*self.usr_np.log(temperature) + 0.0015647848002851477*self.usr_np.log(temperature)**2 + -0.00015019828712517556*self.usr_np.log(temperature)**3 + 5.320779247518453e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.00905166214758596 + 0.004761002619202597*self.usr_np.log(temperature) + -0.0008507988153643708*self.usr_np.log(temperature)**2 + 7.018609346634951e-05*self.usr_np.log(temperature)**3 + -2.118223235603562e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0019364269556070043 + 0.0008309848230287911*self.usr_np.log(temperature) + -0.00010613902008003644*self.usr_np.log(temperature)**2 + 5.474741525862431e-06*self.usr_np.log(temperature)**3 + -4.991875443999132e-08*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.001732870278755321 + 0.0006970694562179823*self.usr_np.log(temperature) + -7.498493379425965e-05*self.usr_np.log(temperature)**2 + 2.3790313073441086e-06*self.usr_np.log(temperature)**3 + 6.258149853836124e-08*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003127289937114834 + 0.001633232188528128*self.usr_np.log(temperature) + -0.00029023244731756897*self.usr_np.log(temperature)**2 + 2.3795154190948053e-05*self.usr_np.log(temperature)**3 + -7.135754459081666e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0019538762606351678 + 0.0008581597928151936*self.usr_np.log(temperature) + -0.00011552373820595307*self.usr_np.log(temperature)**2 + 6.588031407432268e-06*self.usr_np.log(temperature)**3 + -9.570226239447783e-08*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003009672869056897 + 0.0015847679011338385*self.usr_np.log(temperature) + -0.00028341591661372364*self.usr_np.log(temperature)**2 + 2.3400497455366053e-05*self.usr_np.log(temperature)**3 + -7.068324598269804e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0044168550074850646 + -0.0031775743855656643*self.usr_np.log(temperature) + 0.0008133764546254836*self.usr_np.log(temperature)**2 + -8.454231010204543e-05*self.usr_np.log(temperature)**3 + 3.1913607821347743e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.008078545001672823 + 0.004640788076930059*self.usr_np.log(temperature) + -0.0008759648030254944*self.usr_np.log(temperature)**2 + 7.705769878335738e-05*self.usr_np.log(temperature)**3 + -2.473269089069558e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0030300534762843713 + 0.0015963119455998603*self.usr_np.log(temperature) + -0.0002855827123818112*self.usr_np.log(temperature)**2 + 2.3589003022680055e-05*self.usr_np.log(temperature)**3 + -7.128139183189626e-07*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.0016279660308488376 + -0.0012518611559844848*self.usr_np.log(temperature) + 0.0003350193750336406*self.usr_np.log(temperature)**2 + -3.560522825082911e-05*self.usr_np.log(temperature)**3 + 1.3658158187700505e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0019538762606351678 + 0.0008581597928151936*self.usr_np.log(temperature) + -0.00011552373820595307*self.usr_np.log(temperature)**2 + 6.588031407432268e-06*self.usr_np.log(temperature)**3 + -9.570226239447783e-08*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0008536776054310564 + -0.0007825552849291938*self.usr_np.log(temperature) + 0.00023088784386152654*self.usr_np.log(temperature)**2 + -2.5636905937189747e-05*self.usr_np.log(temperature)**3 + 1.0133819725200464e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.002093331054537941 + 0.0009551433118481891*self.usr_np.log(temperature) + -0.00013894323923985315*self.usr_np.log(temperature)**2 + 8.974603118087413e-06*self.usr_np.log(temperature)**3 + -1.8416952591081097e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.011799896789255535 + -0.007176547139722343*self.usr_np.log(temperature) + 0.0016046798538743986*self.usr_np.log(temperature)**2 + -0.00015361512650166273*self.usr_np.log(temperature)**3 + 5.428853321357422e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.009335286151697049 + 0.0049774080579552*self.usr_np.log(temperature) + -0.0009000585925378521*self.usr_np.log(temperature)**2 + 7.519297228955668e-05*self.usr_np.log(temperature)**3 + -2.3000420567826175e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0021194343807509125 + 0.0009691103714398469*self.usr_np.log(temperature) + -0.0001415437410209868*self.usr_np.log(temperature)**2 + 9.196204490081458e-06*self.usr_np.log(temperature)**3 + -1.9113978374469097e-07*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.0019075474076651014 + 0.0008156938751059555*self.usr_np.log(temperature) + -0.0001033057352502026*self.usr_np.log(temperature)**2 + 5.235151668430256e-06*self.usr_np.log(temperature)**3 + -4.243764677111869e-08*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.003009672869056897 + 0.0015847679011338385*self.usr_np.log(temperature) + -0.00028341591661372364*self.usr_np.log(temperature)**2 + 2.3400497455366053e-05*self.usr_np.log(temperature)**3 + -7.068324598269804e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.002093331054537941 + 0.0009551433118481891*self.usr_np.log(temperature) + -0.00013894323923985315*self.usr_np.log(temperature)**2 + 8.974603118087413e-06*self.usr_np.log(temperature)**3 + -1.8416952591081097e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.002913105750739417 + 0.0015474779150351247*self.usr_np.log(temperature) + -0.000278885486786419*self.usr_np.log(temperature)**2 + 2.3216200335791097e-05*self.usr_np.log(temperature)**3 + -7.074354736489869e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0034801095778208327 + -0.00261654567161524*self.usr_np.log(temperature) + 0.0006901368404514886*self.usr_np.log(temperature)**2 + -7.282721342074574e-05*self.usr_np.log(temperature)**3 + 2.779507883753783e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0072859272712669865 + 0.004223298411087991*self.usr_np.log(temperature) + -0.0007987477271429515*self.usr_np.log(temperature)**2 + 7.054617033869104e-05*self.usr_np.log(temperature)**3 + -2.2704437485901175e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0029348181666410227 + 0.0015599234311142202*self.usr_np.log(temperature) + -0.0002812808486235039*self.usr_np.log(temperature)**2 + 2.3428922377638197e-05*self.usr_np.log(temperature)**3 + -7.143582738165882e-07*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.011422240725275307 + -0.006973463254020516*self.usr_np.log(temperature) + 0.0015647848002851477*self.usr_np.log(temperature)**2 + -0.00015019828712517556*self.usr_np.log(temperature)**3 + 5.320779247518453e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0044168550074850646 + -0.0031775743855656643*self.usr_np.log(temperature) + 0.0008133764546254836*self.usr_np.log(temperature)**2 + -8.454231010204543e-05*self.usr_np.log(temperature)**3 + 3.1913607821347743e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.011799896789255535 + -0.007176547139722343*self.usr_np.log(temperature) + 0.0016046798538743986*self.usr_np.log(temperature)**2 + -0.00015361512650166273*self.usr_np.log(temperature)**3 + 5.428853321357422e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0034801095778208327 + -0.00261654567161524*self.usr_np.log(temperature) + 0.0006901368404514886*self.usr_np.log(temperature)**2 + -7.282721342074574e-05*self.usr_np.log(temperature)**3 + 2.779507883753783e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.008153691915940649 + -0.003950287262275618*self.usr_np.log(temperature) + 0.0006415133652182064*self.usr_np.log(temperature)**2 + -3.490182562127687e-05*self.usr_np.log(temperature)**3 + 3.2184246744576303e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.009701326278444037 + 0.004014323899384872*self.usr_np.log(temperature) + -0.0004679109587932634*self.usr_np.log(temperature)**2 + 1.9380852658722403e-05*self.usr_np.log(temperature)**3 + 9.241023547807315e-08*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.003278860518160964 + -0.002505478451489052*self.usr_np.log(temperature) + 0.0006678165711086637*self.usr_np.log(temperature)**2 + -7.083588486066636e-05*self.usr_np.log(temperature)**3 + 2.713492698249412e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.00905166214758596 + 0.004761002619202597*self.usr_np.log(temperature) + -0.0008507988153643708*self.usr_np.log(temperature)**2 + 7.018609346634951e-05*self.usr_np.log(temperature)**3 + -2.118223235603562e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.008078545001672823 + 0.004640788076930059*self.usr_np.log(temperature) + -0.0008759648030254944*self.usr_np.log(temperature)**2 + 7.705769878335738e-05*self.usr_np.log(temperature)**3 + -2.473269089069558e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.009335286151697049 + 0.0049774080579552*self.usr_np.log(temperature) + -0.0009000585925378521*self.usr_np.log(temperature)**2 + 7.519297228955668e-05*self.usr_np.log(temperature)**3 + -2.3000420567826175e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0072859272712669865 + 0.004223298411087991*self.usr_np.log(temperature) + -0.0007987477271429515*self.usr_np.log(temperature)**2 + 7.054617033869104e-05*self.usr_np.log(temperature)**3 + -2.2704437485901175e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.009701326278444037 + 0.004014323899384872*self.usr_np.log(temperature) + -0.0004679109587932634*self.usr_np.log(temperature)**2 + 1.9380852658722403e-05*self.usr_np.log(temperature)**3 + 9.241023547807315e-08*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.006865233575642219 + 0.0045279883627338076*self.usr_np.log(temperature) + -0.0008654211810261634*self.usr_np.log(temperature)**2 + 7.992970847127505e-05*self.usr_np.log(temperature)**3 + -2.6377961782764532e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.007323280548496199 + 0.004247486239692243*self.usr_np.log(temperature) + -0.0008033594542376295*self.usr_np.log(temperature)**2 + 7.096837773031612e-05*self.usr_np.log(temperature)**3 + -2.2842690699174716e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(-0.0019364269556070043 + 0.0008309848230287911*self.usr_np.log(temperature) + -0.00010613902008003644*self.usr_np.log(temperature)**2 + 5.474741525862431e-06*self.usr_np.log(temperature)**3 + -4.991875443999132e-08*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0030300534762843713 + 0.0015963119455998603*self.usr_np.log(temperature) + -0.0002855827123818112*self.usr_np.log(temperature)**2 + 2.3589003022680055e-05*self.usr_np.log(temperature)**3 + -7.128139183189626e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0021194343807509125 + 0.0009691103714398469*self.usr_np.log(temperature) + -0.0001415437410209868*self.usr_np.log(temperature)**2 + 9.196204490081458e-06*self.usr_np.log(temperature)**3 + -1.9113978374469097e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0029348181666410227 + 0.0015599234311142202*self.usr_np.log(temperature) + -0.0002812808486235039*self.usr_np.log(temperature)**2 + 2.3428922377638197e-05*self.usr_np.log(temperature)**3 + -7.143582738165882e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.003278860518160964 + -0.002505478451489052*self.usr_np.log(temperature) + 0.0006678165711086637*self.usr_np.log(temperature)**2 + -7.083588486066636e-05*self.usr_np.log(temperature)**3 + 2.713492698249412e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.007323280548496199 + 0.004247486239692243*self.usr_np.log(temperature) + -0.0008033594542376295*self.usr_np.log(temperature)**2 + 7.096837773031612e-05*self.usr_np.log(temperature)**3 + -2.2842690699174716e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0029567385229400263 + 0.0015724917419100686*self.usr_np.log(temperature) + -0.000283699905787975*self.usr_np.log(temperature)**2 + 2.364377574619869e-05*self.usr_np.log(temperature)**3 + -7.213510385737903e-07*self.usr_np.log(temperature)**4),
            ]),
        ])

    def get_mixture_viscosity_mixavg(self, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fractions = self.get_mole_fractions(mmw, mass_fractions)
        viscosities = self.get_species_viscosities(temperature)
        mix_rule_f = self._pyro_make_array([
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[0])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[1])*self.usr_np.sqrt(1.1405860126898126)))**2) / self.usr_np.sqrt(15.013938371148196) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[2])*self.usr_np.sqrt(1.5687246025522208)))**2) / self.usr_np.sqrt(13.099684155513645) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[3])*self.usr_np.sqrt(0.9984315962073145)))**2) / self.usr_np.sqrt(16.012566940378434) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[4])*self.usr_np.sqrt(0.642154416482498)))**2) / self.usr_np.sqrt(20.458062725506522) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[5])*self.usr_np.sqrt(0.07186141013759179)))**2) / self.usr_np.sqrt(119.32539682539682) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[6])*self.usr_np.sqrt(0.998574178370286)))**2) / self.usr_np.sqrt(16.01142285999857),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[0])*self.usr_np.sqrt(0.8767422963935245)))**2) / self.usr_np.sqrt(17.1246881015185) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[1])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[2])*self.usr_np.sqrt(1.3753672104506531)))**2) / self.usr_np.sqrt(13.816628416914721) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[3])*self.usr_np.sqrt(0.875367210450653)))**2) / self.usr_np.sqrt(17.139021777936453) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[4])*self.usr_np.sqrt(0.5630039377461091)))**2) / self.usr_np.sqrt(22.209492089925064) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[5])*self.usr_np.sqrt(0.06300393774610913)))**2) / self.usr_np.sqrt(134.97619047619048) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[6])*self.usr_np.sqrt(0.8754922182636414)))**2) / self.usr_np.sqrt(17.137716855857786),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[0])*self.usr_np.sqrt(0.6374605194392056)))**2) / self.usr_np.sqrt(20.549796820417768) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[1])*self.usr_np.sqrt(0.7270785521143402)))**2) / self.usr_np.sqrt(19.002937683605225) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[2])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[3])*self.usr_np.sqrt(0.6364607239428298)))**2) / self.usr_np.sqrt(20.569510888968225) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[4])*self.usr_np.sqrt(0.4093480878911132)))**2) / self.usr_np.sqrt(27.543269497640853) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[5])*self.usr_np.sqrt(0.04580881183394306)))**2) / self.usr_np.sqrt(182.63888888888889) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[6])*self.usr_np.sqrt(0.6365516144425004)))**2) / self.usr_np.sqrt(20.567716141929036),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[0])*self.usr_np.sqrt(1.0015708675473045)))**2) / self.usr_np.sqrt(15.987452769658516) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[1])*self.usr_np.sqrt(1.1423777222420566)))**2) / self.usr_np.sqrt(15.002937683605225) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[2])*self.usr_np.sqrt(1.5711888611210283)))**2) / self.usr_np.sqrt(13.091685791542638) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[3])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[4])*self.usr_np.sqrt(0.6431631560157087)))**2) / self.usr_np.sqrt(20.43852345267832) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[5])*self.usr_np.sqrt(0.07197429489468048)))**2) / self.usr_np.sqrt(119.15079365079364) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[6])*self.usr_np.sqrt(1.000142806140664)))**2) / self.usr_np.sqrt(15.998857714000142),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[0])*self.usr_np.sqrt(1.557257840688315)))**2) / self.usr_np.sqrt(13.137235331859983) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[1])*self.usr_np.sqrt(1.7761865112406328)))**2) / self.usr_np.sqrt(12.504031501968873) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[2])*self.usr_np.sqrt(2.4429086872051067)))**2) / self.usr_np.sqrt(11.274784703128905) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[3])*self.usr_np.sqrt(1.5548154315847902)))**2) / self.usr_np.sqrt(13.14530524812567) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[4])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[5])*self.usr_np.sqrt(0.11190674437968359)))**2) / self.usr_np.sqrt(79.48809523809524) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[6])*self.usr_np.sqrt(1.55503746877602)))**2) / self.usr_np.sqrt(13.144570571856928),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[0])*self.usr_np.sqrt(13.915674603174603)))**2) / self.usr_np.sqrt(8.574891281100735) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[1])*self.usr_np.sqrt(15.87202380952381)))**2) / self.usr_np.sqrt(8.504031501968873) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[2])*self.usr_np.sqrt(21.82986111111111)))**2) / self.usr_np.sqrt(8.366470494671544) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[3])*self.usr_np.sqrt(13.893849206349206)))**2) / self.usr_np.sqrt(8.575794359157443) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[4])*self.usr_np.sqrt(8.936011904761905)))**2) / self.usr_np.sqrt(8.895253955037468) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[5])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[6])*self.usr_np.sqrt(13.895833333333332)))**2) / self.usr_np.sqrt(8.575712143928037),
            (mole_fractions[0]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[0])*self.usr_np.sqrt(1.0014278574998214)))**2) / self.usr_np.sqrt(15.988593426962288) + (mole_fractions[1]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[1])*self.usr_np.sqrt(1.1422146069822232)))**2) / self.usr_np.sqrt(15.00393774610913) + (mole_fractions[2]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[2])*self.usr_np.sqrt(1.5709645177411296)))**2) / self.usr_np.sqrt(13.092412915540002) + (mole_fractions[3]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[3])*self.usr_np.sqrt(0.9998572142500178)))**2) / self.usr_np.sqrt(16.00114244912531) + (mole_fractions[4]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[4])*self.usr_np.sqrt(0.6430713214821161)))**2) / self.usr_np.sqrt(20.440299750208162) + (mole_fractions[5]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[5])*self.usr_np.sqrt(0.0719640179910045)))**2) / self.usr_np.sqrt(119.16666666666666) + (mole_fractions[6]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[6])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0),
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
            mole_fractions[0] / bdiff_ij[0][0] + mole_fractions[1] / bdiff_ij[1][0] + mole_fractions[2] / bdiff_ij[2][0] + mole_fractions[3] / bdiff_ij[3][0] + mole_fractions[4] / bdiff_ij[4][0] + mole_fractions[5] / bdiff_ij[5][0] + mole_fractions[6] / bdiff_ij[6][0],
            mole_fractions[0] / bdiff_ij[0][1] + mole_fractions[1] / bdiff_ij[1][1] + mole_fractions[2] / bdiff_ij[2][1] + mole_fractions[3] / bdiff_ij[3][1] + mole_fractions[4] / bdiff_ij[4][1] + mole_fractions[5] / bdiff_ij[5][1] + mole_fractions[6] / bdiff_ij[6][1],
            mole_fractions[0] / bdiff_ij[0][2] + mole_fractions[1] / bdiff_ij[1][2] + mole_fractions[2] / bdiff_ij[2][2] + mole_fractions[3] / bdiff_ij[3][2] + mole_fractions[4] / bdiff_ij[4][2] + mole_fractions[5] / bdiff_ij[5][2] + mole_fractions[6] / bdiff_ij[6][2],
            mole_fractions[0] / bdiff_ij[0][3] + mole_fractions[1] / bdiff_ij[1][3] + mole_fractions[2] / bdiff_ij[2][3] + mole_fractions[3] / bdiff_ij[3][3] + mole_fractions[4] / bdiff_ij[4][3] + mole_fractions[5] / bdiff_ij[5][3] + mole_fractions[6] / bdiff_ij[6][3],
            mole_fractions[0] / bdiff_ij[0][4] + mole_fractions[1] / bdiff_ij[1][4] + mole_fractions[2] / bdiff_ij[2][4] + mole_fractions[3] / bdiff_ij[3][4] + mole_fractions[4] / bdiff_ij[4][4] + mole_fractions[5] / bdiff_ij[5][4] + mole_fractions[6] / bdiff_ij[6][4],
            mole_fractions[0] / bdiff_ij[0][5] + mole_fractions[1] / bdiff_ij[1][5] + mole_fractions[2] / bdiff_ij[2][5] + mole_fractions[3] / bdiff_ij[3][5] + mole_fractions[4] / bdiff_ij[4][5] + mole_fractions[5] / bdiff_ij[5][5] + mole_fractions[6] / bdiff_ij[6][5],
            mole_fractions[0] / bdiff_ij[0][6] + mole_fractions[1] / bdiff_ij[1][6] + mole_fractions[2] / bdiff_ij[2][6] + mole_fractions[3] / bdiff_ij[3][6] + mole_fractions[4] / bdiff_ij[4][6] + mole_fractions[5] / bdiff_ij[5][6] + mole_fractions[6] / bdiff_ij[6][6],
            ])
        denom = self._pyro_make_array([
            x_sum[0] - mole_fractions[0]/bdiff_ij[0][0],
            x_sum[1] - mole_fractions[1]/bdiff_ij[1][1],
            x_sum[2] - mole_fractions[2]/bdiff_ij[2][2],
            x_sum[3] - mole_fractions[3]/bdiff_ij[3][3],
            x_sum[4] - mole_fractions[4]/bdiff_ij[4][4],
            x_sum[5] - mole_fractions[5]/bdiff_ij[5][5],
            x_sum[6] - mole_fractions[6]/bdiff_ij[6][6],
            ])

        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(denom[0], zeros), (mmw - mole_fractions[0] * self.molecular_weights[0])/(pressure * mmw * denom[0]), bdiff_ij[0][0] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[1], zeros), (mmw - mole_fractions[1] * self.molecular_weights[1])/(pressure * mmw * denom[1]), bdiff_ij[1][1] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[2], zeros), (mmw - mole_fractions[2] * self.molecular_weights[2])/(pressure * mmw * denom[2]), bdiff_ij[2][2] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[3], zeros), (mmw - mole_fractions[3] * self.molecular_weights[3])/(pressure * mmw * denom[3]), bdiff_ij[3][3] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[4], zeros), (mmw - mole_fractions[4] * self.molecular_weights[4])/(pressure * mmw * denom[4]), bdiff_ij[4][4] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[5], zeros), (mmw - mole_fractions[5] * self.molecular_weights[5])/(pressure * mmw * denom[5]), bdiff_ij[5][5] / pressure),
            self.usr_np.where(self.usr_np.greater(denom[6], zeros), (mmw - mole_fractions[6] * self.molecular_weights[6])/(pressure * mmw * denom[6]), bdiff_ij[6][6] / pressure),
            ])

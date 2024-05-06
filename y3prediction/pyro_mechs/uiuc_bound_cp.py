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
        self.model_name = 'uiuc_bound_cp.yaml'
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
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 15.857914857900802, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.21785087461938 + 0.0066860330395617*temperature + -1.79535702793147e-06*temperature**2 + 2.18865397063934e-10*temperature**3 + -1.01220733135537e-14*temperature**4, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.73312012902641 + -0.00225161088114689*temperature + 2.35442451235782e-05*temperature**2 + -1.37084841614577e-08*temperature**3, 3.6776866372578287))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 5.099229143204035, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.6441176454853 + 0.000646414432661291*temperature + -1.00626310785635e-07*temperature**2 + 5.5517226390456e-12*temperature**3, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.3625982879046 + -2.75301645330651e-06*temperature + 2.04226666018565e-06*temperature**2 + -1.20665444163695e-09*temperature**3, 3.367415471927194))),
            self.usr_np.where(self.usr_np.greater(temperature, 3500.0), 7.595736055228211, self.usr_np.where(self.usr_np.greater(temperature, 1500.0), 5.84211519292204 + 0.00107835571675337*temperature + -2.15537796575996e-07*temperature**2 + 1.44539676204158e-11*temperature**3, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 2.38429599080778 + 0.00518822098664269*temperature + 1.29694718447492e-06*temperature**2 + -3.38385694223856e-09*temperature**3 + 1.05861321692637e-12*temperature**4, 2.646533042315928))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 15.496611891669087, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.29879520889301 + 0.00649479687716679*temperature + -1.64661794766909e-06*temperature**2 + 1.77033728953477e-10*temperature**3 + -6.73765734420406e-15*temperature**4, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.61750268526206 + -0.000207109710356553*temperature + 1.98078104556147e-05*temperature**2 + -1.19009332205202e-08*temperature**3, 3.655179109230704))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 7.583472682888712, self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 4.62591833656212 + 0.000946420098943144*temperature + -7.26216815885404e-08*temperature**2 + -4.93452361781564e-13*temperature**3, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.95858864602834 + -0.000175726158627177*temperature + 1.81002662158116e-06*temperature**2 + -7.0786721708685e-10*temperature**3 + 6.50011944247605e-14*temperature**4, 3.954239327506263))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 5.091850434206049, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.12772071195504 + 0.000511469323457161*temperature + -7.68518627733334e-09*temperature**2 + -3.687942228041e-12*temperature**3 + -2.42469068295526e-17*temperature**4, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.49143165605928 + 5.36892022839055e-06*temperature + -6.71954397620306e-08*temperature**2 + 1.78789998010718e-10*temperature**3 + 1.93975254636421e-14*temperature**4, 3.4915545834555797))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 4.632800922094482, self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 3.97578551859744 + 0.000221907249698863*temperature + -2.48333158774068e-08*temperature**2 + 1.01653368930791e-12*temperature**3, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.43881298555407 + -8.5699234117912e-05*temperature + 8.87661153413292e-07*temperature**2 + -4.12286806217124e-10*temperature**3 + 5.05396464228488e-14*temperature**4, 3.4366959567537205))),
            ])

    def get_species_enthalpies_rt(self, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 15.857914857900802 + -8484.010667579607 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.21785087461938 + 0.00334301651978085*temperature + -5.984523426438233e-07*temperature**2 + 5.47163492659835e-11*temperature**3 + -2.02441466271074e-15*temperature**4 + 3102.94429958825 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.73312012902641 + -0.001125805440573445*temperature + 7.8480817078594e-06*temperature**2 + -3.427121040364425e-09*temperature**3 + 5089.77593 / temperature, 3.6776866372578287 + 5090.692681693975 / temperature))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 5.099229143204035 + -3800.3459661406823 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.6441176454853 + 0.0003232072163306455*temperature + -3.3542103595211667e-08*temperature**2 + 1.3879306597614e-12*temperature**3 + -1258.80052621656 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.3625982879046 + -1.376508226653255e-06*temperature + 6.807555533952166e-07*temperature**2 + -3.016636104092375e-10*temperature**3 + -1063.94356 / temperature, 3.367415471927194 + -1064.1046514250868 / temperature))),
            self.usr_np.where(self.usr_np.greater(temperature, 3500.0), 7.595736055228211 + -51997.66185962649 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 1500.0), 5.84211519292204 + 0.000539177858376685*temperature + -7.184593219199866e-08*temperature**2 + 3.61349190510395e-12*temperature**3 + -49926.772892947 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 2.38429599080778 + 0.002594110493321345*temperature + 4.3231572815830666e-07*temperature**2 + -8.4596423555964e-10*temperature**3 + 2.11722643385274e-13*temperature**4 + -48371.9697 / temperature, 2.646533042315928 + -48378.54745798923 / temperature))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 15.496611891669087 + -6765.663019472601 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.29879520889301 + 0.003247398438583395*temperature + -5.4887264922303e-07*temperature**2 + 4.425843223836925e-11*temperature**3 + -1.347531468840812e-15*temperature**4 + 3190.86204113572 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.61750268526206 + -0.0001035548551782765*temperature + 6.6026034852049e-06*temperature**2 + -2.97523330513005e-09*temperature**3 + 5089.77593 / temperature, 3.655179109230704 + 5088.439951891115 / temperature))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 7.583472682888712 + -37364.17447224589 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 4.62591833656212 + 0.000473210049471572*temperature + -2.4207227196180135e-08*temperature**2 + -1.23363090445391e-13*temperature**3 + -31265.7705356708 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.95858864602834 + -8.78630793135885e-05*temperature + 6.0334220719372e-07*temperature**2 + -1.769668042717125e-10*temperature**3 + 1.30002388849521e-14*temperature**4 + -30293.7267 / temperature, 3.954239327506263 + -30293.654575976234 / temperature))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 5.091850434206049 + -5141.873336769686 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.12772071195504 + 0.0002557346617285805*temperature + -2.5617287591111132e-09*temperature**2 + -9.2198555701025e-13*temperature**3 + -4.84938136591052e-18*temperature**4 + -777.607342137931 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.49143165605928 + 2.684460114195275e-06*temperature + -2.2398479920676867e-08*temperature**2 + 4.46974995026795e-11*temperature**3 + 3.87950509272842e-15*temperature**4 + -917.935173 / temperature, 3.4915545834555797 + -917.9371274578024 / temperature))),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 4.632800922094482 + -3012.902725704837 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 3.97578551859744 + 0.0001109536248494315*temperature + -8.2777719591356e-09*temperature**2 + 2.541334223269775e-13*temperature**3 + -1606.49897146459 / temperature, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.43881298555407 + -4.2849617058956e-05*temperature + 2.95887051137764e-07*temperature**2 + -1.03071701554281e-10*temperature**3 + 1.010792928456976e-14*temperature**4 + -1020.8999 / temperature, 3.4366959567537205 + -1020.8648277606445 / temperature))),
            ])

    def get_species_entropies_r(self, pressure, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 15.857914857900802*self.usr_np.log(temperature) + -63.06819284336632, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.21785087461938*self.usr_np.log(temperature) + 0.0066860330395617*temperature + -8.97678513965735e-07*temperature**2 + 7.295513235464466e-11*temperature**3 + -2.530518328388425e-15*temperature**4 + -13.974343144185, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.73312012902641*self.usr_np.log(temperature) + -0.00225161088114689*temperature + 1.17721225617891e-05*temperature**2 + -4.5694947204858996e-09*temperature**3 + 4.09733096, 3.6776866372578287*self.usr_np.log(temperature) + 101.92865427057696)))
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 5.099229143204035*self.usr_np.log(temperature) + -10.401626990559514, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.6441176454853*self.usr_np.log(temperature) + 0.000646414432661291*temperature + -5.03131553928175e-08*temperature**2 + 1.8505742130152e-12*temperature**3 + 1.73121921648199, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.3625982879046*self.usr_np.log(temperature) + -2.75301645330651e-06*temperature + 1.021133330092825e-06*temperature**2 + -4.0221814721231667e-10*temperature**3 + 3.65767573, 3.367415471927194*self.usr_np.log(temperature) + -21.2953512294836)))
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 3500.0), 7.595736055228211*self.usr_np.log(temperature) + -25.91460000928386, self.usr_np.where(self.usr_np.greater(temperature, 1500.0), 5.84211519292204*self.usr_np.log(temperature) + 0.00107835571675337*temperature + -1.07768898287998e-07*temperature**2 + 4.817989206805267e-12*temperature**3 + -10.003690576177, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 2.38429599080778*self.usr_np.log(temperature) + 0.00518822098664269*temperature + 6.4847359223746e-07*temperature**2 + -1.12795231407952e-09*temperature**3 + 2.646533042315925e-13*temperature**4 + 9.90105222, 2.646533042315928*self.usr_np.log(temperature) + -968.2043784850189)))
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 15.496611891669087*self.usr_np.log(temperature) + -59.59164583286962, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.29879520889301*self.usr_np.log(temperature) + 0.00649479687716679*temperature + -8.23308973834545e-07*temperature**2 + 5.901124298449233e-11*temperature**3 + -1.684414336051015e-15*temperature**4 + -14.4233785793103, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.61750268526206*self.usr_np.log(temperature) + -0.000207109710356553*temperature + 9.90390522780735e-06*temperature**2 + -3.9669777401734e-09*temperature**3 + 4.09733096, 3.655179109230704*self.usr_np.log(temperature) + 101.66203596800656)))
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 7.583472682888712*self.usr_np.log(temperature) + -26.60444765542961, self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 4.62591833656212*self.usr_np.log(temperature) + 0.000946420098943144*temperature + -3.63108407942702e-08*temperature**2 + -1.6448412059385465e-13*temperature**3 + -6.02666126447272, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.95858864602834*self.usr_np.log(temperature) + -0.000175726158627177*temperature + 9.0501331079058e-07*temperature**2 + -2.3595573902895e-10*temperature**3 + 1.6250298606190125e-14*temperature**4 + -0.849032208, 3.954239327506263*self.usr_np.log(temperature) + -605.8640725334411)))
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 5.091850434206049*self.usr_np.log(temperature) + -14.559481960291059, self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.12772071195504*self.usr_np.log(temperature) + 0.000511469323457161*temperature + -3.84259313866667e-09*temperature**2 + -1.2293140760136665e-12*temperature**3 + -6.06172670738815e-18*temperature**4 + 2.7252623253607, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.49143165605928*self.usr_np.log(temperature) + 5.36892022839055e-06*temperature + -3.35977198810153e-08*temperature**2 + 5.959666600357267e-11*temperature**3 + 4.849381365910525e-15*temperature**4 + 0.683010238, 3.4915545834555797*self.usr_np.log(temperature) + -18.35899242319872)))
            - self.usr_np.log(pressure/self.one_atm),
            self.usr_np.where(self.usr_np.greater(temperature, 6000.0), 4.632800922094482*self.usr_np.log(temperature) + -5.0258307830712, self.usr_np.where(self.usr_np.greater(temperature, 2000.0), 3.97578551859744*self.usr_np.log(temperature) + 0.000221907249698863*temperature + -1.24166579387034e-08*temperature**2 + 3.388445631026367e-13*temperature**3 + 0.178688469772858, self.usr_np.where(self.usr_np.greater(temperature, 50.0), 3.43881298555407*self.usr_np.log(temperature) + -8.5699234117912e-05*temperature + 4.43830576706646e-07*temperature**2 + -1.37428935405708e-10*temperature**3 + 1.26349116057122e-14*temperature**4 + 3.950372, 3.4366959567537205*self.usr_np.log(temperature) + -20.412908619542737)))
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
            self.usr_np.sqrt(temperature)*(0.005851301893060288 + -0.003811513854751528*self.usr_np.log(temperature) + 0.0009573415643508345*self.usr_np.log(temperature)**2 + -9.945391621310037e-05*self.usr_np.log(temperature)**3 + 3.744812984437627e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(0.0012953423457640438 + -0.0009696871017770621*self.usr_np.log(temperature) + 0.00035911106366705726*self.usr_np.log(temperature)**2 + -4.572662614642755e-05*self.usr_np.log(temperature)**3 + 2.008217191504956e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(0.0071027189633221365 + -0.004709059891677268*self.usr_np.log(temperature) + 0.0012049763552506596*self.usr_np.log(temperature)**2 + -0.00012780069461178095*self.usr_np.log(temperature)**3 + 4.915878462194422e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(0.0004622215034199173 + -0.0003873269871324575*self.usr_np.log(temperature) + 0.00020625783310658478*self.usr_np.log(temperature)**2 + -2.8953927998488606e-05*self.usr_np.log(temperature)**3 + 1.340674166522076e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(-0.0022919006395473655 + 0.0019078598472048287*self.usr_np.log(temperature) + -0.0005267170363522416*self.usr_np.log(temperature)**2 + 6.79366968756776e-05*self.usr_np.log(temperature)**3 + -3.108252365682122e-06*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(-0.001385814273341086 + 0.0010978434115242876*self.usr_np.log(temperature) + -0.00022571790610812796*self.usr_np.log(temperature)**2 + 2.151909391634159e-05*self.usr_np.log(temperature)**3 + -7.605118763551978e-07*self.usr_np.log(temperature)**4)**2,
            self.usr_np.sqrt(temperature)*(0.00042156890211018873 + -0.00035949980170071845*self.usr_np.log(temperature) + 0.00020028936391357915*self.usr_np.log(temperature)**2 + -2.8378643672896005e-05*self.usr_np.log(temperature)**3 + 1.320142568892464e-06*self.usr_np.log(temperature)**4)**2,
            ])

    def get_species_thermal_conductivities(self, temperature):
        return self._pyro_make_array([
            self.usr_np.sqrt(temperature)*(-0.08284266603709366 + 0.06352524171889348*self.usr_np.log(temperature) + -0.017757661313977192*self.usr_np.log(temperature)**2 + 0.002138079495999018*self.usr_np.log(temperature)**3 + -9.201489975617881e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(0.0007155398409704719 + -0.0006023992925310735*self.usr_np.log(temperature) + 0.00020224063052055546*self.usr_np.log(temperature)**2 + -1.6239099353220886e-05*self.usr_np.log(temperature)**3 + 6.420810817980829e-07*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.013772062673381665 + 0.011798489276943283*self.usr_np.log(temperature) + -0.0036597931694925465*self.usr_np.log(temperature)**2 + 0.0004877022507334383*self.usr_np.log(temperature)**3 + -2.2712368383465553e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.09848075675466232 + 0.07514007677177321*self.usr_np.log(temperature) + -0.02092323364525931*self.usr_np.log(temperature)**2 + 0.0025188264054755646*self.usr_np.log(temperature)**3 + -0.00010836975774724577*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.036988362547714836 + 0.026707080245830128*self.usr_np.log(temperature) + -0.0069384651980212335*self.usr_np.log(temperature)**2 + 0.0007664041615176073*self.usr_np.log(temperature)**3 + -2.8677998934992115e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.028225472546148527 + 0.014308340650303324*self.usr_np.log(temperature) + -0.0011195001883220425*self.usr_np.log(temperature)**2 + -0.0001392252783182794*self.usr_np.log(temperature)**3 + 1.8573757792919486e-05*self.usr_np.log(temperature)**4),
            self.usr_np.sqrt(temperature)*(-0.010286946075589862 + 0.007013742372959807*self.usr_np.log(temperature) + -0.001706921773506421*self.usr_np.log(temperature)**2 + 0.00019084268803529054*self.usr_np.log(temperature)**3 + -7.66487246812478e-06*self.usr_np.log(temperature)**4),
            ])

    def get_species_binary_mass_diffusivities(self, temperature):
        return self._pyro_make_array([
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.002271101988894026 + -0.0015134978679144687*self.usr_np.log(temperature) + 0.0003655951461039401*self.usr_np.log(temperature)**2 + -3.5862434307317536e-05*self.usr_np.log(temperature)**3 + 1.2778594426822881e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.002775448306129899 + -0.00198544359720254*self.usr_np.log(temperature) + 0.0005197646970852268*self.usr_np.log(temperature)**2 + -5.586392207967723e-05*self.usr_np.log(temperature)**3 + 2.1885911283605558e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.002324559274686921 + -0.001569835690560499*self.usr_np.log(temperature) + 0.000385127984719893*self.usr_np.log(temperature)**2 + -3.8618356438211e-05*self.usr_np.log(temperature)**3 + 1.4104582573793424e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.002637403347649054 + -0.0018990653721549329*self.usr_np.log(temperature) + 0.0005007939496333914*self.usr_np.log(temperature)**2 + -5.412716594731041e-05*self.usr_np.log(temperature)**3 + 2.1314376675674003e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0015189786364662409 + -0.0008235609648541394*self.usr_np.log(temperature) + 0.00014629315350617254*self.usr_np.log(temperature)**2 + -6.129520464110185e-06*self.usr_np.log(temperature)**3 + -1.1768169549222425e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0036041946252137326 + -0.0029764317133841047*self.usr_np.log(temperature) + 0.0009079431693159895*self.usr_np.log(temperature)**2 + -0.00010602090367469384*self.usr_np.log(temperature)**3 + 4.449671995954592e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0026514846430530166 + -0.0019100110219294102*self.usr_np.log(temperature) + 0.0005039194873985993*self.usr_np.log(temperature)**2 + -5.448422948721137e-05*self.usr_np.log(temperature)**3 + 2.1461864939643698e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.002775448306129899 + -0.00198544359720254*self.usr_np.log(temperature) + 0.0005197646970852268*self.usr_np.log(temperature)**2 + -5.586392207967723e-05*self.usr_np.log(temperature)**3 + 2.1885911283605558e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0013996384984963312 + -0.0011293647033478542*self.usr_np.log(temperature) + 0.00033669628138626833*self.usr_np.log(temperature)**2 + -3.8925644118021084e-05*self.usr_np.log(temperature)**3 + 1.6212507954018387e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0025031120448872836 + -0.0018089969726089927*self.usr_np.log(temperature) + 0.0004790141492455792*self.usr_np.log(temperature)**2 + -5.192937287720725e-05*self.usr_np.log(temperature)**3 + 2.050464908717095e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.001174545314485485 + -0.0009741306298801209*self.usr_np.log(temperature) + 0.0002983877890533498*self.usr_np.log(temperature)**2 + -3.490406716653499e-05*self.usr_np.log(temperature)**3 + 1.4668652255513687e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0044677271325266514 + -0.0029885924646760307*self.usr_np.log(temperature) + 0.0007251139988479542*self.usr_np.log(temperature)**2 + -7.158933991218702e-05*self.usr_np.log(temperature)**3 + 2.5697535021804744e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.001979066782775548 + 0.0008466494772890718*self.usr_np.log(temperature) + -3.7669754660562133e-07*self.usr_np.log(temperature)**2 + -1.1842471465202141e-05*self.usr_np.log(temperature)**3 + 8.794944288185929e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0011713797855077278 + -0.000973473615577208*self.usr_np.log(temperature) + 0.0002987686824672578*self.usr_np.log(temperature)**2 + -3.497731180583652e-05*self.usr_np.log(temperature)**3 + 1.4708557819154825e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.002324559274686921 + -0.001569835690560499*self.usr_np.log(temperature) + 0.000385127984719893*self.usr_np.log(temperature)**2 + -3.8618356438211e-05*self.usr_np.log(temperature)**3 + 1.4104582573793424e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0025031120448872836 + -0.0018089969726089927*self.usr_np.log(temperature) + 0.0004790141492455792*self.usr_np.log(temperature)**2 + -5.192937287720725e-05*self.usr_np.log(temperature)**3 + 2.050464908717095e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0022772108632264565 + -0.0015554611453821626*self.usr_np.log(temperature) + 0.00038661378161274394*self.usr_np.log(temperature)**2 + -3.943627515353187e-05*self.usr_np.log(temperature)**3 + 1.4671001680753625e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0023660852238935764 + -0.0017220367590927322*self.usr_np.log(temperature) + 0.0004595999081938157*self.usr_np.log(temperature)**2 + -5.009823700287825e-05*self.usr_np.log(temperature)**3 + 1.987830528529638e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0013656952564267105 + -0.0007105562471462386*self.usr_np.log(temperature) + 0.00011603241012130588*self.usr_np.log(temperature)**2 + -2.665882095963678e-06*self.usr_np.log(temperature)**3 + -2.612193255958686e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.002844270749279069 + -0.002487828229731366*self.usr_np.log(temperature) + 0.0008006603086921086*self.usr_np.log(temperature)**2 + -9.554464569128792e-05*self.usr_np.log(temperature)**3 + 4.075442327628586e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.00237729552582244 + -0.0017310144290815796*self.usr_np.log(temperature) + 0.0004622408742451766*self.usr_np.log(temperature)**2 + -5.040404703461491e-05*self.usr_np.log(temperature)**3 + 2.0005929297812444e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.002637403347649054 + -0.0018990653721549329*self.usr_np.log(temperature) + 0.0005007939496333914*self.usr_np.log(temperature)**2 + -5.412716594731041e-05*self.usr_np.log(temperature)**3 + 2.1314376675674003e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.001174545314485485 + -0.0009741306298801209*self.usr_np.log(temperature) + 0.0002983877890533498*self.usr_np.log(temperature)**2 + -3.490406716653499e-05*self.usr_np.log(temperature)**3 + 1.4668652255513687e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0023660852238935764 + -0.0017220367590927322*self.usr_np.log(temperature) + 0.0004595999081938157*self.usr_np.log(temperature)**2 + -5.009823700287825e-05*self.usr_np.log(temperature)**3 + 1.987830528529638e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0009606321469760227 + -0.0008254672717657775*self.usr_np.log(temperature) + 0.0002614317603907832*self.usr_np.log(temperature)**2 + -3.100154939997733e-05*self.usr_np.log(temperature)**3 + 1.3162477546431314e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.00446841478271087 + -0.003009644503873177*self.usr_np.log(temperature) + 0.0007360917111922527*self.usr_np.log(temperature)**2 + -7.34999132088629e-05*self.usr_np.log(temperature)**3 + 2.6719789818844827e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.00225956866781686 + 0.0010867500833295913*self.usr_np.log(temperature) + -7.29762106412335e-05*self.usr_np.log(temperature)**2 + -3.310215675439311e-06*self.usr_np.log(temperature)**3 + 5.203663600652946e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0009562551925276512 + -0.0008238588219283307*self.usr_np.log(temperature) + 0.00026154843260841*self.usr_np.log(temperature)**2 + -3.1044891421602505e-05*self.usr_np.log(temperature)**3 + 1.3190163269443288e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.0015189786364662409 + -0.0008235609648541394*self.usr_np.log(temperature) + 0.00014629315350617254*self.usr_np.log(temperature)**2 + -6.129520464110185e-06*self.usr_np.log(temperature)**3 + -1.1768169549222425e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0044677271325266514 + -0.0029885924646760307*self.usr_np.log(temperature) + 0.0007251139988479542*self.usr_np.log(temperature)**2 + -7.158933991218702e-05*self.usr_np.log(temperature)**3 + 2.5697535021804744e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0013656952564267105 + -0.0007105562471462386*self.usr_np.log(temperature) + 0.00011603241012130588*self.usr_np.log(temperature)**2 + -2.665882095963678e-06*self.usr_np.log(temperature)**3 + -2.612193255958686e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.00446841478271087 + -0.003009644503873177*self.usr_np.log(temperature) + 0.0007360917111922527*self.usr_np.log(temperature)**2 + -7.34999132088629e-05*self.usr_np.log(temperature)**3 + 2.6719789818844827e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.004256427358016503 + 0.0033253532707845043*self.usr_np.log(temperature) + -0.0009491326704636848*self.usr_np.log(temperature)**2 + 0.00011881898843723012*self.usr_np.log(temperature)**3 + -5.21976761107646e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.014576156075593425 + -0.01045520752523761*self.usr_np.log(temperature) + 0.002745254157141355*self.usr_np.log(temperature)**2 + -0.00029574934183211425*self.usr_np.log(temperature)**3 + 1.1611582133625754e-05*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0045483949118831106 + -0.0030695092928276426*self.usr_np.log(temperature) + 0.0007524360217193606*self.usr_np.log(temperature)**2 + -7.536672719962163e-05*self.usr_np.log(temperature)**3 + 2.74928995841195e-06*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.0036041946252137326 + -0.0029764317133841047*self.usr_np.log(temperature) + 0.0009079431693159895*self.usr_np.log(temperature)**2 + -0.00010602090367469384*self.usr_np.log(temperature)**3 + 4.449671995954592e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.001979066782775548 + 0.0008466494772890718*self.usr_np.log(temperature) + -3.7669754660562133e-07*self.usr_np.log(temperature)**2 + -1.1842471465202141e-05*self.usr_np.log(temperature)**3 + 8.794944288185929e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.002844270749279069 + -0.002487828229731366*self.usr_np.log(temperature) + 0.0008006603086921086*self.usr_np.log(temperature)**2 + -9.554464569128792e-05*self.usr_np.log(temperature)**3 + 4.075442327628586e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.00225956866781686 + 0.0010867500833295913*self.usr_np.log(temperature) + -7.29762106412335e-05*self.usr_np.log(temperature)**2 + -3.310215675439311e-06*self.usr_np.log(temperature)**3 + 5.203663600652946e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.014576156075593425 + -0.01045520752523761*self.usr_np.log(temperature) + 0.002745254157141355*self.usr_np.log(temperature)**2 + -0.00029574934183211425*self.usr_np.log(temperature)**3 + 1.1611582133625754e-05*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.007223792882702381 + 0.0046621286417665875*self.usr_np.log(temperature) + -0.00087866117653409*self.usr_np.log(temperature)**2 + 7.971305164992727e-05*self.usr_np.log(temperature)**3 + -2.5782770043946415e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0023036999702341645 + 0.0011142865066931617*self.usr_np.log(temperature) + -7.819008659227211e-05*self.usr_np.log(temperature)**2 + -2.8417207214378e-06*self.usr_np.log(temperature)**3 + 5.052783390750949e-07*self.usr_np.log(temperature)**4),
            ]),
            self._pyro_make_array([
                self.usr_np.sqrt(temperature)*temperature*(0.0026514846430530166 + -0.0019100110219294102*self.usr_np.log(temperature) + 0.0005039194873985993*self.usr_np.log(temperature)**2 + -5.448422948721137e-05*self.usr_np.log(temperature)**3 + 2.1461864939643698e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0011713797855077278 + -0.000973473615577208*self.usr_np.log(temperature) + 0.0002987686824672578*self.usr_np.log(temperature)**2 + -3.497731180583652e-05*self.usr_np.log(temperature)**3 + 1.4708557819154825e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.00237729552582244 + -0.0017310144290815796*self.usr_np.log(temperature) + 0.0004622408742451766*self.usr_np.log(temperature)**2 + -5.040404703461491e-05*self.usr_np.log(temperature)**3 + 2.0005929297812444e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0009562551925276512 + -0.0008238588219283307*self.usr_np.log(temperature) + 0.00026154843260841*self.usr_np.log(temperature)**2 + -3.1044891421602505e-05*self.usr_np.log(temperature)**3 + 1.3190163269443288e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0045483949118831106 + -0.0030695092928276426*self.usr_np.log(temperature) + 0.0007524360217193606*self.usr_np.log(temperature)**2 + -7.536672719962163e-05*self.usr_np.log(temperature)**3 + 2.74928995841195e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(-0.0023036999702341645 + 0.0011142865066931617*self.usr_np.log(temperature) + -7.819008659227211e-05*self.usr_np.log(temperature)**2 + -2.8417207214378e-06*self.usr_np.log(temperature)**3 + 5.052783390750949e-07*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*temperature*(0.0009517637604027024 + -0.0008221858860867466*self.usr_np.log(temperature) + 0.0002616548684685721*self.usr_np.log(temperature)**2 + -3.108746696762806e-05*self.usr_np.log(temperature)**3 + 1.3217663090033962e-06*self.usr_np.log(temperature)**4),
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

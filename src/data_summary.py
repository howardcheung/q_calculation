#!/usr/bin/python

"""
    This file contains scripts from machine readable data to human-friendly
    summaries from the water-side measurement of a heat exchanger
"""

import copy
import csv
import inspect
from math import sqrt
import pandas as pd
import pdb
from scipy.stats import t
from statistics import stdev

from CoolProp.CoolProp import PropsSI

import misc_func as misc

time_col_name = 'time_col'
uncer_str_end = '_uncer'
mean_str_end = '_mean'
dev_str_end = '_dev'
env_str_end = '_env'
prod_str = 'mdotdeltah'


def append_uncer_to_df(
    df_option, mea_col_name, abs_uncer=0.0, rel_uncer=0.0
):
    """
        This function appends uncertainty to the data in
        the DataFrame in df_option and returns df_option
        with new columns named in the pattern of (mea_col_name+'_uncer').

        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file.

        mea_col_name: string
            column name containing the data which uncertainty
            is going to be appended

        abs_uncer: float
            uncertainty from manufacturer of device measured in
            the same unit as the data in mea_col_name. Default 0.0.

        rel_uncer: float
            uncertainty from manufacturer of device measured as
            a ratio to the measurement in mea_col_name. Default 0.0.
    """

    # check inputs
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    if mea_col_name not in df.columns.values:
        raise TypeError(
            "DataFrame to "+inspect.stack()[0][3]+"() " +
            "does not have column "+mea_col_name+"."
        )
    if abs_uncer < 0.0:
        raise ValueError(
            "DataFrame to "+inspect.stack()[0][3]+"() " +
            "has a negative absolute uncertainty."
        )
    if rel_uncer < 0.0 or rel_uncer > 1.0:
        raise ValueError(
            "DataFrame to "+inspect.stack()[0][3]+"() " +
            "has an inappropriate relative uncertainty."
        )

    # create new column
    new_col_name = mea_col_name+uncer_str_end
    df[new_col_name] = [
        misc.sqrt_sum_of_squares(
            [abs_uncer, rel_uncer*df[mea_col_name][ind]]
        ) for ind in df.index
    ]

    # return df_option
    df_option.set(df)
    return df_option


def data_mean_cal(df_option, col_names, alpha=0.95):
    """
        This function calculates the mean of data stored in
        column col_names and pass the results (mean and uncertainty)
        to the 'details' attribute in the DataFrame. It returns
        df_option with the new attributes.

        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file.

        col_names: list
            list of strings for the names of the columns to
            be summarized with their mean and uncertainty of
            the mean

        alpha: float
            level of confidence interval you want in the uncertainty
            of the mean. Default 0.95
    """

    # check inputs
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    for col_name in col_names:
        if col_name not in df.columns.values:
            raise TypeError(
                "DataFrame to "+inspect.stack()[0][3]+"() " +
                "does not have column "+col_name+"."
            )
        if col_name+uncer_str_end not in df.columns.values:
            # make one column with zero uncertainty
            df_option = append_uncer_to_df(df_option, col_name, 0.0, 0.0)
            df = df_option.get()
    if not hasattr(df, 'details'):
        # make the details attribute if it does not exist
        df = misc.details_in_dataframe(df)

    # calculate the summary values
    len_df = df.shape[0]
    for col_name in col_names:
        df.details[col_name+mean_str_end] = df[col_name].mean()
        # set the uncertainty to be unavailable if it cannot be calculated
        df.details[col_name+mean_str_end+uncer_str_end] = float('inf')
        df.details[col_name+mean_str_end+uncer_str_end+dev_str_end] = \
            float('inf')
        df.details[col_name+mean_str_end+uncer_str_end+env_str_end] = \
            float('inf')
        expt_uncer = misc.sqrt_sum_of_squares(
            df[col_name+uncer_str_end].tolist()
        )
        if expt_uncer:
            expt_uncer.set(expt_uncer.get()/len_df)
            df.details[col_name+mean_str_end+uncer_str_end+dev_str_end] = \
                expt_uncer.get()
            env_uncer = df[col_name].std()*t.interval(
                alpha, len_df-1
            )[1]/len_df
            df.details[col_name+mean_str_end+uncer_str_end+env_str_end] = \
                env_uncer
            uncer_optionalvar = misc.sqrt_sum_of_squares(
                [expt_uncer, env_uncer]
            )
            if uncer_optionalvar:
                df.details[col_name+mean_str_end+uncer_str_end] = \
                    uncer_optionalvar.get()

    # return values
    df_option.set(df)
    return df_option


def cal_mdot(vdot, tvdot, vdot_uncer, tvdot_uncer, medium, full_output=False):
    """
        This function calculates the mass flow rate
        based on volumetric flow rate and temperature
        measured at the flow rate station. It returns
        two OptionalVariable(): "mdot" for mass flow rate and
        "mdot_uncer" for the uncertainty of the mass flow rate,
        and both are in kg/s.

        vdot: float
            volumetric flow rate in m3/s.

        tvdot: float
            temperature measurement in K at flow rate measurement
            station.

        vdot_uncer: float
            uncertainty of volumetric flow rate in m3/s.

        tvdot_uncer: float
            temperature measurement in K at flow rate measurement
            station.

        medium: string
            name of medium in the flow

        full_output: boolean
            whether output all three uncertainty components. Defaulted
            False (Uncertainty of equation of state, uncertainty of
            volumetric flow measurement and uncertainty of temperature
            measurement)
    """

    # change all inputs to ordinary variables
    # and test if they have errors
    vdot = misc.getOptionalVariable(vdot)
    tvdot = misc.getOptionalVariable(tvdot)
    vdot_uncer = misc.getOptionalVariable(vdot_uncer)
    tvdot_uncer = misc.getOptionalVariable(tvdot_uncer)

    # prepare calculation. If there is an error
    # set these outputs with the errors and return them
    mdot = misc.OptionalVariable()
    mdot_uncer = misc.OptionalVariable()

    # set uncertainty values of the EOS of water according to
    # Wagner et al. 2002
    rel_uncer_rho_eos = 0.02/100.0  # maximum
    # add for other medium later

    # start calculation
    try:
        rho = PropsSI('D', 'T', tvdot, 'Q', 0, medium)
        rho_uncer_expt = abs(
            PropsSI('D', 'T', tvdot*1.0001, 'Q', 0, medium)-rho
        )/tvdot/0.0001*tvdot_uncer
        rho_uncer_eos = rho*rel_uncer_rho_eos
        mdot.set(rho*vdot)
        mdot_uncer = misc.sqrt_sum_of_squares([
            rho_uncer_eos*vdot, rho*vdot_uncer, rho_uncer_expt*vdot
        ])
    except ValueError:
        uncer_temp = misc.OptionalVariable()
        uncer_temp.set(float('nan'))
        if not mdot:
            mdot = copy.deepcopy(uncer_temp)
        if not mdot_uncer:
            mdot_uncer = copy.deepcopy(uncer_temp)
    except Exception as e:
        if not mdot:
            mdot.setError(e)
        if not mdot_uncer:
            mdot_uncer.setError(e)

    # return the values
    if full_output:
        mdot_uncer_eos = misc.OptionalVariable()
        mdot_uncer_vdot = misc.OptionalVariable()
        mdot_uncer_temp = misc.OptionalVariable()
        mdot_uncer_eos.set(rho_uncer_eos*vdot)
        mdot_uncer_vdot.set(rho*vdot_uncer)
        mdot_uncer_temp.set(rho_uncer_expt*vdot)
        return mdot, mdot_uncer, mdot_uncer_eos, mdot_uncer_vdot, \
            mdot_uncer_temp
    else:
        return mdot, mdot_uncer


def cal_mdotdeltah_water(
    vdot, tvdot, tout, tin, vdot_uncer, tvdot_uncer,
    tout_uncer, tin_uncer, mdot=float('-inf'), mdot_uncer=float('-inf'),
    full_output=False
):
    """
        This function calculates mdot*(hout-hin) and
        the uncertainty propagated from the inputs for
        water flow. Returns "q" in OptionalVariable() for the
        value from mdot*(hout-hin) and "q_uncer" in OptionalVariable()
        for the uncertainty of the value mdot*(hout-hin). Both values
        are recorded in W.

        vdot: float
            volumetric flow rate in m3/s. If you are measuring
            mass flow rate in kg/s, set it to float('-inf') and enter
            mass flow rate at variable mdot.

        tvdot: float
            temperature measurement in K at flow rate measurement
            station. If you are measuring
            mass flow rate in kg/s, set it to 0.0.

        tout: float
            temperature measurement of water at outlet in K

        tin: float
            temperature measurement of water at inlet in K

        vdot_uncer: float
            uncertainty of volumetric flow rate in m3/s. If you
            are measuring mass flow rate in kg/s, set it to float('-inf')
            and enter the uncertainty of mass flow rate at
            variable mdot_uncer.

        tvdot_uncer: float
            temperature measurement in K at flow rate measurement
            station. If you are measuring mass flow rate in kg/s,
            set it to zero.

        tout_uncer: float
            uncertainty of temperature measurement of water at outlet in K

        tin_uncer: float
            uncertainty of temperature measurement of water at inlet in K

        mdot: float
            mass flow rate in kg/s. If you measure volumetric
            float rate, set it to float('-inf'). Default float('-inf').

        mdot_uncer: float
            uncertainty of mass flow rate in kg/s. If you measure volumetric
            float rate, set it to float('-inf'). Default float('-inf').

        full_output: boolean
            whether output all four uncertainty components. Defaulted
            False (Uncertainty from mass flow rate, uncertainty from
            outlet temperature, uncertainty from inlet temperature,
            and uncertainty from equation of state of enthalpy)
    """

    # change all inputs to ordinary variables
    # and test if they have errors
    vdot = misc.getOptionalVariable(vdot)
    tvdot = misc.getOptionalVariable(tvdot)
    tout = misc.getOptionalVariable(tout)
    tin = misc.getOptionalVariable(tin)
    vdot_uncer = misc.getOptionalVariable(vdot_uncer)
    tvdot_uncer = misc.getOptionalVariable(tvdot_uncer)
    tout_uncer = misc.getOptionalVariable(tout_uncer)
    tin_uncer = misc.getOptionalVariable(tin_uncer)
    mdot = misc.getOptionalVariable(mdot)
    mdot_uncer = misc.getOptionalVariable(mdot_uncer)

    # prepare calculation. If there is an error
    # set these outputs with the errors and return them
    q = misc.OptionalVariable()
    q_uncer = misc.OptionalVariable()
    medium = 'Water'

    # set uncertainty values of the EOS of water according to
    # Wagner et al. 2002
    rel_uncer_deltah = 0.5/100.0  # less than or equal to cp uncertainty

    # start calculation
    try:
        # check if calculation of mdot and mdot_uncer is needed
        if vdot != float('inf'):
            mdot, mdot_uncer = cal_mdot(
                vdot, tvdot, vdot_uncer, tvdot_uncer, medium
            )
            if not (mdot_uncer and mdot):
                q.setError(mdot.getError())
                q_uncer.setError(mdot_uncer.getError())
                if full_output:
                    q_uncer_mdot = misc.OptionalVariable()
                    q_uncer_tin = misc.OptionalVariable()
                    q_uncer_tout = misc.OptionalVariable()
                    q_uncer_heos = misc.OptionalVariable()
                    q_uncer_mdot.setError(mdot_uncer.getError())
                    q_uncer_tin.setError(mdot_uncer.getError())
                    q_uncer_tout.setError(mdot_uncer.getError())
                    q_uncer_heos.setError(mdot_uncer.getError())
                    return q, q_uncer, q_uncer_mdot, q_uncer_tout, \
                        q_uncer_tin, q_uncer_heos
                else:
                    return q, q_uncer
            else:
                mdot = mdot.get()
                mdot_uncer = mdot_uncer.get()

        # get enthalpy differences
        hout = PropsSI('H', 'T', tout, 'Q', 0, medium)
        hin = PropsSI('H', 'T', tin, 'Q', 0, medium)
        deltah = hout-hin

        # calculate q
        q.set(mdot*deltah)

        # calculate uncertainty of deltah
        hout_uncer = abs(
            PropsSI('H', 'T', tout*1.0001, 'Q', 0, medium)-hout
        )/tout/0.0001*tout_uncer
        hin_uncer = abs(
            PropsSI('H', 'T', tin*1.0001, 'Q', 0, medium)-hin
        )/tin/0.0001*tin_uncer
        deltah_uncer_eos = rel_uncer_deltah*deltah
        deltah_uncer = misc.sqrt_sum_of_squares([
            hout_uncer, hin_uncer, deltah_uncer_eos
        ])
        if not deltah_uncer:
            q_uncer.setError(deltah_uncer.getError())
        else:
            deltah_uncer = deltah_uncer.get()
            q_uncer = misc.sqrt_sum_of_squares([
                mdot_uncer*deltah, deltah_uncer*mdot
            ])
    except ValueError:  # temperature out of range
        uncer_temp = misc.OptionalVariable()
        uncer_temp.set(float('nan'))
        if not q:
            q = copy.deepcopy(uncer_temp)
        if not q_uncer:
            q_uncer = copy.deepcopy(uncer_temp)
    except Exception as e:
        if not q:
            q.setError(e)
        if not q_uncer:
            q_uncer.setError(e)

    # return the values
    if full_output:
        q_uncer_mdot = misc.OptionalVariable()
        q_uncer_tin = misc.OptionalVariable()
        q_uncer_tout = misc.OptionalVariable()
        q_uncer_heos = misc.OptionalVariable()
        q_uncer_mdot.set(mdot_uncer*deltah)
        q_uncer_tin.set(hin_uncer*mdot)
        q_uncer_tout.set(hout_uncer*mdot)
        q_uncer_heos.set(deltah_uncer_eos*mdot)
        return q, q_uncer, q_uncer_mdot, q_uncer_tout, q_uncer_tin, \
            q_uncer_heos
    else:
        return q, q_uncer


def uncer_comp_vdot_cal_mdotdeltah_liquid(
    vdot, tvdot, tout, tin, vdot_uncer, tvdot_uncer,
    tout_uncer, tin_uncer, medium='Water'
):
    """
        This function calculates mdot*(hout-hin) and
        the uncertainty propagated from the inputs for
        water flow. Returns "q" in OptionalVariable() for the
        value from mdot*(hout-hin), "q_uncer" in OptionalVariable()
        for the uncertainty of the value mdot*(hout-hin), "q_uncer_vdot"
        in OptionalVariable() for uncertainty propagated from vdot,
        "q_uncer_tvdot" for uncertainty propagated from tvdot,
        "q_uncer_tout" for uncertainty propagated from tout,
        "q_uncer_tin" for uncertainty propagated from tin,
        "q_uncer_rhoeos" for uncertainty propagated from equation of state
        of density, and "q_uncer_heos" for uncertainty propagated from
        equation of state of enthalpy. This function cannot process
        cases which mass flow rates are measured directly.

        vdot: float
            volumetric flow rate in m3/s. If you are measuring
            mass flow rate in kg/s, set it to float('-inf') and enter
            mass flow rate at variable mdot.

        tvdot: float
            temperature measurement in K at flow rate measurement
            station. If you are measuring
            mass flow rate in kg/s, set it to 0.0.

        tout: float
            temperature measurement of water at outlet in K

        tin: float
            temperature measurement of water at inlet in K

        vdot_uncer: float
            uncertainty of volumetric flow rate in m3/s. If you
            are measuring mass flow rate in kg/s, set it to float('-inf')
            and enter the uncertainty of mass flow rate at
            variable mdot_uncer.

        tvdot_uncer: float
            temperature measurement in K at flow rate measurement
            station. If you are measuring mass flow rate in kg/s,
            set it to zero.

        tout_uncer: float
            uncertainty of temperature measurement of water at outlet in K

        tin_uncer: float
            uncertainty of temperature measurement of water at inlet in K

        medium: string
            name of the medium in the heat exchanger. Defaulted 'Water'
    """

    # change all inputs to ordinary variables
    # and test if they have errors
    vdot = misc.getOptionalVariable(vdot)
    tvdot = misc.getOptionalVariable(tvdot)
    tout = misc.getOptionalVariable(tout)
    tin = misc.getOptionalVariable(tin)
    vdot_uncer = misc.getOptionalVariable(vdot_uncer)
    tvdot_uncer = misc.getOptionalVariable(tvdot_uncer)
    tout_uncer = misc.getOptionalVariable(tout_uncer)
    tin_uncer = misc.getOptionalVariable(tin_uncer)

    # prepare calculation. If there is an error
    # set these outputs with the errors and return them
    q = misc.OptionalVariable()
    q_uncer = misc.OptionalVariable()
    q_uncer_comps = {}
    comp_names = ['vdot', 'tvdot', 'tout', 'tin', 'rhoeos', 'heos']
    for comp_name in comp_names:
        q_uncer_comps[comp_name] = misc.OptionalVariable()

    # set uncertainty values of the EOS of water according to
    # Wagner et al. 2002
    rel_uncer_deltah = 0.5/100.0  # less than or equal to cp uncertainty

    # start calculation
    try:
        # calculate mass flow rate and related uncertainty components
        mdot, mdot_uncer, mdot_uncer_eos, mdot_uncer_vdot, \
            mdot_uncer_temp = cal_mdot(
                vdot, tvdot, vdot_uncer, tvdot_uncer, medium,
                full_output=True
            )
        if not (mdot_uncer and mdot):
            q.setError(mdot.getError())
            q_uncer.setError(mdot_uncer.getError())
            for comp_name in comp_names:
                q_uncer_comps[comp_name].setError(mdot_uncer.getError())
            return q, q_uncer, q_uncer_comps['vdot'], q_uncer_comps['tvdot'], \
                q_uncer_comps['tout'], q_uncer_comps['tin'], \
                q_uncer_comps['rhoeos'], q_uncer_comps['heos']
        mdot = mdot.get()
        mdot_uncer = mdot_uncer.get()

        # get enthalpy differences
        hout = PropsSI('H', 'T', tout, 'Q', 0, medium)
        hin = PropsSI('H', 'T', tin, 'Q', 0, medium)
        deltah = hout-hin

        # calculate q
        q.set(mdot*deltah)

        # calculate uncertainty of deltah
        hout_uncer = abs(
            PropsSI('H', 'T', tout*1.001, 'Q', 0, medium)-hout
        )/tout/0.001*tout_uncer
        hin_uncer = abs(
            PropsSI('H', 'T', tin*1.001, 'Q', 0, medium)-hin
        )/tin/0.001*tin_uncer
        deltah_uncer_eos = rel_uncer_deltah*deltah
        deltah_uncer = misc.sqrt_sum_of_squares([
            hout_uncer, hin_uncer, deltah_uncer_eos
        ])
        if not deltah_uncer:
            q_uncer.setError(deltah_uncer.getError())
            for comp_name in comp_names:
                q_uncer_comps[comp_name].setError(deltah_uncer.getError())
        else:
            deltah_uncer = deltah_uncer.get()
            q_uncer = misc.sqrt_sum_of_squares([
                mdot_uncer*deltah, deltah_uncer*mdot
            ])
            q_uncer_comps['vdot'].set(mdot_uncer_vdot.get()*deltah)
            q_uncer_comps['tvdot'].set(mdot_uncer_temp.get()*deltah)
            q_uncer_comps['tout'].set(hout_uncer*mdot)
            q_uncer_comps['tin'].set(hin_uncer*mdot)
            q_uncer_comps['rhoeos'].set(mdot_uncer_eos.get()*deltah)
            q_uncer_comps['heos'].set(deltah_uncer_eos*mdot)
    except ValueError:  # temperature out of range
        uncer_temp = misc.OptionalVariable()
        uncer_temp.set(float('nan'))
        if not q:
            q = copy.deepcopy(uncer_temp)
        if not q_uncer:
            q_uncer = copy.deepcopy(uncer_temp)
        for comp_name in comp_names:
            if not q_uncer_comps[comp_name]:
                q_uncer_comps[comp_name] = copy.deepcopy(uncer_temp)
    except Exception as e:
        if not q:
            q.setError(e)
        if not q_uncer:
            q_uncer.setError(e)
        for comp_name in comp_names:
            if not q_uncer_comps[comp_name]:
                q_uncer_comps[comp_name].setError(e)

    return q, q_uncer, q_uncer_comps['vdot'], q_uncer_comps['tvdot'], \
        q_uncer_comps['tout'], q_uncer_comps['tin'], \
        q_uncer_comps['rhoeos'], q_uncer_comps['heos']


def uncer_comp_mdot_cal_mdotdeltah_liquid(
    mdot, tout, tin, mdot_uncer, tout_uncer, tin_uncer, medium
):
    """
        This function calculates mdot*(hout-hin) and
        the uncertainty propagated from the inputs for
        water flow. Returns "q" in OptionalVariable() for the
        value from mdot*(hout-hin), "q_uncer" in OptionalVariable()
        for the uncertainty of the value mdot*(hout-hin), "q_uncer_mdot"
        in OptionalVariable() for uncertainty propagated from mdot,
        "q_uncer_tout" for uncertainty propagated from tout,
        "q_uncer_tin" for uncertainty propagated from tin and
        "q_uncer_heos" for uncertainty propagated from
        equation of state of enthalpy. This function cannot process
        cases which volumetric flow rates are measured.

        mdot: float
            mass flow rate in kg/s.

        tout: float
            temperature measurement of water at outlet in K

        tin: float
            temperature measurement of water at inlet in K

        vdot_uncer: float
            uncertainty of volumetric flow rate in m3/s. If you
            are measuring mass flow rate in kg/s, set it to float('-inf')
            and enter the uncertainty of mass flow rate at
            variable mdot_uncer.

        tvdot_uncer: float
            temperature measurement in K at flow rate measurement
            station. If you are measuring mass flow rate in kg/s,
            set it to zero.

        tout_uncer: float
            uncertainty of temperature measurement of water at outlet in K

        mdot_uncer: float
            uncertainty of mass flow rate in kg/s.

        tin_uncer: float
            uncertainty of temperature measurement of water at inlet in K

        medium: string
            name of the medium in the heat exchanger. Defaulted 'Water'
    """

    # change all inputs to ordinary variables
    # and test if they have errors
    mdot = misc.getOptionalVariable(mdot)
    tout = misc.getOptionalVariable(tout)
    tin = misc.getOptionalVariable(tin)
    mot_uncer = misc.getOptionalVariable(mdot_uncer)
    tout_uncer = misc.getOptionalVariable(tout_uncer)
    tin_uncer = misc.getOptionalVariable(tin_uncer)

    # prepare calculation. If there is an error
    # set these outputs with the errors and return them
    q = misc.OptionalVariable()
    q_uncer = misc.OptionalVariable()
    q_uncer_comps = {}
    comp_names = ['mvdot', 'tout', 'tin', 'heos']
    for comp_name in comp_names:
        q_uncer_comps[comp_name] = misc.OptionalVariable()

    # set uncertainty values of the EOS of water according to
    # Wagner et al. 2002
    rel_uncer_deltah = 0.5/100.0  # less than or equal to cp uncertainty

    # start calculation
    try:
        # get enthalpy differences
        hout = PropsSI('H', 'T', tout, 'Q', 0, medium)
        hin = PropsSI('H', 'T', tin, 'Q', 0, medium)
        deltah = hout-hin

        # calculate q
        q.set(mdot*deltah)

        # calculate uncertainty of deltah
        hout_uncer = abs(
            PropsSI('H', 'T', tout*1.001, 'Q', 0, medium)-hout
        )/tout/0.001*tout_uncer
        hin_uncer = abs(
            PropsSI('H', 'T', tin*1.001, 'Q', 0, medium)-hin
        )/tin/0.001*tin_uncer
        deltah_uncer_eos = rel_uncer_deltah*deltah
        deltah_uncer = misc.sqrt_sum_of_squares([
            hout_uncer, hin_uncer, deltah_uncer_eos
        ])
        if not deltah_uncer:
            q_uncer.setError(deltah_uncer.getError())
            for comp_name in comp_names:
                q_uncer_comps[comp_name].setError(deltah_uncer.getError())
        else:
            deltah_uncer = deltah_uncer.get()
            q_uncer = misc.sqrt_sum_of_squares([
                mdot_uncer*deltah, deltah_uncer*mdot
            ])
            q_uncer_comps['mdot'].set(mdot_uncer*deltah)
            q_uncer_comps['tout'].set(hout_uncer*mdot)
            q_uncer_comps['tin'].set(hin_uncer*mdot)
            q_uncer_comps['heos'].set(deltah_uncer_eos*mdot)
    except ValueError:  # temperature out of range
        uncer_temp = misc.OptionalVariable()
        uncer_temp.set(float('nan'))
        if not q:
            q = copy.deepcopy(uncer_temp)
        if not q_uncer:
            q_uncer = copy.deepcopy(uncer_temp)
        for comp_name in comp_names:
            if not q_uncer_comps[comp_name]:
                q_uncer_comps[comp_name] = copy.deepcopy(uncer_temp)
    except Exception as e:
        if not q:
            q.setError(e)
        if not q_uncer:
            q_uncer.setError(e)
        for comp_name in comp_names:
            if not q_uncer_comps[comp_name]:
                q_uncer_comps[comp_name].setError(e)

    return q, q_uncer, q_uncer_comps['mdot'], \
        q_uncer_comps['tout'], q_uncer_comps['tin'], \
        q_uncer_comps['heos']


def cal_q_from_sample_result(
    df_option, vdot_col_name, tvdot_col_name, tout_col_name,
    tin_col_name, hx_name='hx', mdot_col_name=''
):
    """
        This function calculates the heat transfer rate and its
        uncertainty of a heat exchanger from the mean observations
        of properties. It stores the information to the 'details'
        attribute of the DataFrame and returns df_option with the new
        entries to the 'details' attribute.

        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file with
            mean values of observations stroed in the 'details'
            attribute

        vdot_col_name: string
            column name for volumetric flow rate value in m3/s.
            If mass flow rate is measured, set this to empty string and
            put the mass flow rate column name to mdot_col_name

        tvdot_col_name: string
            column name for temperature value at flow rate measurement
            station in K. Set this to empty string if mass flow rate
            is measured

        tout_col_name: string
            column name for temperature value at heat exchanger outlet
            station in K

        tin_col_name: string
            column name for temperature value at heat exchanger outlet
            station in K

        hx_name: string
            name of heat exchanger. Default 'hx'

        mdot_col_name: string
            column name for volumetric flow rate value in kg/s.
            Default empty.
    """

    # check inputs
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    # need code to check column names

    # calculate q and q_result
    if vdot_col_name is not '':
        q, q_uncer = cal_mdotdeltah_water(
            vdot=df.details[vdot_col_name+mean_str_end],
            tvdot=df.details[tvdot_col_name+mean_str_end],
            tout=df.details[tout_col_name+mean_str_end],
            tin=df.details[tin_col_name+mean_str_end],
            vdot_uncer=df.details[vdot_col_name+mean_str_end+uncer_str_end],
            tvdot_uncer=df.details[tvdot_col_name+mean_str_end+uncer_str_end],
            tout_uncer=df.details[tout_col_name+mean_str_end+uncer_str_end],
            tin_uncer=df.details[tin_col_name+mean_str_end+uncer_str_end]
        )
    else:
        q, q_uncer = cal_mdotdeltah_water(
            vdot=float('-inf'), tvdot=float('-inf'),
            tout=df.details[tout_col_name+mean_str_end],
            tin=df.details[tin_col_name+mean_str_end],
            vdot_uncer=df.detailsfloat('-inf'),
            tvdot_uncer=df.detailsfloat('-inf'),
            tout_uncer=df.details[tout_col_name+mean_str_end+uncer_str_end],
            tin_uncer=df.details[tin_col_name+mean_str_end+uncer_str_end],
            mdot=df.details[mdot_col_name+mean_str_end],
            mdot_uncer=df.details[mdot_col_name+mean_str_end+uncer_str_end],
        )

    # check error
    if not q:
        raise q.getError()
    if not q_uncer:
        raise q_uncer.getError()

    # return values
    df.details['q_'+hx_name+'_mean_obs'] = q.get()
    df.details['q_uncer_'+hx_name+'_mean_obs'] = q_uncer.get()
    df_option.set(df)
    return df_option


def cal_q_with_uncer_from_sample_result(
    df_option, vdot_col_name, tvdot_col_name, tout_col_name,
    tin_col_name, hx_name='hx', mdot_col_name='', medium='Water'
):
    """
        This function calculates the heat transfer rate and its
        uncertainty of a heat exchanger from the mean observations
        of properties. It stores the information to the 'details'
        attribute of the DataFrame and returns df_option with the new
        entries to the 'details' attribute. It also calculates the
        effect of uncertainty propagated from the sources to the final
        heat transfer rate.

        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file with
            mean values of observations stroed in the 'details'
            attribute

        vdot_col_name: string
            column name for volumetric flow rate value in m3/s.
            If mass flow rate is measured, set this to empty string and
            put the mass flow rate column name to mdot_col_name

        tvdot_col_name: string
            column name for temperature value at flow rate measurement
            station in K. Set this to empty string if mass flow rate
            is measured

        tout_col_name: string
            column name for temperature value at heat exchanger outlet
            station in K

        tin_col_name: string
            column name for temperature value at heat exchanger outlet
            station in K

        hx_name: string
            name of heat exchanger. Default 'hx'

        mdot_col_name: string
            column name for volumetric flow rate value in kg/s.
            Default empty.

        medium: string
            name of the medium in the heat exchanger. Defaulted 'Water'
    """

    # check inputs
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    # need code to check column names

    # calculate q and q_result
    if vdot_col_name is not '':
        q, q_uncer, q_uncer_vdot, q_uncer_tvdot, \
            q_uncer_tout, q_uncer_tin, \
            q_uncer_rhoeos, q_uncer_heos = \
            uncer_comp_vdot_cal_mdotdeltah_liquid(
                vdot=df.details[vdot_col_name+mean_str_end],
                tvdot=df.details[tvdot_col_name+mean_str_end],
                tout=df.details[tout_col_name+mean_str_end],
                tin=df.details[tin_col_name+mean_str_end],
                vdot_uncer=df.details[
                    vdot_col_name+mean_str_end+uncer_str_end
                ],
                tvdot_uncer=df.details[
                    tvdot_col_name+mean_str_end+uncer_str_end
                ],
                tout_uncer=df.details[
                    tout_col_name+mean_str_end+uncer_str_end
                ],
                tin_uncer=df.details[
                    tin_col_name+mean_str_end+uncer_str_end
                ], medium=medium
            )
    else:
        q, q_uncer, q_uncer_mdot, \
            q_uncer_tout, q_uncer_tin, \
            q_uncer_heos = uncer_comp_mdot_cal_mdotdeltah_liquid(
                mdot=df.details[mdot_col_name+mean_str_end],
                tout=df.details[tout_col_name+mean_str_end],
                tin=df.details[tin_col_name+mean_str_end],
                mdot_uncer=df.details[
                    mdot_col_name+mean_str_end+uncer_str_end
                ],
                tout_uncer=df.details[
                    tout_col_name+mean_str_end+uncer_str_end
                ],
                tin_uncer=df.details[
                    tin_col_name+mean_str_end+uncer_str_end
                ], medium=medium
            )

    # check error
    if not q:
        raise q.getError()
    if not q_uncer:
        raise q_uncer.getError()

    # return values
    df.details['q_'+hx_name+'_mean_obs'] = q.get()
    df.details['q_uncer_'+hx_name+'_mean_obs'] = q_uncer.get()
    if vdot_col_name is not '':
        df.details['q_uncer_'+hx_name+'_mean_obs_vdot'] = q_uncer_vdot.get()
        df.details['q_uncer_'+hx_name+'_mean_obs_tvdot'] = q_uncer_tvdot.get()
        df.details['q_uncer_'+hx_name+'_mean_obs_rhoeos'] = \
            q_uncer_rhoeos.get()
        df.details['rel_q_uncer_'+hx_name+'_mean_obs_vdot'] = \
            (q_uncer_vdot.get()/q_uncer.get())**2
        df.details['rel_q_uncer_'+hx_name+'_mean_obs_tvdot'] = \
            (q_uncer_tvdot.get()/q_uncer.get())**2
        df.details['rel_q_uncer_'+hx_name+'_mean_obs_rhoeos'] = \
            (q_uncer_rhoeos.get()/q_uncer.get())**2
    else:
        df.details['q_uncer_'+hx_name+'_mean_obs_mdot'] = \
            q_uncer_mdot.get()
        df.details['rel_q_uncer_'+hx_name+'_mean_obs_mdot'] = \
            (q_uncer_mdot.get()/q_uncer.get())**2
    df.details['q_uncer_'+hx_name+'_mean_obs_tout'] = q_uncer_tout.get()
    df.details['q_uncer_'+hx_name+'_mean_obs_tin'] = q_uncer_tin.get()
    df.details['q_uncer_'+hx_name+'_mean_obs_heos'] = q_uncer_heos.get()
    df.details['rel_q_uncer_'+hx_name+'_mean_obs_tout'] = \
        (q_uncer_tout.get()/q_uncer.get())**2
    df.details['rel_q_uncer_'+hx_name+'_mean_obs_tin'] = \
        (q_uncer_tin.get()/q_uncer.get())**2
    df.details['rel_q_uncer_'+hx_name+'_mean_obs_heos'] = \
        (q_uncer_heos.get()/q_uncer.get())**2
    df_option.set(df)
    return df_option


def cal_mdotdeltah_per_time(
    df_option, vdot_col_name, tvdot_col_name, tout_col_name,
    tin_col_name, hx_name='hx', mdot_col_name='', medium='Water'
):
    """
        This function calculates the instantaneous mdot*(hout-hin)
        and its uncertainty at each time stamp. The function stores
        append them to the DataFrame in columns 'mdotdeltah' and
        'mdotdeltah_uncer' in W and returns df_option.

        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file

        vdot_col_name: string
            column name for volumetric flow rate value in m3/s.
            If mass flow rate is measured, set this to empty string and
            put the mass flow rate column name to mdot_col_name

        tvdot_col_name: string
            column name for temperature value at flow rate measurement
            station in K. Set this to empty string if mass flow rate
            is measured

        tout_col_name: string
            column name for temperature value at heat exchanger outlet
            station in K

        tin_col_name: string
            column name for temperature value at heat exchanger outlet
            station in K

        deltat: float
            period of measurement per sample in seconds. If you want to
            the mean of instantaneous heat transfer rate, set it to 0.
            Default 10.

        hx_name: string
            name of heat exchanger that is under analyzed. Default 'hx'

        mdot_col_name: string
            column name for volumetric flow rate value in kg/s.
            Default empty.
    """

    # check inputs
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    # need code to check column names

    # prepare calculation
    len_df = df.shape[0]
    new_col_name = prod_str+"_"+hx_name
    df[new_col_name] = misc.new_optionalvariable_vector(len_df)
    df[new_col_name+uncer_str_end] = \
        misc.new_optionalvariable_vector(len_df)

    # calculation starts
    for ind in df.index:
        # calculate q and q_result
        if vdot_col_name is not '':
            q, q_uncer = cal_mdotdeltah_water(
                vdot=df[vdot_col_name][ind], tvdot=df[tvdot_col_name][ind],
                tout=df[tout_col_name][ind], tin=df[tin_col_name][ind],
                vdot_uncer=df[vdot_col_name+uncer_str_end][ind],
                tvdot_uncer=df[tvdot_col_name+uncer_str_end][ind],
                tout_uncer=df[tout_col_name+uncer_str_end][ind],
                tin_uncer=df[tin_col_name+uncer_str_end][ind]
            )
        else:
            q, q_uncer = cal_mdotdeltah_water(
                vdot=float('-inf'), tvdot=float('-inf'),
                tout=df[tout_col_name][ind], tin=df[tin_col_name][ind],
                vdot_uncer=df[vdot_col_name+uncer_str_end][ind],
                tvdot_uncer=df[tvdot_col_name+uncer_str_end][ind],
                tout_uncer=df[tout_col_name+uncer_str_end][ind],
                tin_uncer=df[tin_col_name+uncer_str_end][ind],
                mdot=df[mdot_col_name],
                mdot_uncer=df[mdot_col_name+uncer_str_end],
            )
        df.loc[ind, new_col_name] = copy.deepcopy(q)
        df.loc[ind, new_col_name+uncer_str_end] = copy.deepcopy(q_uncer)

    # return values
    df_option.set(df)
    return df_option


def cal_mdotdeltah_and_uncer_comp_per_time(
    df_option, vdot_col_name, tvdot_col_name, tout_col_name,
    tin_col_name, hx_name='hx', mdot_col_name='', medium='Water'
):
    """
        This function calculates the instantaneous mdot*(hout-hin)
        and its uncertainty at each time stamp. The function stores
        append them to the DataFrame in columns 'mdotdeltah' and
        'mdotdeltah_uncer' in W and returns df_option. It also calcualtes
        the uncertainty components of mdotdetalh at each time instant.

        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file

        vdot_col_name: string
            column name for volumetric flow rate value in m3/s.
            If mass flow rate is measured, set this to empty string and
            put the mass flow rate column name to mdot_col_name

        tvdot_col_name: string
            column name for temperature value at flow rate measurement
            station in K. Set this to empty string if mass flow rate
            is measured

        tout_col_name: string
            column name for temperature value at heat exchanger outlet
            station in K

        tin_col_name: string
            column name for temperature value at heat exchanger outlet
            station in K

        deltat: float
            period of measurement per sample in seconds. If you want to
            the mean of instantaneous heat transfer rate, set it to 0.
            Default 10.

        hx_name: string
            name of heat exchanger that is under analyzed. Default 'hx'

        mdot_col_name: string
            column name for volumetric flow rate value in kg/s.
            Default empty.

        medium: string
            name of the medium in the heat exchanger. Defaulted 'Water'
    """

    # check inputs
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    # need code to check column names

    # prepare calculation
    len_df = df.shape[0]
    new_col_name = prod_str+"_"+hx_name

    df[new_col_name] = misc.new_optionalvariable_vector(len_df)
    df[new_col_name+uncer_str_end] = \
        misc.new_optionalvariable_vector(len_df)
    if vdot_col_name is not '':
        df[new_col_name+uncer_str_end+'_vdot'] = \
            misc.new_optionalvariable_vector(len_df)
        df[new_col_name+uncer_str_end+'_tvdot'] = \
            misc.new_optionalvariable_vector(len_df)
        df[new_col_name+uncer_str_end+'_rhoeos'] = \
            misc.new_optionalvariable_vector(len_df)
    else:
        df[new_col_name+uncer_str_end+'_mdot'] = \
            misc.new_optionalvariable_vector(len_df)
    df[new_col_name+uncer_str_end+'_tin'] = \
        misc.new_optionalvariable_vector(len_df)
    df[new_col_name+uncer_str_end+'_tout'] = \
        misc.new_optionalvariable_vector(len_df)
    df[new_col_name+uncer_str_end+'_heos'] = \
        misc.new_optionalvariable_vector(len_df)

    # calculation starts
    for ind in df.index:
        # calculate q and q_result
        if vdot_col_name is not '':
            q, q_uncer, q_uncer_vdot, q_uncer_tvdot, \
                q_uncer_tout, q_uncer_tin, q_uncer_rhoeos, q_uncer_heos = \
                uncer_comp_vdot_cal_mdotdeltah_liquid(
                    vdot=df[vdot_col_name][ind], tvdot=df[tvdot_col_name][ind],
                    tout=df[tout_col_name][ind], tin=df[tin_col_name][ind],
                    vdot_uncer=df[vdot_col_name+uncer_str_end][ind],
                    tvdot_uncer=df[tvdot_col_name+uncer_str_end][ind],
                    tout_uncer=df[tout_col_name+uncer_str_end][ind],
                    tin_uncer=df[tin_col_name+uncer_str_end][ind],
                    medium=medium
                )
            df.loc[ind, new_col_name+uncer_str_end+'_vdot'] = \
                copy.deepcopy(q_uncer_vdot)
            df.loc[ind, new_col_name+uncer_str_end+'_tvdot'] = \
                copy.deepcopy(q_uncer_tvdot)
            df.loc[ind, new_col_name+uncer_str_end+'_rhoeos'] = \
                copy.deepcopy(q_uncer_rhoeos)
        else:
            q, q_uncer, q_uncer_mdot, \
                q_uncer_tout, q_uncer_tin, \
                q_uncer_heos = uncer_comp_mdot_cal_mdotdeltah_liquid(
                    mdot=df[mdot_col_name][ind],
                    tout=df[tout_col_name][ind], tin=df[tin_col_name][ind],
                    mdot_uncer=df[mdot_col_name+uncer_str_end][ind],
                    tout_uncer=df[tout_col_name+uncer_str_end][ind],
                    tin_uncer=df[tin_col_name+uncer_str_end][ind],
                    medium=medium
                )
            df.loc[ind, new_col_name+uncer_str_end+'_mdot'] = \
                copy.deepcopy(q_uncer_mdot)

        df.loc[ind, new_col_name] = copy.deepcopy(q)
        df.loc[ind, new_col_name+uncer_str_end] = copy.deepcopy(q_uncer)
        df.loc[ind, new_col_name+uncer_str_end+'_tin'] = \
            copy.deepcopy(q_uncer_tin)
        df.loc[ind, new_col_name+uncer_str_end+'_tout'] = \
            copy.deepcopy(q_uncer_tout)
        df.loc[ind, new_col_name+uncer_str_end+'_heos'] = \
            copy.deepcopy(q_uncer_heos)

    # return values
    df_option.set(df)
    return df_option


def cal_q_and_uncer_comp_from_ind_mea(
    df_option, deltat=0, hx_name='hx', alpha=0.95
):
    """
        This function integrates the mdotdeltah column result
        according to the user-defined period per sample. The heat
        transfer rates from each sample are averaged to obtain the
        mean heat transfer rate. It returns df_option with
        the 'details' attribute contain extra information that summarizes
        the data point with heat transfer rate and its
        uncertainty with deltat in its name. It also integrates the uncertainty
        components as well

        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file

        deltat: float
            period of measurement per sample in seconds. If you want to
            the mean of instantaneous heat transfer rate, set it to 0.
            Default 10.

        hx_name: string
            name of heat exchanger that is under analyzed. Default 'hx'

        alpha: float
            level of confidence interval you want in the uncertainty
            of the mean. Default 0.95
    """

    # check inputs
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    if deltat < 0:
        raise ValueError(
            "deltat to "+inspect.stack()[0][3]+"() is negative."
        )
    data_col_name = prod_str+"_"+hx_name
    uncer_col_name = data_col_name+uncer_str_end
    uncer_add_names = []
    for tempkey in list(df.columns.values):
        if uncer_col_name in tempkey:
            uncer_add_names.append(tempkey)
    if data_col_name not in df.columns.values:
        raise IndexError(
            inspect.stack()[0][3]+"() cannot find column" +
            data_col_name+". Please run cal_mdotdeltah_per_time()" +
            " to generate the column before using "+inspect.stack()[0][3]+"()"
        )
    # need code to check column names

    # identify where the integration is needed
    q_ind = []
    q_uncer_inds = {}
    for name in uncer_add_names:
        q_uncer_inds[name] = []
    if deltat > 0:  # integration by trapezoidal rule required
        time_str = df.details[time_col_name]
        end_ind = df.shape[0]-1
        beg_ind = 0
        acc = 0.0
        acc_uncer_sqs = {}
        for name in uncer_add_names:
            acc_uncer_sqs[name] = 0.0
        acc_time = 0.0
        while beg_ind < end_ind:
            # begin integration
            beg_time = df[time_str][df.index[beg_ind]]
            end_time = beg_time+deltat
            acc = 0.5*df[data_col_name][df.index[beg_ind]].get()
            if beg_ind < end_ind:
                deltaT = df[time_str][df.index[beg_ind+1]] -\
                    df[time_str][df.index[beg_ind]]
            else:
                deltaT = df[time_str][df.index[beg_ind]] -\
                    df[time_str][df.index[beg_ind-1]]
            acc = acc*deltaT
            for name in uncer_add_names:
                acc_uncer_sqs[name] = (
                    0.5*df[name][df.index[beg_ind]].get()*deltaT
                )**2
            while beg_ind < end_ind:
                beg_ind = beg_ind+1
                if beg_ind < end_ind and \
                        df[time_str][df.index[beg_ind+1]] <= end_time:
                    deltaT = df[time_str][df.index[beg_ind+1]] -\
                        df[time_str][df.index[beg_ind-1]]
                    acc = acc +\
                        df[data_col_name][df.index[beg_ind]].get()*deltaT/2.
                    for name in uncer_add_names:
                        acc_uncer_sqs[name] = acc_uncer_sqs[name]+(
                            df[name][df.index[beg_ind]].get()*deltaT/2.
                        )**2
                else:
                    deltaT = df[time_str][df.index[beg_ind]] -\
                        df[time_str][df.index[beg_ind-1]]
                    acc = acc +\
                        df[data_col_name][df.index[beg_ind]].get()*deltaT/2.
                    for name in uncer_add_names:
                        acc_uncer_sqs[name] = acc_uncer_sqs[name]+(
                            df[name][df.index[beg_ind]].get()*deltaT/2.
                        )**2
                    break

            # divide the integrated heat transfer by time
            acc_time = df[time_str][df.index[beg_ind]]-beg_time
            q_ind.append(acc/acc_time)
            for name in uncer_add_names:
                q_uncer_inds[name].append(sqrt(acc_uncer_sqs[name])/acc_time)

    else:  # mean of instantaneous heat transfer rate is needed
        q_ind = [df[data_col_name][ind].get() for ind in df.index]
        for name in uncer_add_names:
            q_uncer_inds[name] = [df[name][ind].get() for ind in df.index]

    # taking the mean values for result
    if deltat == 0:
        time_std_end = '_inst'
    else:
        time_std_end = '_'+str(int(deltat))+'s'
    len_q_ind = len(q_ind)
    df.details['q_'+hx_name+'_mean'+time_std_end] = sum(q_ind)/len_q_ind

    # getting the variation of samples as environmental noise effect
    zero_order_sq = sum(
        [(value/len_q_ind)**2 for value in q_uncer_inds[uncer_col_name]]
    )
    first_order_sq = (
        stdev(q_ind)*t.interval(alpha, len_q_ind-1)[1]/len_q_ind
    )**2  # give NaN for zero degree of freedom
    overall_uncer_sq = zero_order_sq+first_order_sq
    df.details['q_uncer_'+hx_name+'_mean'+time_std_end] = sqrt(
        overall_uncer_sq
    )
    df.details['q_uncer_'+hx_name+'_mean'+time_std_end+'_zero'] = sqrt(
        zero_order_sq
    )
    df.details['q_uncer_'+hx_name+'_mean'+time_std_end+'_first'] = sqrt(
        first_order_sq
    )
    df.details['rel_q_uncer_'+hx_name+'_mean'+time_std_end+'_zero'] = \
        zero_order_sq/overall_uncer_sq
    df.details['rel_q_uncer_'+hx_name+'_mean'+time_std_end+'_first'] = \
        first_order_sq/overall_uncer_sq
    for name in uncer_add_names:
        strend = name.replace(uncer_col_name, '')
        if strend != '':
            temp_uncer = sum(
                [(value/len_q_ind)**2 for value in q_uncer_inds[name]]
            )
            df.details[
                'q_uncer_'+hx_name+'_mean'+time_std_end+strend
            ] = sqrt(temp_uncer)
            df.details[
                'rel_q_uncer_'+hx_name+'_mean'+time_std_end+strend
            ] = temp_uncer/overall_uncer_sq

    # return values
    df_option.set(df)
    return df_option


def cal_q_from_ind_mea(
    df_option, deltat=0, hx_name='hx', alpha=0.95
):
    """
        This function integrates the mdotdeltah column result
        according to the user-defined period per sample. The heat
        transfer rates from each sample are averaged to obtain the
        mean heat transfer rate. It returns df_option with
        the 'details' attribute contain extra information that summarizes
        the data point with heat transfer rate and its
        uncertainty with deltat in its name.

        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file

        deltat: float
            period of measurement per sample in seconds. If you want to
            the mean of instantaneous heat transfer rate, set it to 0.
            Default 10.

        hx_name: string
            name of heat exchanger that is under analyzed. Default 'hx'

        alpha: float
            level of confidence interval you want in the uncertainty
            of the mean. Default 0.95
    """

    # check inputs
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    if deltat < 0:
        raise ValueError(
            "deltat to "+inspect.stack()[0][3]+"() is negative."
        )
    data_col_name = prod_str+"_"+hx_name
    uncer_col_name = data_col_name+uncer_str_end
    if data_col_name not in df.columns.values:
        raise IndexError(
            inspect.stack()[0][3]+"() cannot find column" +
            data_col_name+". Please run cal_mdotdeltah_per_time()" +
            " to generate the column before using "+inspect.stack()[0][3]+"()"
        )
    # need code to check column names

    # identify where the integration is needed
    q_ind = []
    q_uncer_ind = []
    if deltat > 0:  # integration by trapezoidal rule required
        time_str = df.details[time_col_name]
        end_ind = df.shape[0]-1
        beg_ind = 0
        acc = 0.0
        acc_uncer_sq = 0.0
        acc_time = 0.0
        while beg_ind < end_ind:
            # begin integration
            beg_time = df[time_str][df.index[beg_ind]]
            end_time = beg_time+deltat
            acc = 0.5*df[data_col_name][df.index[beg_ind]].get()
            if beg_ind < end_ind:
                deltaT = df[time_str][df.index[beg_ind+1]] -\
                    df[time_str][df.index[beg_ind]]
            else:
                deltaT = df[time_str][df.index[beg_ind]] -\
                    df[time_str][df.index[beg_ind-1]]
            acc = acc*deltaT
            acc_uncer_sq = (
                0.5*df[uncer_col_name][df.index[beg_ind]].get()*deltaT
            )**2
            while beg_ind < end_ind:
                beg_ind = beg_ind+1
                if beg_ind < end_ind and \
                        df[time_str][df.index[beg_ind+1]] <= end_time:
                    deltaT = df[time_str][df.index[beg_ind+1]] -\
                        df[time_str][df.index[beg_ind-1]]
                    acc = acc +\
                        df[data_col_name][df.index[beg_ind]].get()*deltaT/2.
                    acc_uncer_sq = acc_uncer_sq+(
                        df[uncer_col_name][df.index[beg_ind]].get()*deltaT/2.
                    )**2
                else:
                    deltaT = df[time_str][df.index[beg_ind]] -\
                        df[time_str][df.index[beg_ind-1]]
                    acc = acc +\
                        df[data_col_name][df.index[beg_ind]].get()*deltaT/2.
                    acc_uncer_sq = acc_uncer_sq+(
                        df[uncer_col_name][df.index[beg_ind]].get()*deltaT/2.
                    )**2
                    break

            # divide the integrated heat transfer by time
            acc_time = df[time_str][df.index[beg_ind]]-beg_time
            q_ind.append(acc/acc_time)
            q_uncer_ind.append(sqrt(acc_uncer_sq)/acc_time)

    else:  # mean of instantaneous heat transfer rate is needed
        q_ind = [df[data_col_name][ind].get() for ind in df.index]
        q_uncer_ind = [df[uncer_col_name][ind].get() for ind in df.index]

    # taking the mean values for result
    if deltat == 0:
        time_std_end = '_inst'
    else:
        time_std_end = '_'+str(int(deltat))+'s'
    len_q_ind = len(q_ind)
    df.details['q_'+hx_name+'_mean'+time_std_end] = sum(q_ind)/len_q_ind

    # getting the variation of samples as environmental noise effect
    zero_order_sq = sum([(value/len_q_ind)**2 for value in q_uncer_ind])
    first_order_sq = (
        stdev(q_ind)*t.interval(alpha, len_q_ind-1)[1]/len_q_ind
    )**2  # give NaN for zero degree of freedom
    df.details['q_uncer_'+hx_name+'_mean'+time_std_end] = sqrt(
        zero_order_sq+first_order_sq
    )
    df.details['q_uncer_'+hx_name+'_mean'+time_std_end+'_zero'] = sqrt(
        zero_order_sq
    )
    df.details['q_uncer_'+hx_name+'_mean'+time_std_end+'_first'] = sqrt(
        first_order_sq
    )

    # return values
    df_option.set(df)
    return df_option


def print_data(ss_df_options, csv_path, detail_names):
    """
        Print the data in ss_df_options details into a summary
        file.

        ss_df_options: list
            list of misc_func.OptionalVariable() that contains
            a pandas DataFrame with 'details' attribute

        csv_path: string
            path and name of the file with the csv extension.

        detail_names: list
            the key names in the 'details' attribute in the
            pandas DataFrams in ss_df_options that you want
            to print in the file. They should be in the order
            of your output
    """

    # write file
    ofile = open(csv_path, 'w', newline='')
    ofile_writer = csv.writer(ofile, delimiter=',')
    # write header
    ofile_writer.writerow(detail_names)
    # write detailed info
    for ss_df_option in ss_df_options:
        row_infos = [
            ss_df_option.get().details[name]
            for name in detail_names
        ]
        ofile_writer.writerow(row_infos)
    # close file
    ofile.close()

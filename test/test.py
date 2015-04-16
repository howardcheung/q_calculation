#!/usr/bin/python

"""
    This file is a script demonstrating a more thorough analysis
    with multiple files. For detailed discussion, please check
    /demo/Demonstration.ipynb
"""

import sys
sys.path.insert(0, '../src/')  # import modules from another folder

import data_read as dread
import data_summary as ds
import misc_func as misc

# information of differen files
# file paths of different data files
filepaths = [
    '../data/test_data.csv', '../data/test_data_02.csv'
]
# fault type in different files, ordered in the same way as
# list filepaths
fault_types = ['UC', 'NoF']
# fault levels in different files, ordered in the same way as
# list filepaths
fault_levels = ['10', '0']

# information that is the same in all files
alpha = 0.95  # alpha value for confidence intervals
# length of steady state data point and intervals
# and intervals between data points
ss_time = 900
hx_name = 'evap'  # identifier of the heat exchanger analyzed
tei_name = "TEI"  # column name of inlet temperature in degree F
teo_name = "TEO"  # column name of outlet temperature in degree F
# column name of temperature of volumetric airflow in degree F
tvdot_name = "TBO"
vdot_name = "FWE"  # column name of flow rate in GPM
kW_name = "kW"  # column name of power consumption in kW
tset_name = "TWE_set"  # column name of setpoint temperature in degree F
# column names of other important temperature measurement in degree F
other_t_names = ["TSI", "TCI", "TCO"]
# column names of other important flow rates in GPM
other_vdot_names = ["FWC"]
# column name of time measurement in seconds
time_name = "Time"
# string endings for temperature readings after converting to SI values
k_str_end = "_K"
vol_str_end = "_m3s"
# tolerance for steady state operation detection
t_toler = 0.05  # temperature change in K/min
rel_toler = 0.1/100.0  # relative change per minute
# uncertainty values
temp_F_uncer = 0.05  # temperature measurement in Rankine
vdot_rel_uncer = 0.01  # relative volumetric flow rate
kW_rel_uncer = 0.015  # relative power consumption

# create a list to store the data? (potential memory problem)
all_ss_df_options = []

# strings for converted measurement in SI units
temp_mea_si_names = [
    name+k_str_end for name in [
        tei_name, teo_name, tvdot_name
    ]+other_t_names
]
vdot_mea_si_names = [
    name+vol_str_end for name in [vdot_name]+other_vdot_names
]

# check if iteration has the corrected information
if len(filepaths) != len(fault_types):
    raise IndexError(
        "Different number of files to be processed" +
        "with number of fault types defined"
    )
if len(filepaths) != len(fault_levels):
    raise IndexError(
        "Different number of files to be processed" +
        "with number of fault levels defined"
    )
if len(fault_levels) != len(fault_types):
    raise IndexError(
        "Different number of fault levels to be processed" +
        "with number of fault types defined"
    )

# iterate for all files
for filepath, fault_type, fault_level in zip(
    filepaths, fault_types, fault_levels
):
    # read the file
    df_option = dread.read_csv_option(filepath)

    # should add code to remove rows that are bad
    # known bad: water temperature below 32F
    df_option.set(df_option.get()[df_option.get()[tei_name] > 32.0])
    df_option.set(df_option.get()[df_option.get()[teo_name] > 32.0])
    df_option.set(df_option.get()[df_option.get()[tvdot_name] > 32.0])
    # water flow rate below zero should be eliminated
    df_option.set(df_option.get()[df_option.get()[vdot_name] > 0.0])

    # unit conversion
    df_option = dread.unit_convert(df_option, [
        tei_name, teo_name, tvdot_name, tset_name
    ]+other_t_names)
    df_option = dread.unit_convert(
        df_option, [vdot_name]+other_vdot_names, 'GPM', 'm3s'
    )

    # put information not in the files into the pandas DataFrame
    df_option = dread.time_indication(df_option, time_name)
    df_option = dread.fault_allocation(df_option, fault_type, fault_level)

    # calculating slopes for steady state determination
    df_option = dread.cal_for_ss(
        df_option, 'abs_slope', col_names=temp_mea_si_names,
        ss_time=ss_time
    )
    df_option = dread.cal_for_ss(
        df_option, 'rel_slope', col_names=vdot_mea_si_names+[kW_name],
        ss_time=ss_time
    )
    # getting minimum values of water temperature at evaporator only
    df_option = dread.cal_for_ss(
        df_option, 'min', col_names=[
            tei_name+k_str_end, teo_name+k_str_end,
            tvdot_name+k_str_end
        ],
        ss_time=ss_time
    )
    # determining where the steady states are
    # add additional conditions to remove points that
    # are not recorded correctly
    ss_df_options = dread.ss_identifier(
        df_option, temp_mea_si_names+vdot_mea_si_names+[
            kW_name, tset_name+k_str_end
        ],
        abs_slope_cols=temp_mea_si_names,
        rel_slope_cols=vdot_mea_si_names+[kW_name],
        abs_slope_thres=[t_toler for x in range(len(temp_mea_si_names))],
        rel_slope_thres=[
            rel_toler for x in range(len(vdot_mea_si_names+[kW_name]))
        ], deltaT=ss_time, ss_time=ss_time,
        all_conditions=[
            tei_name+k_str_end+'_min > 273.15',
            teo_name+k_str_end+'_min > 273.15',
            tvdot_name+k_str_end+'_min > 273.15'
        ]
    )

    # iterate for all steady state periods
    for ss_df_option in ss_df_options:

        # assign manufacturer uncertainty
        for col_name in temp_mea_si_names:
            ss_df_option = ds.append_uncer_to_df(
                ss_df_option, col_name, misc.r2k(temp_F_uncer), 0.0
            )
        for col_name in vdot_mea_si_names:
            ss_df_option = ds.append_uncer_to_df(
                ss_df_option, col_name, 0.0, vdot_rel_uncer
            )
        ss_df_option = ds.append_uncer_to_df(
            ss_df_option, kW_name, 0.0, kW_rel_uncer
        )

        # getting the means of temperature and
        # mass flow rate readings from the steady
        # state period
        ss_df_option = ds.data_mean_cal(
            ss_df_option,
            temp_mea_si_names+vdot_mea_si_names+[
                kW_name, tset_name+k_str_end
            ],
            alpha=alpha
        )

        # calculate heat transfer rate from the mean
        # observations of the temperature and mass flow rates
        ss_df_option = ds.cal_q_with_uncer_from_sample_result(
            ss_df_option, vdot_name+vol_str_end,
            tvdot_name+k_str_end,
            teo_name+k_str_end,
            tei_name+k_str_end, hx_name=hx_name
        )

        # calculate mdot*deltah at each time instant
        ss_df_option = ds.cal_mdotdeltah_and_uncer_comp_per_time(
            ss_df_option, vdot_name+vol_str_end,
            tvdot_name+k_str_end,
            teo_name+k_str_end,
            tei_name+k_str_end, hx_name=hx_name
        )
        # integrate it at each time intervals
        # and getting their mean for a heat transfer rate
        for small_deltat in range(0, 70, 10):
            ss_df_option = ds.cal_q_and_uncer_comp_from_ind_mea(
                ss_df_option, deltat=small_deltat,
                hx_name=hx_name, alpha=alpha
            )

    # append the results per file
    all_ss_df_options = all_ss_df_options+ss_df_options
    del df_option, ss_df_options  # delete the unnecessary files

# print results
ds.print_data(
    all_ss_df_options, '../result/test_file.csv',
    detail_names=[
        'TEI_K_mean', 'TEO_K_mean',
        'TCI_K_mean', 'TCO_K_mean', 'TWE_set_K_mean',
        'FWE_m3s_mean', 'FWC_m3s_mean',
        'fault_type', 'fault_level',
        'kW_mean', 'q_evap_mean_inst',
        'TEI_K_mean', 'TEO_K_mean',
        'TCI_K_mean_uncer', 'TCO_K_mean_uncer',
        'FWE_m3s_mean_uncer', 'FWC_m3s_mean_uncer',
        'kW_mean_uncer', 'q_uncer_evap_mean_inst',
        'TEI_K_abs_slope', 'TEO_K_abs_slope',
        'TCI_K_abs_slope', 'TCO_K_abs_slope',
        'FWE_m3s_rel_slope', 'FWC_m3s_rel_slope',
        'kW_rel_slope',  'q_uncer_evap_mean_inst',
        'q_uncer_evap_mean_inst_first',
        'q_uncer_evap_mean_inst_heos',
        'q_uncer_evap_mean_inst_rhoeos',
        'q_uncer_evap_mean_inst_tin',
        'q_uncer_evap_mean_inst_tout',
        'q_uncer_evap_mean_inst_tvdot',
        'q_uncer_evap_mean_inst_vdot',
        'q_uncer_evap_mean_inst_zero',
        'q_uncer_evap_mean_obs',
        'q_uncer_evap_mean_obs_heos',
        'q_uncer_evap_mean_obs_rhoeos',
        'q_uncer_evap_mean_obs_tin',
        'q_uncer_evap_mean_obs_tout',
        'q_uncer_evap_mean_obs_tvdot',
        'q_uncer_evap_mean_obs_vdot',
        'ss_time'
    ]
)

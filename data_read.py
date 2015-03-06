#!/usr/bin/python

"""
    This file contains scripts to read and process the data
    so as to output pandas dataframes with each dataframe
    representing a steady state time series
"""

import copy
import inspect
import numpy as np
import os
import pandas as pd

import misc_func

time_col_name = 'time_col'
beg_ind_col_name = 'beg_index'
ss_time_col_name = 'ss_time'


def read_csv_option(filepath):
    """
        Read the csv file to create a dataframe. If the csv
        file does not exist, include it in the error message.
        Return OptionalVariable() with error messages in it.

        Inputs:
        ===========
        filepath: string
            a string showing the path or filename
            from the current working directory

        Outputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains pandas Dataframe obtained by
            reading a raw csv from Comstock data file.
            If the file does not exist, set the error of the
            variable that the file does not exist
    """

    df_option = misc_func.OptionalVariable()
    if os.path.isfile(filepath):
        try:
            # use the new Dataframe defined in misc_func
            df = pd.read_csv(filepath)
            df = misc_func.details_in_dataframe(df)
            df_option.set(df)
        except BaseException:
            df_option.setError(misc_func.IncorrectFileError(
                "read_csv_option() is not " +
                "reading a correct csv at "+filepath
            ))
    else:
        df_option.setError(FileExistsError(
            "read_csv_option() cannot find a file in "+filepath
        ))
    return df_option


def unit_convert(
        df_option, col_names=[], init_unit='F', final_unit='K'
):
    """
        This function converts the temperature
        readings with T in the beginning from degree Fahrenheit
        to Kelvin and put them back to the dataframe

        Inputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file

        col_names: list of strings
            column names with temperature readings that
            you would like to do the conversion

        init_unit: string
            unit of the original values.
            'F' stands for degree Fahrenheit
            'C' stands for degree centigrade. Default 'F'
            'GPM' stands for gallons per minute

        final_unit: string
            unit of the original values. 'K' stands for
            Kelvin and 'C' stands for degree
            centigrade. Default 'K'
            'm3s' stands for m3/s

        Outputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by reading
            a raw csv from Comstock data file with
            temperature in K included
    """

    # define function to do unit conversion for each
    # column
    # need to add error checking for units not for
    # same type of measurement
    def unit_conversion(values):
        final_values = values
        # for temperature
        if init_unit is 'F':
            final_values = misc_func.f2c(final_values)
        if final_unit is 'K':
            final_values = misc_func.c2k(final_values)
        # for volumetric flow
        if init_unit is 'GPM':
            if final_unit is 'm3s':
                final_values = misc_func.gpm2m3s(final_values)
        return final_values

    # check if the dataframe is defined
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )

    # convert for each column
    df = df_option.get()
    for col_name in col_names:
        new_col_name = col_name+"_"+final_unit
        try:
            df[new_col_name] = unit_conversion(np.array(
                df[col_name]
            ))
        except KeyError:
            raise KeyError(
                "No column name " +
                col_name+" in df in temp_unit_convert()"
            )

    # for successful runs, assign the values back
    df_option.set(df)
    return df_option


def time_indication(df_option, time_string):
    """
        This function appends the column name for time
        to the dataframe

        Inputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file

        time_string: string
            a string for column name of the time column

        Outputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by reading
            a raw csv from Comstock data file. It contains
            'time_col' in df_option.get().details that
            which column stores the time indicators
    """

    # check if the dataframe is defined
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()

    # add the attribute 'details' if df does not have it
    if not hasattr(df, 'details'):
        df = misc_func.details_in_dataframe(df)

    # set the variables and return the variables
    df.details[time_col_name] = time_string
    df_option.set(df)
    return df_option


def fault_allocation(df_option, fault_type, fault_level):
    """
        This function appends the type and level of fault
        tested in the dataframe as a new column to the data

        Inputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file

        fault_type: string
            a string for the fault type in the test

        fault_level: float
            a float for a dimensionless fault level
            in the test

        Outputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by reading
            a raw csv from Comstock data file. It contains
            'fault_type' and 'fault_level' in
            df_option.get().details that shows the details
            about the faults
    """

    # check if the dataframe is defined
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()

    # add the attribute 'details' if df does not have it
    if not hasattr(df, 'details'):
        df = misc_func.details_in_dataframe(df)

    # set the variables and return the variables
    df.details['fault_type'] = fault_type
    df.details['fault_level'] = fault_level
    df_option.set(df)
    return df_option


def cal_for_ss(df_option, cal_method='mean', col_names=[], ss_time=900):
    """
        This function calculates steady state values of measurement
        done in a time window specified in ss_time. The
        averages will be stored in the Dataframe on the row
        corresponding to the end time of the window. It also introduces
        a column called 'beg_index' to indicate when the steady
        state starts and 'ss_time' in details to indicate the length of
        potential steady state period.

        Inputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by
            reading a raw csv from Comstock data file.
            df_option.get().details.time_col must be defined
            before use. The time column values should also be in
            seconds.

        cal_method: string
            whether you want
            'mean': averages of values
            'abs_slope': average rate of change of values
            'rel_slope': average rate of change of values relative to
            the mean value within the same period
            Default 'mean'

        col_names: list
            list of strings for names of columns to be averaged.
            If the list is empty, the function will only create a
            new column 'beg_index' that marks the beginning of
            potential steady states. Default empty.

        ss_time: float
            length of desired steady state period in seconds.
            Defaulted to be 900.

        Outputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains pandas dataframe obtained by reading
            a raw csv from Comstock data file. The values calculated
            are misc_func.OptionalVariable().
    """

    # check if the dataframe is defined appropriately
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    if time_col_name not in df.details:
        raise TypeError(
            "DataFrame to "+inspect.stack()[0][3]+"() " +
            "does not have "+time_col_name+" in 'details' attribute. " +
            "Use time_indication() to introduce it."
        )

    # create a column to indicate what the index that marks
    # the beginning time stamp for steady state is
    len_df = len(df.index)
    df[beg_ind_col_name] = [-1 for ind in range(len_df)]
    df.details[ss_time_col_name] = ss_time

    # find the beginning time step for calculation
    # for each time instant in the data series
    time_col = df.details[time_col_name]
    beg_time = df[time_col][0]
    beg_cal_time = ss_time+beg_time
    search_indexes = {}
    search_ind = 0
    for ind in df.index:
        if df[time_col][ind] >= beg_cal_time:
            # find the beginning time step
            current_time = df[time_col][ind] -\
                df[time_col][search_ind]
            while current_time > ss_time:
                search_ind = search_ind+1
                current_time = df[time_col][ind] -\
                    df[time_col][search_ind]
            search_indexes[ind] = search_ind
            df[beg_ind_col_name][ind] = search_ind

    # calculate the values and get them into the DataFrame
    len_df = len(df.index)
    beg_cal_time = ss_time+beg_time
    for col_name in col_names:
        # define calculation function
        new_col_name = col_name+"_"+cal_method
        if cal_method is 'mean':  # mean
            def _cal_func(ind):
                value_option = misc_func.OptionalVariable()
                try:
                    value_option.set(
                        df[col_name][search_indexes[ind]:ind+1].mean()
                    )
                except Exception as e:
                    value_option.setError(e)
                return value_option
        elif cal_method is 'abs_slope':
            def _cal_func(ind):
                try:
                    value_option = misc_func.abs_slope(
                        df[time_col][search_indexes[ind]:ind+1],
                        df[col_name][search_indexes[ind]:ind+1]
                    )
                except Exception as e:
                    value_option = misc_func.OptionalVariable()
                    value_option.setError(e)
                return value_option
        elif cal_method is 'rel_slope':
            def _cal_func(ind):
                try:
                    value_option = misc_func.abs_slope(
                        df[time_col][search_indexes[ind]:ind+1],
                        df[col_name][search_indexes[ind]:ind+1]
                    )
                except Exception as e:
                    value_option = misc_func.OptionalVariable()
                    value_option.setError(e)
                return value_option
        else:
            raise NameError(
                "cal_method input "+cal_method+" to " +
                inspect.stack()[0][3]+"() is invalid"
            )

        df[new_col_name] = [
            misc_func.OptionalVariable() for ind in range(len_df)
        ]

        for ind in search_indexes.keys():
            df[new_col_name][ind] = _cal_func(ind)

    # return new df
    df_option.set(df)
    return df_option


def ss_identifier(
    df_option, col_names, abs_slope_cols, rel_slope_cols,
    abs_slope_thres, rel_slope_thres, deltaT=900, ss_time=900
):
    """
        This function identifies all steady state operation
        within a data series and return multiple pandas
        Dataframe in a list. These outputs also carries
        the 'details' attribute from the original Dataframe

        Inputs:
        ===========
        df_option: misc_func.OptionalVariable()
            contains the original data series in pandas Dataframe.

        col_names: list
            list of strings containing the column names that
            you want to pass to the new dataframes

        abs_slope_cols: list
            list of strings for names of columns where the
            average rate of change is calculated

        rel_slope_cols: list
            list of strings for names of columns where the
            relative average rate of change is calculated

        abs_slope_thres: list
            values of threshold which the magnitude if the average
            rate of change should be smaller than to accept the
            operation as steady state. Should be in the same order
            as abs_slope_cols

        rel_slope_thres: list
            values of threshold which the magnitude if the relative average
            rate of change should be smaller than to accept the
            operation as steady state. Should be in the same order
            as rel_slope_cols

        deltaT: float
            minimum time in seconds between two steady state operation.
            Defaulted at 900

        ss_time: float
            length of desired steady state period in seconds.
            Defaulted to be 900.

        Outputs:
        ===========
        df_options: list
            list of misc_func.OptionalVariable() that
            contains pandas dataframe obtained by reading
            a raw csv from Comstock data file.
    """

    # check if the dataframe is defined appropriately
    df_option.check_type(
        pd.core.frame.DataFrame, inspect.stack()[0][3]
    )
    df = df_option.get()
    if time_col_name not in df.details:
        raise TypeError(
            "DataFrame to "+inspect.stack()[0][3]+"() " +
            "does not have "+time_col_name+" in 'details' attribute. " +
            "Use time_indication() to introduce it."
        )
    for col_name in col_names:
        if col_name not in df.columns.values:
            raise NameError(
                "DataFrame to "+inspect.stack()[0][3]+"() " +
                "does not have the column '"+col_name+"'."
            )
    if len(abs_slope_cols) != len(abs_slope_thres):
        raise AttributeError(
            "abs_slope_cols and abs_slope_thres in " +
            "ss_identifier has no equal length."
        )
    if len(rel_slope_cols) != len(rel_slope_thres):
        raise AttributeError(
            "rel_slope_cols and rel_slope_thres in " +
            "ss_identifier has no equal length."
        )
    # run cal_for_ss() to get the steady state indexes
    # if it is not available
    if beg_ind_col_name not in df:
        df_option = cal_for_ss(
            df_option, cal_method='', col_names=[], ss_time=ss_time
        )
        df = df_option.get()

    # define function to compare thres with column values
    # return a boolean if the value is within range
    def _compare(name, thres, ind):
        val = df[name][ind].get()
        if abs(val) < thres:
            return True
        else:
            return False

    # check each row
    abs_slope_cols = [name+'_abs_slope' for name in abs_slope_cols]
    rel_slope_cols = [name+'_rel_slope' for name in rel_slope_cols]
    col_bool_names = abs_slope_cols+rel_slope_cols
    ind = df.index[0]
    df_options = []
    while ind <= df.index[-1]:
        # check if the values are defined
        check_bool = [df[name][ind] for name in col_bool_names]
        if all(check_bool):
            slope_bools = [
                _compare(name, thres, ind)
                for (name, thres) in zip(abs_slope_cols, abs_slope_thres)
            ]+[
                _compare(name, thres, ind)
                for (name, thres) in zip(rel_slope_cols, rel_slope_thres)
            ]
            # check if the values are within range
            if all(slope_bools):
                # create a new DaraFrame for the steady state data
                new_df_option = misc_func.OptionalVariable()
                try:
                    df_small = pd.DataFrame()
                    for col_name in col_names+[df.details[time_col_name]]:
                        df_small[col_name] = \
                            df[col_name][df[beg_ind_col_name][ind]:ind+1]
                    df_small = misc_func.details_in_dataframe(df_small)
                    df_small.details = copy.deepcopy(df.details)
                    # pass new information to indicate the change of values
                    for ext_col in col_bool_names:
                        df_small.details[ext_col] = df[ext_col][ind].get()
                    new_df_option.set(df_small)
                except Exception as e:
                    # if fails, pass the error instead
                    new_df_option.setError(e)
                df_options.append(new_df_option)
                # skip time by deltaT
                next_timestamp = df[df.details[time_col_name]][ind]+deltaT
                ind = ind+1
                while ind <= df.index[-1] and \
                        df[df.details[time_col_name]][ind] < next_timestamp:
                    ind = ind+1
                ind = ind-1 # shift back for the increment at the end
        ind = ind+1

    # return the list of df_option
    return df_options

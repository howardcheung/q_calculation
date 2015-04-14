#!/usr/bin/python

"""
    This file contains misc. scripts including the calculation of
    absolute and relative slope of the change
"""

from math import sqrt
import numpy as np
import pandas as pd

# global variable for unit conversion
k2c_convert = 273.15
gpm2m3s_convert = 6.30901964*1.e-5


class OptionalVariable:
    """
        This function stores a boolean and a variable. It returns true
        if it contains a valid value, and returns false if it doesn't.
        If you want to change the value, you need to use the function
        get and set. If you want to reinitialize the OptionalVariable,
        use the function clear. If you want to see or set the
        Exception e, use function getError or setError.

        _exist: boolean
            indicates if the variable contains a valid object in var

        _var: anything
            if exist is true, it should have a valid variable, no
            default if exist is false. No default.

        _e: Exception
            Exception object recording the cause of no value in var.
            Defaulted AttributeError("No assignment after initialization")
    """

    def __init__(self):
        self._exist = False
        self._e = AttributeError("No var assignment after initialization")

    def __bool__(self):
        """
            Check if the var is defined
        """
        return self._exist

    def set(self, var):
        """
            Initialize and set variables var and type

            var: anything
                variable to be set to var
        """

        self._exist = True
        self._var = var
        self._e = BaseException("Var assigned.")

    def get(self):
        """
            Returns var
        """

        if self:
            return self._var
        else:
            raise self._e

    def clear(self):
        """
            Clear var
        """

        if self:
            del self._var
            self._e = AttributeError("Var cleared")
        self._exist = False

    def getError(self):
        """
            Check error value
        """

        return self._e

    def setError(self, ee):
        """
            Set error value for unsuccessful assignment

            Inputs:
            ===========
            ee: Exception
                Exception to be set to var
        """

        if issubclass(type(ee), BaseException):
            self._e = ee
        else:
            raise AttributeError(
                "ee to setError does not belong to BaseException"
            )

    def check(self):
        """
            Raise the previous error if var is not defined.
        """

        if not self:
            raise self.getError()

    def check_type(self, type_expect, func_name=""):
        """
            Raise an error if var is a class of the type
            expected. Also use the function check to check if
            var is defined.

            Inputs:
            ===========
            type_expect: type
                type of variable expected in self._var

            func_name: string
                name of function. Default ""
        """

        self.check()
        if not issubclass(type(self.get()), type_expect):
            if func_name:
                raise TypeError(
                    "Not "+type_expect.__name__+" in "+func_name+"()"
                )
            else:
                raise TypeError("Not "+type_expect.__name__)


class IncorrectFileError(Exception):
    """
        This error class occurred when the file to
        be read does not match the expected type.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def details_in_dataframe(df):
    """
        This class inherits the pandas DataFrame
        structure with new attributes related to the
        testing conditions not listed in the DataFrame.
        Returns a pandas Dataframe object with attribute
        'details'.

        df: pandas Dataframe
            pandas Dataframe with test data
    """

    df.__setattr__('details', {})
    return df


def abs_slope(time_series, data_series):
    """
        This function calculates the average rate of change
        of the data in the variable data_series per minute.
        Returns a misc_func.OptionalVariable object with
        the absolute slope stored in it.

        time_series: list or numpy array
            a list of timestamps showing when the data
            in data_series are collected in seconds

        data_series:  list or numpy array
            a list of data collected
    """

    slope = OptionalVariable()

    # check if both have the same length. If they don't,
    # return the value
    if len(time_series) != len(data_series):
        slope.setError(IndexError(
            "Time series and data series in abs_slope" +
            "has no equal length."
        ))
        return slope

    # calculate slope by linear regression
    cov_mat_td = np.cov(time_series, data_series)
    var_tt = cov_mat_td[0][0]  # variance
    var_td = cov_mat_td[0][1]  # covariance
    slope.set(var_td/var_tt*60.0)
    return slope


def rel_slope(time_series, data_series):
    """
        This function calculates the average rate of change
        of the data in the variable data_series per minute.
        Returns a misc_func.OptionalVariable object with
        the relative slope stored in it.

        time_series: list or numpy array
            a list of timestamps showing when the data
            in data_series are collected in seconds

        data_series:  list or numpy array
            a list of data collected
    """

    rel_slope = OptionalVariable()
    slope = abs_slope(time_series, data_series)
    if not slope:
        # slope is not defined. Return OptionalVariable
        # as False
        slope.setError(IndexError(
            "Time series and data series in rel_slope" +
            "has no equal length."
        ))
        return rel_slope
    mean_dd = np.mean(data_series)
    rel_slope.set(slope.get()/mean_dd)
    return rel_slope


def sum_of_squares(values):
    """
        This function calculates the sum of squares of multiple
        entries in values. Returns an misc_func.OptionalVariable
        object with a sum of square value in it.

        values: list or numpy array
            contains the floats or OptionaVariable()s
            for the sum of squares
    """

    ssq = OptionalVariable()
    try:
        sq = [(getOptionalVariable(value))**2 for value in values]
        ssq.set(sum(sq))
    except Exception as e:
        ssq.setError(e)
    return ssq


def sqrt_sum_of_squares(values):
    """
        This function calculates the square root of the sum of squares
        from multiple entries in values. Returns a misc_func.OptionalVariable
        object with a square root of the sum of square value in it.

        values: list or numpy array
            contains the values in the sum of squares

    """

    ssq = OptionalVariable()
    try:
        sq = [(getOptionalVariable(value))**2 for value in values]
        ssq.set(sqrt(sum(sq)))
    except Exception as e:
        ssq.setError(e)
    return ssq


def getOptionalVariable(value):
    """
        This function is used to get a value without
        when the value can be an ordinary float or
        OptionalVariable(). If it is OptionalVariable()
        and is undefined, it will output an error instead.
    """
    if issubclass(type(value), OptionalVariable):
        if value:
            return value.get()
        else:
            raise value.getError()
    else:
        return value


def r2k(temp_R):
    """
        This function converts temperature from degree Rankine to Kelvin.
        Returns the temperature in Kelvin.

        temp_F: float or numpy array
            temperature or temperature numpy array in
            degree Rankine
    """

    return temp_R*5.0/9.0


def new_optionalvariable_vector(nlen):
    """
        This function returns a list of misc.OptionalVariable
        for initialization

        nlen: int
            length of list to be generated
    """
    return [OptionalVariable() for x in range(nlen)]


def f2c(temp_F):
    """
        This function converts temperature from degree Fahrenheit
        to degree centigrade. Returns the temperature in degree
        centigrade.

        temp_F: float or numpy array
            temperature or temperature numpy array in
            degree Fahrenheit
    """

    return r2k(temp_F-32.0)


def c2k(temp_C):
    """
        This function converts temperature from degree centigrade
        to Kelvin. Returns the temperature in Kelvin.

        temp_C: float or numpy array
            temperature or temperature numpy array in
            degree centigrade
    """

    return temp_C+k2c_convert


def k2c(temp_K):
    """
        This function converts temperature from degree centigrade
        to Kelvin. Returns the temperature in degree centigrade.

        temp_K: float or numpy array
            temperature or temperature numpy array in
            Kelvin
    """

    return temp_K-k2c_convert


def gpm2m3s(gpm):
    """
        This function converts volumetric flow from
        gpm to m3/s. Returns the flow rate in m3/s

        gpm: float or numpy array
            flow rate in gpm
    """

    return gpm*gpm2m3s_convert

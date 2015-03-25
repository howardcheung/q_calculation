.. Data_analysis documentation master file, created by
   sphinx-quickstart on Tue Mar 17 13:07:53 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to documentation for *q_calculation*
=========================================

These pages explain the mathematics behind the project *q_calculation* and describe its application.

Why *q_calculation*?
=========================================

After each experiment, there are lots of post-processing to examine if the data file is obtained appropriately. This project aims at shorten the time to calculate the steady-state heat transfer rate from experiments on heat exchangers by doing the following tasks automatically:

* Filter time-series data from experiments to multiple steady-state time series
* Sample the steady-state time series data to estimate the heat transfer rate during the heat exchanger operation
* Calculate the uncertainty of the observations and estimated heat transfer rate based on sensor specification, statistical sampling procedure and equation of states of the flowing medium

Software requirements
=========================================

The modules developed in this project depend on 

* `Python 3.4.x or above <https://www.python.org/>`_
* Numpy 
* Scipy 
* Pandas 
* CoolProp 

Contacts
=========================================
For questions, please email Howard Cheung at `howard.at@gmail.com <howard.at@gmail.com>`_ .

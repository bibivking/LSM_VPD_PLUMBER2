#!/usr/bin/env python

"""
UNFINISH
"""

import os
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from quality_control import *
# from check_vars import check_variable_exists

import pandas as pd
import numpy as np

def spi(data, period):
  """
  Calculates the Standardized Precipitation Index (SPI) for a given time period.

  Args:
    data: A Pandas DataFrame of daily precipitation data.
    period: The time period for which to calculate the SPI.

  Returns:
    A Pandas DataFrame of the SPI values.
  """

  # Calculate the long-term average precipitation.
  mean = data['precipitation'].mean()

  # Calculate the standard deviation of the precipitation.
  std = data['precipitation'].std()

  # Calculate the SPI values.
  spi = (data['precipitation'] - mean) / std

  # Return the SPI values as a Pandas DataFrame.
  return pd.DataFrame(spi, columns=['SPI_' + str(period)])


if __name__ == '__main__':
  # Load the daily precipitation data.
  data = pd.read_csv('daily_precipitation.csv')

  # Calculate the SPI for a 12-month period.
  spi_12 = spi(data, 12)

  # Print the SPI values.
  print(spi_12)
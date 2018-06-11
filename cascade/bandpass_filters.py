"""Implementations of bandpass filters for separating different spatial scales 
from two-dimensional images in the frequency domain.

The methods in this module implement the following interface:

  filter_xxx(L, n, optional arguments)

where L is the width and height of the input field, respectively, and n is the 
number of frequency bands to use.

The output of each filter function is a dictionary containing the following 
key-value pairs:
  filters_1d       2d array of shape (n, L/2) containing 1d filter weights 
                   for each frequency band k=1,2,...,n
  filters_2d       3d array of shape (n, L, L) containing the 2d filter weights 
                   for each frequency band k=1,2,...,n
  central_freqs    1d array of central frequencies of the filters
"""

# TODO: Should the filter always return an 1d array and should we use a separate 
# method for generating the 2d filter from the 1d filter?

def filter_gaussian(L, n, l_0=3, gauss_scale=0.2, gauss_scale_0=0.3):
  """Gaussian band-pass filter in logarithmic frequency scale. The method is 
  described in
  
  S. Pulkkinen, V. Chandrasekar and A.-M. Harri, Nowcasting of Precipitation in 
  the High-Resolution Dallas-Fort Worth (DFW) Urban Radar Remote Sensing 
  Network, IEEE Journal of Selected Topics in Applied Earth Observations and 
  Remote Sensing, 2018, to appear.
  
  Parameters
  ----------
  L : int
    The width and height of the input field.
  n : int
    The number of frequency bands to use.
  l_0 : int
    Central frequency of the second band (the first band is always centered at 
    zero).
  gauss_scale : float
    Optional scaling prameter. Proportional to the standard deviation of the 
    Gaussian weight functions.
  gauss_scale_0 : float
    Optional scaling parameter for the Gaussian function corresponding to the 
    first frequency band.
  """
  pass

def _gaussweights_1d():
  pass

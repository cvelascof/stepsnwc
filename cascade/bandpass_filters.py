
import numpy as np

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
  central_freqs    1d array of shape n containing the central frequencies of the 
                   filters

The filter weights are assumed to be normalized so that for any Fourier 
wavenumber they sum to one.
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
  X,Y = np.ogrid[-L/2+1:L/2+1, -L/2+1:L/2+1]
  R = np.sqrt(X*X + Y*Y)
  
  wfs = _gaussweights_1d(L, n, l_0=l_0, gauss_scale=gauss_scale, 
                         gauss_scale_0=gauss_scale_0)
  
  W = np.empty((n, L, L))
  
  for i,wf in enumerate(wfs):
    W[i, :, :] = wf(R)
  
  W_sum = np.sum(W, axis=0)
  for k in xrange(W.shape[0]):
    W[k, :, :] /= W_sum
  
  return W

def _gaussweights_1d(l, n, l_0=3, gauss_scale=0.2, gauss_scale_0=0.3):
  e = pow(0.5*l/l_0, 1.0/(n-2))
  r = [(l_0*pow(e, k-1), l_0*pow(e, k)) for k in xrange(1, n-1)]
  
  f = lambda x,s: np.exp(-x**2.0 / (2.0*s**2.0))
  def log_e(x):
    if len(np.shape(x)) > 0:
      res = np.empty(x.shape)
      res[x == 0] = 0.0
      res[x > 0] = np.log(x[x > 0]) / np.log(e)
    else:
      if x == 0.0:
        res = 0.0
      else:
        res = np.log(x) / np.log(e)
    
    return res
  
  class gaussfunc:
    
    def __init__(self, c, s):
      self.c = c
      self.s = s
    
    def __call__(self, x):
      return f(log_e(x) - self.c, self.s)
  
  weight_funcs = []
  
  s = gauss_scale * e
  weight_funcs.append(gaussfunc(0.0, gauss_scale_0 * e))
  
  for i,ri in enumerate(r):
    rc = log_e(ri[0])
    weight_funcs.append(gaussfunc(rc, s))
  
  gf = gaussfunc(log_e(l/2), s)
  def g(x):
    res = np.ones(x.shape)
    mask = x <= l/2
    res[mask] = gf(x[mask])
    
    return res
  
  weight_funcs.append(g)
  
  return weight_funcs

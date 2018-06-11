"""Implementations of cascade decompositions for separating two-dimensional 
images into multiple spatial scales.

The methods in this module implement the following interface:

  decomposition_xxx(X, filter, optional arguments)

where X is the input field and filter is a dictionary returned by a filter 
method implemented in bandpass_filters.py. X is required to have a square shape. 
The output of each method is a dictionary with the following key-value pairs:

  cascade_levels    three-dimensional array of shape (n,L,L), where n is 
                    the number of cascade levels and L is the size of the input 
                    field
  means             list of mean values for each cascade level
  stds              list of standard deviations for each cascade level
"""

import numpy as np
# Use the SciPy fft by default. If SciPy is not installed, fall back to the 
# numpy implementation.
try:
    import scipy.fftpack as fft
except ImportError:
    from numpy import fft

def decomposition_fft(X, filter, conditional=False, cond_thr=None):
    """Decompose a 2d input field into multiple spatial scales by using the Fast 
    Fourier Transform (FFT) and a bandpass filter.
    
    Parameters
    ----------
    X : array_like
      Two-dimensional array containing the input field. The width and height of 
      the field must be equal.
    filter : dict
      A filter returned by any method implemented in bandpass_filters.py.
    conditional : bool
      If set to True, compute the statistics of each cascade level conditionally 
      excluding the areas where the values are below the given threshold. This 
      requires cond_thr to be set.
    cond_thr : float
      Threshold value for conditional computation of cascade statistics, see 
      above.
    
    Returns
    -------
    out : ndarray
      Three-dimensional array of shape (n,L,L) containing the field decomposed 
      into n spatial scales. The parameter n is determined upon initialization 
      of the filter (see bandpass_filters.py), and L is the size of the input 
      field.
    """
    if len(X.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    if X.shape[0] != X.shape[1]:
        raise ValueError("the dimensions of the input field are %dx%d, but square shape expected" % \
                         (X.shape[0], X.shape[1]))
    if conditional and cond_thr is None:
      raise Exception("conditional=True, but cond_thr was not supplied")
    
    result = {}
    means  = []
    stds   = []
    
    if conditional:
      MASK = X >= cond_thr
    
    F = fft.fftshift(fft.fft2(X))
    X_decomp = []
    for k in xrange(len(filter["weights_1d"])):
        W_k = filter["weights_2d"][k, :, :]
        X_ = np.real(fft.ifft2(fft.ifftshift(F*W_k)))
        X_decomp.append(X_)
        
        if conditional:
          X_ = X_[MASK]
        means.append(np.mean(X_))
        stds.append(np.std(X_))
    
    result["cascade_levels"] = np.stack(X_decomp)
    result["means"] = means
    result["stds"]  = stds
    
    return result

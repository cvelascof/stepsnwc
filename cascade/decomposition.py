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

def get_method(name):
    """Return a callable function for the decomposition method corresponding to 
    the given name. The available options are:\n\
    
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  fft              | decomposition based on Fast Fourier Transform (FFT)    |
    |                   | and a bandpass filter                                  |
    +-------------------+--------------------------------------------------------+
    """
    if name == "fft":
        return decomposition_fft
    else:
        raise ValueError("unknown method %s, the only currently implemented method is 'fft'" % name)

def decomposition_fft(X, filter, MASK=None):
    """Decompose a 2d input field into multiple spatial scales by using the Fast 
    Fourier Transform (FFT) and a bandpass filter.
    
    Parameters
    ----------
    X : array_like
      Two-dimensional array containing the input field. The width and height of 
      X must be equal, and all values are required to be finite.
    filter : dict
      A filter returned by any method implemented in bandpass_filters.py.
    MASK : array_like
      Optional mask to use for computing the statistics for the cascade levels. 
      Pixels with MASK==False are excluded from the computations.
    
    Returns
    -------
    out : ndarray
      A dictionary described in the module documentation. The parameter n is 
      determined from the filter (see bandpass_filters.py).
    """
    if len(X.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    if X.shape[0] != X.shape[1]:
        raise ValueError("the dimensions of the input field are %dx%d, but square shape expected" % \
                         (X.shape[0], X.shape[1]))
    if MASK is not None and MASK.shape != X.shape:
      raise ValueError("dimension mismatch between X and MASK: X.shape=%s, MASK.shape=%s" % \
        (str(X.shape), str(MASK.shape)))
    if np.any(~np.isfinite(X)):
      raise ValueError("X contains non-finite values")
    
    result = {}
    means  = []
    stds   = []
    
    F = fft.fftshift(fft.fft2(X))
    X_decomp = []
    for k in xrange(len(filter["weights_1d"])):
        W_k = filter["weights_2d"][k, :, :]
        X_ = np.real(fft.ifft2(fft.ifftshift(F*W_k)))
        X_decomp.append(X_)
        
        if MASK is not None:
            X_ = X_[MASK]
        means.append(np.mean(X_))
        stds.append(np.std(X_))
    
    result["cascade_levels"] = np.stack(X_decomp)
    result["means"] = means
    result["stds"]  = stds
    
    return result

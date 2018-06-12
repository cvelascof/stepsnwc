"""Implementations of nowcasting methods."""

import numpy as np
from timeseries import autoregression
from cascade import bandpass_filters, decomposition
from timeseries import correlation
from motion import advection

def simple_advection(R, V, num_timesteps, extrap_method, extrap_args={}):
    """Generate a nowcast by applying a simple advection-based extrapolation to 
    the given precipitation field.
    
    Parameters
    ----------
    R : array-like
      Two-dimensional array of shape (m,n) containing the input precipitation 
      field.
    V : array-like
      Array of shape (2,m,n) containing the x- and y-components of the advection 
      field. The velocities are assumed to represent one time step.
    num_timesteps : int
      Number of time steps to forecast.
    extrap_method : str
      Name of the extrapolation method to use. See the documentation of the 
      advection module for the available choices.
    extrap_args : dict
      Optional dictionary that is supplied as keyword arguments to the 
      extrapolation method.
    
    Returns
    -------
    out : ndarray
      Three-dimensional array of shape (num_timesteps,m,n) containing a time 
      series of nowcast precipitation fields.
    """
    extrap_method = advection.get_method(extrap_method)
    
    return extrap_method(R[-1, :, :], V, num_timesteps)

def s_prog(R, V, num_timesteps, extrap_method, num_cascade_levels, 
    conditional=True, cond_thr=None, 
    bandpass_filter_method=bandpass_filters.filter_gaussian, 
    decomp_method=decomposition.decomposition_fft, 
    extrap_kwargs=None, filter_kwargs=None, decomposition_kwargs=None):
    """Generate a nowcast by using the S-PROG method (Seed, 2003).
    
    Parameters
    ----------
    R : array-like
      Three-dimensional array of shape (3,m,n) containing three input 
      precipitation fields ordered by timestamp from oldest to newest. The time 
      steps between the inputs are assumed to be regular.
    V : array-like
      Array of shape (2,m,n) containing the x- and y-components of the advection 
      field. The velocities are assumed to represent one time step between the 
      inputs.
    num_timesteps : int
      Number of time steps to forecast.
    extrap_method : str
      Name of the extrapolation method to use. See the documentation of the 
      advection module for the available choices.
    extrap_args : dict
      Optional dictionary that is supplied as keyword arguments to the 
      extrapolation method.
    conditional : bool
      If set to True, compute the correlation coefficients conditionally by 
      excluding the areas where the values are below the given threshold. This 
      requires cond_thr to be set.
    cond_thr : float
      Threshold value for conditional computation of correlation coefficients, 
      see above.
    
    Returns
    -------
    out : ndarray
      Three-dimensional array of shape (num_timesteps,m,n) containing a time 
      series of nowcast precipitation fields.
    """
    if np.any(~np.isfinite(R)):
        raise ValueError("R contains non-finite values")
    
    L = R.shape[1]
    extrap_method = advection.get_method(extrap_method)
    R = R.copy()
    
    # Advect the previous precipitation fields to the same position with the 
    # most recent one (i.e. transform them into the Lagrangian coordinates).
    R[0, :, :] = extrap_method(R[0, :, :], V, 2, outval="min")[1]
    R[1, :, :] = extrap_method(R[1, :, :], V, 1, outval="min")[0]
    
    if conditional:
        MASK = np.logical_and.reduce([R[i, :, :] >= cond_thr for i in xrange(R.shape[0])])
    else:
        MASK = None
    
    # Initialize the band-pass filter.
    filter = bandpass_filters.filter_gaussian(L, num_cascade_levels)
    
    # Compute the cascade decompositions of the input precipitation fields.
    R_d = []
    for i in xrange(3):
        R_ = decomposition.decomposition_fft(R[i, :, :], filter, MASK=MASK)
        R_d.append(R_)
    
    # Normalize the cascades and rearrange them into a four-dimensional array of 
    # shape (num_cascade_levels,3,L,L) for the AR(2) model.
    R_c = []
    for i in xrange(num_cascade_levels):
        R_ = []
        for j in xrange(3):
            mu    = R_d[j]["means"][i]
            sigma = R_d[j]["stds"][i]
            R__ = (R_d[j]["cascade_levels"][i, :, :] - mu) / sigma
            R_.append(R__)
        R_c.append(np.stack(R_))
    R_c = np.stack(R_c)
    
    # Compute lag-1 and lag-2 temporal autocorrelation coefficients for each 
    # cascade level.
    GAMMA = np.empty((num_cascade_levels, 2))
    for i in xrange(num_cascade_levels):
        R_c_ = np.stack([R_c[i, j, :, :] for j in xrange(3)])
        GAMMA[i, :] = correlation.temporal_autocorrelation(R_c_, MASK=MASK)
    
    print(GAMMA)
    
    # Adjust the correlation coefficients to ensure that the AR(2) process 
    # is stationary.
    for i in xrange(num_cascade_levels):
      GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef(GAMMA[i, 0], GAMMA[i, 1])
    
    # Estimate the parameters of the AR(2) model from the autocorrelation 
    # coefficients.
    PHI = np.empty((num_cascade_levels, 2))
    for i in xrange(num_cascade_levels):
        PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])
    
    print(PHI)
    
    # Discard the first of the three cascades because it is not needed for the 
    # AR(2) model.
    R_c = R_c[:, 1:, :, :]
    
    # TODO: Implement the extrapolation.
    R_f = []
    for k in xrange(num_timesteps):
        # Iterate the AR(2) model for each cascade level.
        for i in xrange(num_cascade_levels):
            R_c[i, :, :, :] = autoregression.iterate_ar_model(R_c[i, :, :, :], 
                                                              PHI[i, :])
    
    return R_f

def steps(R, V, num_timesteps, extrap_method, num_ens_members, 
          num_cascade_levels, extrap_args={}, ar_order=2):
    """Generate a nowcast ensemble by using the STEPS method (Bowler et al., 2006).
    
    Parameters
    ----------
    R : array-like
      Three-dimensional array of shape (3,m,n) containing three input 
      precipitation fields ordered by timestamp from oldest to newest. The time 
      steps between the inputs are assumed to be regular.
    V : array-like
      Array of shape (2,m,n) containing the x- and y-components of the advection 
      field. The velocities are assumed to represent one time step between the 
      inputs.
    num_timesteps : int
      Number of time steps to forecast.
    extrap_method : str
      Name of the extrapolation method to use. See the documentation of the 
      advection module for the available choices.
    num_ens_members : int
      Number of ensemble members to generate.
    num_cascade_levels : int
      Number of cascade levels to use.
    extrap_args : dict
      Optional dictionary that is supplied as keyword arguments to the 
      extrapolation method.
    ar_order : int
      Optional parameter for the order of the autoregressive model to use.
    
    Returns
    -------
    out : ndarray
      Four-dimensional array of shape (num_ens_members,num_timesteps,m,n) 
      containing a time series of forecast precipitation fields for each ensemble 
      member.
    """
    pass

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
    _check_inputs(R, V, 1)
    
    extrap_method = advection.get_method(extrap_method)
    
    return extrap_method(R, V, num_timesteps)

def s_prog(R, V, num_timesteps, num_cascade_levels, R_thr, extrap_method, 
           decomp_method, bandpass_filter_method, conditional=True, 
           extrap_kwargs={}, filter_kwargs={}):
    """Generate a nowcast by using the S-PROG method (Seed, 2003).
    
    Parameters
    ----------
    R : array-like
      Three-dimensional array of shape (3,m,n) containing three input 
      precipitation fields ordered by timestamp from oldest to newest. The time 
      steps between the inputs are assumed to be regular, and the inputs are 
      required to have finite values.
    V : array-like
      Array of shape (2,m,n) containing the x- and y-components of the advection 
      field. The velocities are assumed to represent one time step between the 
      inputs.
    num_timesteps : int
      Number of time steps to forecast.
    num_cascade_levels : int
      The number of cascade levels to use.
    R_thr : float
      Specifies the threshold value to use if conditional is True.
    extrap_method : str
      Name of the extrapolation method to use. See the documentation of the 
      advection module for the available choices.
    decomp_method : str
      Name of the cascade decomposition method to use, see the documentation 
      of the cascade.decomposition module.
    bandpass_filter_method : str
      Name of the bandpass filter method to use with the cascade decomposition, 
      see the documentation of the cascade.bandpass_filters module.
    conditional : bool
      If set to True, compute the correlation coefficients conditionally by 
      excluding the areas where the values are below the given threshold. This 
      requires R_thr to be set.
    extrap_kwargs : dict
      Optional dictionary that is supplied as keyword arguments to the 
      extrapolation method.
    filter_kwargs : dict
      Optional dictionary that is supplied as keyword arguments to the 
      filter method.
    
    Returns
    -------
    out : ndarray
      Three-dimensional array of shape (num_timesteps,m,n) containing a time 
      series of nowcast precipitation fields.
    """
    _check_inputs(R, V, 2)
    
    if np.any(~np.isfinite(R)):
        raise ValueError("R contains non-finite values")
    
    L = R.shape[1]
    extrap_method = advection.get_method(extrap_method)
    R = R.copy()
    
    # Advect the previous precipitation fields to the same position with the 
    # most recent one (i.e. transform them into the Lagrangian coordinates).
    R[0, :, :] = extrap_method(R[0, :, :], V, 2, outval="min", **extrap_kwargs)[1]
    R[1, :, :] = extrap_method(R[1, :, :], V, 1, outval="min", **extrap_kwargs)[0]
    
    if conditional:
        MASK = np.logical_and.reduce([R[i, :, :] >= R_thr for i in xrange(R.shape[0])])
    else:
        MASK = None
    
    # Initialize the band-pass filter.
    filter_method = bandpass_filters.get_method(bandpass_filter_method)
    filter = filter_method(L, num_cascade_levels, **filter_kwargs)
    
    # Compute the cascade decompositions of the input precipitation fields.
    decomp_method = decomposition.get_method(decomp_method)
    R_d = []
    for i in xrange(3):
        R_ = decomp_method(R[i, :, :], filter, MASK=MASK)
        R_d.append(R_)
    
    # Normalize the cascades and rearrange them into a four-dimensional array 
    # of shape (num_cascade_levels,3,L,L) for the AR(2) model.
    R_c,mu,sigma = _stack_cascades(R_d, num_cascade_levels)
    
    # Compute lag-1 and lag-2 temporal autocorrelation coefficients for each 
    # cascade level.
    GAMMA = np.empty((num_cascade_levels, 2))
    for i in xrange(num_cascade_levels):
        R_c_ = np.stack([R_c[i, j, :, :] for j in xrange(3)])
        GAMMA[i, :] = correlation.temporal_autocorrelation(R_c_, MASK=MASK)
    
    _print_corrcoefs(GAMMA)
    
    # Adjust the lag-2 correlation coefficient to ensure that the AR(2) process 
    # is stationary.
    for i in xrange(num_cascade_levels):
        GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef(GAMMA[i, 0], GAMMA[i, 1])
    
    # Estimate the parameters of the AR(2) model from the autocorrelation 
    # coefficients.
    PHI = np.empty((num_cascade_levels, 2))
    for i in xrange(num_cascade_levels):
        PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])
    
    _print_ar2_params(PHI, False)
    
    # Discard the first of the three cascades because it is not needed for the 
    # AR(2) model.
    R_c = R_c[:, 1:, :, :]
    
    D = None
    R_f = []
    
    #MASK_p = (R[-1, :, :] >= R_thr).astype(float) # precipitation mask
    #R_min = np.min(R)
    
    for t in xrange(num_timesteps):
        # Iterate the AR(2) model for each cascade level.
        for i in xrange(num_cascade_levels):
            R_c[i, :, :, :] = autoregression.iterate_ar_model(R_c[i, :, :, :], 
                                                              PHI[i, :])
        
        # Compute the recomposed precipitation field from the cascade obtained 
        # from the AR(2) model.
        R_r = [(R_c[i, -1, :, :] * sigma[i]) + mu[i] for i in xrange(num_cascade_levels)]
        R_r = np.sum(np.stack(R_r), axis=0)
        
        # Advect the recomposed precipitation field to obtain the forecast for 
        # time step t.
        R_f_,D = advection.semilagrangian(R_r, V, 1, D_prev=D, return_displacement=True)
        R_f_ = R_f_[0]
        
        # Advect the precipitation mask and apply it to the output.
        # TODO: Make the precipitation mask optional.
        #MASK_p_ = advection.semilagrangian(MASK_p, V, 1, D_prev=D)
        #R_f_[MASK_p < 0.5] = R_min
        
        R_f.append(R_f_)
    
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

def _check_inputs(R, V, method):
  if method == 1:
    if len(R.shape) != 2:
      raise ValueError("R must be a two-dimensional array")
  else:
    if len(R.shape) != 3:
      raise ValueError("R must be a three-dimensional array")
    if R.shape[1] != R.shape[2]:
      raise ValueError("the dimensions of the input fields are %dx%d, but square shape expected" % \
                         (R.shape[1], R.shape[2]))
  
  if len(V.shape) != 3:
    raise ValueError("V must be a three-dimensional array")

def _print_ar2_params(PHI, include_perturb_term):
  if include_perturb_term:
    print("****************************************")
    print("* AR(2) parameters for cascade levels: *")
    print("****************************************")
    print("------------------------------------------------------")
    print("| Level |       0      |       1      |       2      |")
    print("------------------------------------------------------")
    for k in range(PHI.shape[0]):
      print("| %-5d | %-13.6f | %-13.6f | %-13.6f |" % \
            (k+1, PHI[k, 0], PHI[k, 1], PHI[k, 2]))
      print("------------------------------------------------------")
  else:
    print("****************************************")
    print("* AR(2) parameters for cascade levels: *")
    print("****************************************")
    print("---------------------------------------")
    print("| Level |       1      |       2      |")
    print("---------------------------------------")
    for k in range(PHI.shape[0]):
      print("| %-5d | %-13.6f | %-13.6f |" % \
            (k+1, PHI[k, 0], PHI[k, 1]))
      print("------------------------------------------------------")

def _print_corrcoefs(GAMMA):
  print("************************************************")
  print("* Correlation coefficients for cascade levels: *")
  print("************************************************")
  print("-----------------------------------------")
  print("| Level |     Lag-1     |      Lag-2    |")
  print("-----------------------------------------")
  for k in range(GAMMA.shape[0]):
      print("| %-5d | %-13.6f | %-13.6f |" % (k+1, GAMMA[k, 0], GAMMA[k, 1]))
      print("-----------------------------------------")

def _stack_cascades(R_d, num_levels):
  R_c   = []
  mu    = np.empty(num_levels)
  sigma = np.empty(num_levels)
  
  for i in xrange(num_levels):
      R_ = []
      for j in xrange(3):
          mu_    = R_d[j]["means"][i]
          sigma_ = R_d[j]["stds"][i]
          if j == 2:
              mu[i]    = mu_
              sigma[i] = sigma_
          R__ = (R_d[j]["cascade_levels"][i, :, :] - mu_) / sigma_
          R_.append(R__)
      R_c.append(np.stack(R_))
  
  return np.stack(R_c),mu,sigma

"""Implementations of nowcasting methods."""

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

def s_prog(R, V, num_timesteps, extrap_method):
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
  
  Returns
  -------
  out : ndarray
    Three-dimensional array of shape (num_timesteps,m,n) containing a time 
    series of nowcast precipitation fields.
  """
  pass

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

"""Miscellaneous utility methods."""

from numpy import dstack

def read_input_files(inputfns, importer, **kwargs):
  """Read a list of input files using iotools and stack them into a 3d array.
  
  Parameters
  ----------
  inputfns : list
    List of input files returned by any function implemented in iotools.archive.
  importer : function
    Any function implemented in iotools.importers.
  kwargs : dict
    Optional keyword arguments for the importer.
  """
  R = []
  for ifn in inputfns[::-1]:
    R_ = importer(ifn[0], **kwargs)[0]
    R.append(R_)
  
  return dstack(R)
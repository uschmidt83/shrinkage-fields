%STRUCT_PUT - Set fields of a STRUCT based on the names and contents of provided arguments.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function s = struct_put(s, varargin)
  for i = 1:numel(varargin)
    s.(inputname(i+1)) = varargin{i};
  end
end
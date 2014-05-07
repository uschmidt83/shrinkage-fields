%STRUCT2VEC - Convert a STRUCT to a row vector.
%   See also VEC2STRUCT.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function vec = struct2vec(theta)
  assert(isstruct(theta));
  % theta = orderfields(theta);
  fnames = fieldnames(theta);
  vec = [];
  for i = 1:numel(fnames)
    value = theta.(fnames{i});
    vec = [vec; value(:)]; %#ok
  end
end
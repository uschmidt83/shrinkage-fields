%VEC2STRUCT - Convert a row vector to a STRUCT (based on the provided template).
%   See also STRUCT2VEC.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function theta = vec2struct(vec, THETA)
  assert(isstruct(THETA));
  % theta = orderfields(THETA);
  theta = THETA;
  fnames = fieldnames(theta);
  e = 0; % end-point
  for i = 1:numel(fnames)
    field_size = size(theta.(fnames{i}));
    field_numel = prod(field_size);
    % special case of last field that takes remaining elements of vec
    if field_numel == 0 && i == numel(fnames)
      field_numel = numel(vec)-e;
      field_size = [field_numel,1];
    end
    field_value = reshape( vec(e+1:e+field_numel), field_size );
    theta.(fnames{i}) = field_value;
    e = e + field_numel;
  end
end
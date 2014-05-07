%SOLVE_Z - Compute and shrink filter responses of input image (and compute several gradients).
%   See also SOLVE_X.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function varargout = solve_z(x, shrinkage, s)
  varargout = repmat({cell(1,s.nfilters)},1,nargout);
  tmp = cell(1,nargout);

  for j = 1:s.nfilters
    tmp{end} = train.filter_circ_conv(x,j,s);
    % compute only what is requested
    [tmp{1:min(nargout,3)}] = shrinkage(j).eval(tmp{end}(:)',~s.use_lut);
    % first output is shrunk filtered image
    tmp{1} = reshape(tmp{1},s.imdims); 
    % other outputs are gradients w.r.t. RBF weights and filter responses (Eqs. 44 and 42 of the supplemental material)
    for i = 1:nargout, varargout{i}{j} = tmp{i}; end
  end
end
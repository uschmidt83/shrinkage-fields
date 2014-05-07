%SOLVE_X - Inference to obtain restored image by solving linear equation system.
%   See also SOLVE_Z.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [x, bottom, top] = solve_x(z, lambda, s, Kty_, KtK_)
  % z are filter values passed through shrinkage functions
  if nargin < 5, KtK_ = 1; end % denoising
  Fz = zeros(s.imdims);
  for j = 1:s.nfilters
    Fz = Fz + train.filter_circ_corr(z{j},j,s);
  end
  bottom = s.Filt_   + lambda * KtK_;
  top    = fft2(Fz)  + lambda * Kty_;
  x = real(ifft2( top ./ bottom ));
end
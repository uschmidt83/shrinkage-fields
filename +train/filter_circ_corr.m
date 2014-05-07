%FILTER_CIRC_CORR - 2D correlation with circular boundary handling.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function r = filter_circ_corr(img,j,s)
  r = imfilter(img,s.f{j},'same','circular','corr');
  % r = real( ifft2( s.f_otf_tr{j} .* fft2(img)) );
end

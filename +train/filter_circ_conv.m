%FILTER_CIRC_CONV - 2D convolution with circular boundary handling.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function r = filter_circ_conv(img,j,s)
  r = imfilter(img,s.f{j},'same','circular','conv');
  % r = real( ifft2( s.f_otf{j} .* fft2(img)) );
end

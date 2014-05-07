%KERNEL_CIRC - 2D convolution/correlation with circular boundary handling for blur kernel.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function r = kernel_circ(img,k,ctype,~)
  % equivalent to r = imfilter(img,k,'same','circular',ctype)
  % fft-based convolution/correlation typically faster due to larger kernel sizes
  switch ctype
    case 'conv'
      r = real( ifft2( psf2otf(k,size(img)) .* fft2(img)) );
    case 'corr'
      r = real( ifft2( conj(psf2otf(k,size(img))) .* fft2(img)) );
  end
end

%LOSS_AND_C - Compute (negtive) loss and c as defined in Eq. (20) of the supplemental material.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [f,c] = loss_and_c(x,x_gt,bottom,s)  
  x       = s.rst(s.T*x(:));
  x_gt    = s.rst(s.T*x_gt(:));
  [f,lgx] = train.psnr(x,x_gt);

  if nargout > 1
    lgx = s.rs(s.T'*lgx(:));
    c   = real(ifft2( fft2(lgx) ./ bottom ));
  end
end

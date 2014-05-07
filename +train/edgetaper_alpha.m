%EDGETAPER_ALPHA - Weights for edge tapering.
%   Based on Matlab's EDGETAPER function,
%   and used as in Sec. 1.1. of the supplemental material.
%
%   See also EDGETAPER.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function alpha = edgetaper_alpha(k,imdims)
  assert(all(imdims > 2*size(k)), 'kernel too small for image size')
  alpha = 1;
  for i = 1:2
    z = real(ifft(abs(fft(sum(k,3-i),imdims(i)-1)).^2));
    alpha = alpha * (1 - z([1:end,1])/max(z));
  end
end
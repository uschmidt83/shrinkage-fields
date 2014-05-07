%FIX_BNDRY - Boundary operations (padding, truncation, edge tapering).
%   See Sec. 1.1. of the supplemental material.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function I = fix_bndry(I,k,alpha,s,transpose)
  if ~exist('transpose','var'), transpose = false; end
  if transpose
    if numel(k)~=1 || k ~= 1 % no edgetapering when denoising
      I = s.rs(I);
      for i = 1:s.ntapers
        blurredI = train.kernel_circ((1-alpha).*I,k,'corr',s);
        I = alpha.*I + blurredI;
      end
    end
    I = s.rs(s.PT' * I(:));
  else
    I = s.rs(s.PT * I(:));
    if numel(k)~=1 || k ~= 1 % no edgetapering when denoising
      for i = 1:s.ntapers
        blurredI = train.kernel_circ(I,k,'conv',s);
        I = alpha.*I + (1-alpha).*blurredI;
      end
    end
  end
end
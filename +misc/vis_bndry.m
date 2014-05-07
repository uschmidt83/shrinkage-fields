%VIS_BNDRY - Visualize image boundary (and embed blur kernel).

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function x = vis_bndry(x,k,t)
  if t > 0
    % scale by inner region
    x_inside = x(1+t:end-t,1+t:end-t);
    x = x - min(x_inside(:));
    x = x / max(x_inside(:));
    % clip values at boundary
    x = min(max(x,0),1);

    x = repmat(x,[1,1,3]);
    x([1+t,end-t],1+t:end-t,:) = 0;
    x([1+t,end-t],1+t:end-t,2) = 1;
    x(1+t:end-t,[1+t,end-t],:) = 0;
    x(1+t:end-t,[1+t,end-t],2) = 1;
    if numel(k) > 1
      kdims = size(k);
      k = k / max(k(:));
      x(1:kdims(1),1:kdims(2),:) = repmat(k,[1,1,3]);
    end
  end
end
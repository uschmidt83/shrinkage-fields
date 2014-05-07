%EVAL - Evaluate function value and compute gradients.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [p,gw,gx,q] = eval(this, x, no_lut)
  % p  = function value
  % gw = gradient wrt. RBF weights
  % gx = gradient wrt. input
  % q  = individual RBF values
  
  assert(size(x,1)==1, 'only univariate row vectors allowed');

  if nargin == 3 && no_lut
    %% compute on demand
    x_mu = bsxfun(@minus, x, this.means(:));
    gw = exp((-0.5 * this.precision) * x_mu.^2);
    q = bsxfun(@times, gw, this.weights(:));
    gx = -this.precision * sum(bsxfun(@times, q, x_mu),1);
    p = sum(q,1);

  elseif ~this.has_mex || this.use_gpu
    %% lookup with linear interpolation
    x_hit = (x-this.offsetD)/this.step;
    xl = floor(x_hit)+1;
    xh = ceil(x_hit)+1;

    % assume that requested index is valid
    % xl(xl < 1) = 1; xl(xl > this.nD) = this.nD;
    % xh(xh < 1) = 1; xh(xh > this.nD) = this.nD;
    % assert(all(xl >= 1) && all(xh >= 1) && all(xl <= this.nD) && all(xh <= this.nD))

    wh = x_hit-(xl-1);
    pl = this.P(xl);
    ph = this.P(xh);
    p = pl + (ph-pl).*wh;

    assert(~(this.use_gpu && nargout > 1), 'GPU-based learning not implemented.')
    if nargout > 1
      gwl = this.G(:,xl);
      gwh = this.G(:,xh);
      gw = bsxfun(@times,gwl,1-wh) + bsxfun(@times,gwh,wh);
      if nargout > 2
        gxl = this.GX(xl);
        gxh = this.GX(xh);
        gx = gxl + (gxh-gxl).*wh;
        if nargout > 3
            ql = this.Q(:,xl);
            qh = this.Q(:,xh);
            q = ql + bsxfun(@times,qh-ql,wh);
        end
      end
    end

  else
    %% lookup with linear interpolation via mex function (identical to above)
    switch nargout
      case 1, [p]         = lut_eval(x, this.offsetD, this.step, this.P, this.G, this.GX, this.Q);
      case 2, [p,gw]      = lut_eval(x, this.offsetD, this.step, this.P, this.G, this.GX, this.Q);
      case 3, [p,gw,gx]   = lut_eval(x, this.offsetD, this.step, this.P, this.G, this.GX, this.Q);
      case 4, [p,gw,gx,q] = lut_eval(x, this.offsetD, this.step, this.P, this.G, this.GX, this.Q);
    end
  end

end

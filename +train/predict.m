%PREDICT - Prediction for all training images at a particular stage of a shrinkage field cascade.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [U,psnrs] = predict(U, theta, data, s, shrinkage)

  THETA = misc.vec2struct(theta,s.THETA);
  lambda = s.pos.from_raw(THETA.lambda);
  shrinkage = rbfmix.update(shrinkage,THETA.weights);
  if s.do_filterlearning
    s = train.model_filter_update(s,THETA.filters);
  end  
  psnrs = zeros(s.nimages,1);

  for i = 1:s.nimages
    fprintf(' %04d/%04d\r', i, s.nimages);
    x_old    = s.rs(U(:,i));
    x_gt     = s.rs(data.X(:,i));
    z        = train.solve_z(x_old, shrinkage, s);
    xbnd     = train.solve_x(z, lambda, s, data.Kty_(:,:,i), data.KtK_(:,:,i));
    psnrs(i) = train.loss_and_c(xbnd,x_gt,[],s);
    xpad     = train.fix_bndry(xbnd,data.ks{i},data.alphas(:,:,i),s);
    U(:,i)   = xpad(:);
  end
end

%OBJECTIVE_ONE_STAGE - Objective function to (greedily) train (a single model stage of) a shrinkage field (cascade).
%   See Sec. 1.2.1 of the supplemental material.
%   See also OBJECTIVE_ALL_STAGES, GRAD_PARAMS.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [F,G] = objective_one_stage(U, theta, data, s, shrinkage)

  THETA     = misc.vec2struct(theta,s.THETA);
  weights   = THETA.weights;
  lambda    = s.pos.from_raw(THETA.lambda);
  shrinkage = rbfmix.update(shrinkage,weights);

  if s.do_filterlearning, [s,fnorms] = train.model_filter_update(s,THETA.filters);
  else                       fnorms  = []; end

  train.plot_progress(s,shrinkage,THETA);
  randomimage = randsample(s.nimages,1);
  
  [F,G] = deal(0);
  %% loop over all training images
  for i = 1:s.nimages
    y     = s.rs(data.Y(:,i));
    x_gt  = s.rs(data.X(:,i));
    x_old = s.rs(U(:,i));
    [f,g] = one_stage_one_image(i, x_gt, x_old, y, data.Kty_(:,:,i), data.KtK_(:,:,i), data.ks{i},...
                                s, shrinkage, lambda, fnorms, s.do_plot && i==randomimage);
    F = F + f;
    G = G + g;
  end

  % negate since we're minimizing (also normalize)
  F = -F/s.nimages; G = -G/s.nimages;
end

function [f,g] = one_stage_one_image(i, x_gt,x_old,y,Kty_,KtK_,k,s,shrinkage,lambda,fnorms,show_x)
  fprintf(' %04d/%04d\r', i, s.nimages);
  [z,gw,gx,fx_old] = train.solve_z(x_old, shrinkage, s);
  [x, bottom]      = train.solve_x(z, lambda, s, Kty_, KtK_);
  [f,c]            = train.loss_and_c(x,x_gt,bottom,s);
  g                = train.grad_params(s,x,x_old,y,k,[],c,lambda,gw,gx,z,fx_old,fnorms,[]);
  if show_x
    m = 2 + s.do_filterlearning;
    subplot(1,m,m)
    imagesc(misc.vis_bndry(x,k,s.discard_bndry)), axis image on
    title 'restoration example'
  end
end
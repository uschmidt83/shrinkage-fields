%OBJECTIVE_ALL_STAGES - Objective function for jointly training all model stages of a shrinkage field cascade.
%   See Sec. 1.2.1 of the supplemental material.
%   See also OBJECTIVE_ONE_STAGE, GRAD_PARAMS.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [F,G] = objective_all_stages(theta, data, s, shrinkage)

  if numel(s) > 1, ss = s; s = ss(1);
  else             ss = repmat(s,1,s.nstages); end

  fnorms           = cell(1,s.nstages);
  [weights,lambda] = deal([]);
  THETA            = repmat(s.THETA,1,s.nstages);
  for i = 1:s.nstages
    THETA(i) = misc.vec2struct(theta(:,i),s.THETA);
    weights  = [weights, THETA(i).weights];    %#ok
    lambda   = [lambda, s.pos.from_raw(THETA(i).lambda)]; %#ok
    if s.do_filterlearning
      [ss(i),fnorms{i}] = train.model_filter_update(ss(i),THETA(i).filters);
    end  
  end
  shrinkage = rbfmix.update(shrinkage,weights);

  train.plot_progress(ss,shrinkage,THETA);
  randomimage = randsample(s.nimages,1);
  
  [F,G] = deal(0);
  %% loop over all training images
  for i = 1:s.nimages
    y     = s.rs(data.Y(:,i));
    x_gt  = s.rs(data.X(:,i));
    [f,g] = all_stages_one_image(i, x_gt, y, data.Kty_(:,:,i), data.KtK_(:,:,i), data.ks{i}, data.alphas(:,:,i), ...
                                 s, ss, shrinkage, lambda, fnorms, s.do_plot && i==randomimage);
    F = F + f;
    G = G + g;
  end

  % negate since we're minimizing (also normalize)
  F = -F/s.nimages; G = -G/s.nimages;
end

function [f,g] = all_stages_one_image(i,x_gt,y,Kty_,KtK_,kernel,alpha,s,ss,shrinkage,lambda,fnorms,show_x)
  fprintf(' %04d/%04d\r', i, s.nimages);

  [gw,z,gx,fx_old]     = deal(cell(s.nfilters,s.nstages));
  [xbnd,xpad,c,bottom] = deal(zeros([s.imdims,s.nstages]));
  g                    = cell(1,s.nstages);

  %% prediction (forward pass)
  for k = 1:s.nstages
    if k == 1, x_old = y;
    else       x_old = xpad(:,:,k-1); end
    [z(:,k),gw(:,k),gx(:,k),fx_old(:,k)] = train.solve_z(x_old, shrinkage(:,k), ss(k));
    [xbnd(:,:,k), bottom(:,:,k)]         = train.solve_x(z(:,k), lambda(k), ss(k), Kty_, KtK_);
    xpad(:,:,k)                          = train.fix_bndry(xbnd(:,:,k),kernel,alpha,s);
  end
  % compute loss and last c
  [f,c(:,:,end)] = train.loss_and_c(xbnd(:,:,end),x_gt,bottom(:,:,end),s);

  %% gradients (backward pass)
  for k = s.nstages:-1:1
    if k == 1, [g{k}]            = train.grad_params(ss(k),xbnd(:,:,k),y            ,y,kernel,alpha,c(:,:,k),lambda(k),gw(:,k),gx(:,k),z(:,k),fx_old(:,k),fnorms{k},[]);
    else       [g{k},c(:,:,k-1)] = train.grad_params(ss(k),xbnd(:,:,k),xpad(:,:,k-1),y,kernel,alpha,c(:,:,k),lambda(k),gw(:,k),gx(:,k),z(:,k),fx_old(:,k),fnorms{k},bottom(:,:,k-1)); end
  end
  g = vertcat(g{:});

  %% plot
  if show_x
    m = 2 + s.do_filterlearning;
    for k = 1:s.nstages
      subplot(s.nstages,m,m+(k-1)*m)
      imagesc(misc.vis_bndry(xbnd(:,:,k),kernel,s.discard_bndry)), axis image on
      title 'restoration example'
    end
  end
end
%INIT_PARAMS - Initial model parameters.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [shrink,THETA,learning] = init_params(s)
  % shrinkage function (~ identity function)
  shrink           = struct;
  shrink.precision = 0.01;
  shrink.means     = (-260:10:260)';
  shrink.weights   = [-223.089, -7.62753, -162.908, -41.7806, -124.963, -55.0791, -101.516, -58.5456, -84.9205, -57.0593, -72.0137, -52.7463, -61.3176, -46.6513, -52.0907, -39.2833, -44.0039, -30.8134, -37.0164, -21.1507, -31.3142, -10.0267, -27.0809, 2.36091, -22.5024, 9.79195, 0.000109765, -9.79217, 22.5026, -2.3611, 27.081, 10.0265, 31.3143, 21.1506, 37.0164, 30.8134, 44.0038, 39.2834, 52.0907, 46.6513, 61.3175, 52.7464, 72.0137, 57.0593, 84.9205, 58.5456, 101.516, 55.079, 124.963, 41.7806, 162.908, 7.62747, 223.089]';

  % THETA, theta0: initialization for model parameters
  THETA         = struct;
  THETA.lambda  = s.pos.to_raw(1e-1);
  THETA.weights = repmat(shrink.weights,s.nfilters,1);
  if s.do_filterlearning
  THETA.filters = reshape(eye(size(s.filter_basis,2)),[],1);
  end

  theta0  = misc.struct2vec(THETA);
  nparams = numel(theta0);
  nstages = s.nstages;

  % to store parameters during learning
  learning       = misc.struct_put(struct, nparams, nstages);
  learning.THETA = repmat(THETA,1,nstages+1);
  learning.theta = [theta0, nan(nparams,nstages)];
  learning.psnrs = nan(s.nimages,nstages);
end
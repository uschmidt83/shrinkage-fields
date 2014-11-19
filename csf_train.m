%CSF_TRAIN - Train (cascades of) shrinkage fields.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function csf_train

  %% OPTIONS
  pw         = false; % pairwise model
  deblurring = false; % deblurring instead of denoising  
  static = struct;

  % MODEL
  static.nstages           = 3;
  static.unitnorm_filter   = true;
  if pw
  static.fbank             = 'pw';
  static.fbank_sz          = [];
  static.do_filterlearning = false;
  static.discard_bndry     = 5;
  else
  static.fbank             = 'dct';
  static.fbank_sz          = 3;
  static.do_filterlearning = true;
  static.discard_bndry     = 2*static.fbank_sz;
  end

  % APPLICATION
  if deblurring
  static.is_deblurring     = true;
  static.sigma             = 0.5;
  else
  static.is_deblurring     = false;
  static.sigma             = 25;
  end

  % LEARNING
  static.nimages           = 5;
  static.use_lut           = true;
  static.do_plot           = true;
  static.maxoptiters       = 25;
  static.do_joint_training = false;
  if deblurring
  static.ntapers           = 3;
  static.k_sz_max          = 37; % largest kernel size in training set (assumed to be bigger than filter size)
  static.discard_bndry     = (static.k_sz_max-1)/2;
  end
  % constrain lambda to be positive:
  % static.pos = train.pos_exp;    % as used in the paper
  static.pos = train.pos_expident; % typically works better
  

  %% SETUP
  static.imdims  = [128,128];
  static.npixels = prod(static.imdims);
  static         = train.get_filters(static);
  static         = train.precompute_model(static);  
  data           = train.get_data('data',static);
  data           = train.precompute_data(data);

  U = data.Y; % prediction from previous stage (initialize with observed data)
  [static.shrink,static.THETA,learning] = train.init_params(static);
  % note: recommended for joint training to use parameters from
  %       greedily trained models as initialization, e.g.:
  % load('model.mat','learning');
  % theta0 = arrayfun(@(t){misc.struct2vec(t)}, learning.THETA(2:end));
  % theta0 = vertcat(theta0{:});

  if static.do_plot, figure(1), clf, colormap(gray(256)), end

  %% LEARNING
  experiment = 'model.mat';
  if static.do_joint_training
    if ~exist('theta0','var'), theta0 = repmat(learning.theta(:,1),static.nstages,1); end
    shrinkage = rbfmix.from_struct(repmat(static.shrink,static.nfilters,static.nstages));
    cost_func = @(theta) train.objective_all_stages(reshape(theta,[],static.nstages), data, static, shrinkage);
    minimizer = train.get_minimizer(static,cost_func,theta0);
    learning.theta(:,2:end) = reshape(minimizer(),[],static.nstages);
    for i = 1:static.nstages
      learning.THETA(i+1)       = misc.vec2struct(learning.theta(:,i+1),static.THETA);
      [U,learning.psnrs(:,i+1)] = train.predict(U, learning.theta(:,i+1), data, static, shrinkage(:,i));
    end
    if ~isempty(experiment), save(experiment,'static','learning'); end
  else
    if ~exist('theta0','var'), theta0 = learning.theta(:,1); end
    shrinkage = rbfmix.from_struct(repmat(static.shrink,static.nfilters,1));
    cost_func = @(theta,U) train.objective_one_stage(U, theta, data, static, shrinkage);
    minimizer = train.get_minimizer(static,cost_func,theta0);
    for i = 1:static.nstages
      learning.theta(:,i+1)     = minimizer(U);
      learning.THETA(i+1)       = misc.vec2struct(learning.theta(:,i+1),static.THETA);
      [U,learning.psnrs(:,i+1)] = train.predict(U, learning.theta(:,i+1), data, static, shrinkage);
      if ~isempty(experiment), save(experiment,'static','learning'); end
    end
  end
  
  fprintf('\nAvg. PSNR on training set:\n');
  for i = 1:static.nstages
    fprintf('- Stage %d: %.2fdB\n', i, mean(learning.psnrs(:,i+1),1));
  end
  
end

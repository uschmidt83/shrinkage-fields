%RBFMIX - Linear combination of Gaussian radial basis function (RBF) kernels.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

classdef rbfmix

  properties
    precision = [];
    means     = [];
    weights   = [];
  end
  
  properties (Dependent)
    nmeans;
  end

  properties (Constant,Hidden)
    step = 2^-2;
  end

  properties (SetAccess='private')
    use_gpu = false;
  end

  properties (SetAccess='private',Hidden)
    D; G; Q; P; GX; offsetD; nD; has_mex;
  end

  methods(Static)
    function mix = from_struct(s,use_gpu)
      if nargin == 1, use_gpu = false; end
      mix = repmat(rbfmix(use_gpu),size(s));
      for i = 1:numel(s)
        mix(i).precision = s(i).precision;
        mix(i).means     = s(i).means;
        mix(i).weights   = s(i).weights;
      end
    end
    function s = to_struct(mix)
      s = repmat(struct('precision',[],'means',[],'weights',[]),size(mix));
      for i = 1:numel(mix)
        s(i).precision = mix(i).precision;
        s(i).means     = mix(i).means;
        s(i).weights   = mix(i).weights;
      end
    end
    function mix = update(mix,weights)
      % assumes that all have same #components
      nmix = numel(mix);
      W = reshape(weights,[],nmix);
      for i = 1:nmix
        mix(i).weights = W(:,i);
      end
    end
  end
  
  methods
    function this = rbfmix(use_gpu)
      this.has_mex = exist('lut_eval') == 3; %#ok
      if this.has_mex; try; lut_eval; catch; this.has_mex = false; end; end %#ok
      if nargin == 1, this.use_gpu = use_gpu; end
    end

    function this = set.means(this, m)
      assert(~isempty(this.precision), 'set precision first')
      assert(all(m==sort(m)), 'means must be sorted')
      this.means = m(:);
      % precompute stuff
      this.D = -500+this.means(1):this.step:this.means(end)+500;
      this.offsetD = this.D(1);
      this.nD = numel(this.D);
      D_mu = bsxfun(@minus, this.D, this.means(:));
      this.G = exp(-0.5 * this.precision * D_mu.^2);
    end

    function this = set.weights(this, w)
      assert(~isempty(this.precision), 'set precision first')
      assert(this.nmeans == numel(w),  'set means first')
      this.weights = w(:);
      % precompute more stuff
      this.Q = bsxfun(@times, this.G, this.weights(:));
      this.P = sum(this.Q,1);
      % gradient w.r.t. x (input)
      D_mu = bsxfun(@minus, this.D, this.means(:));
      this.GX = -this.precision * sum(bsxfun(@times, this.Q, D_mu),1);
      if this.use_gpu
        % only for function value, not gradients for learning
        this.P = gpuArray(single(this.P));
      end
    end
    
    function nmeans = get.nmeans(this)
      nmeans = numel(this.means);
    end  
  end
  %#ok<*MCSUP>

end

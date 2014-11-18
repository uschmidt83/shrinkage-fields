%GRAD_PARAMS - Compute gradient w.r.t. all model parameters of one stage of a shrinkage field cascade.
%   See Sec. 1.2.2 of the supplemental material.
%   See also OBJECTIVE_ONE_STAGE, OBJECTIVE_ALL_STAGES.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [g,c_prev] = grad_params(s,x,x_old,y,k,alpha,c,lambda,gw,gx,z,fx_old,fnorms,bottom)
  % c_prev is for the previous model stage as defined in Eq. (34) of the supplemental material

  GRAD      = struct;
  do_c_prev = nargout == 2;
  g_weights = zeros(size(gw{1},1),s.nfilters);
  if s.do_filterlearning, g_filt = zeros(size(s.filter_basis,2),s.nfilters); end
  if do_c_prev,           c_prev = zeros(s.imdims); end

  if s.do_filterlearning
    cliques_c     = s.filter_basis' * c(s.cliques_of_circ_conv);
    cliques_x     = s.filter_basis' * x(s.cliques_of_circ_conv);
    cliques_x_old = s.filter_basis' * x_old(s.cliques_of_circ_conv);
  end

  % grad wrt. (log) lambda
  if s.is_deblurring
    kc = train.kernel_circ(c,k,'conv',s);
    kx = train.kernel_circ(x,k,'conv',s);
    GRAD.lambda = kc(:)' * (y(:)-kx(:));
  else
    GRAD.lambda =  c(:)' * (y(:)- x(:));
  end
  GRAD.lambda = GRAD.lambda * s.pos.d_raw(lambda);

  for j = 1:s.nfilters
    fc = s.vec(train.filter_circ_conv(c,j,s));
    % grad wrt. shrinkage weights
    g_weights(:,j) = gw{j} * fc;
    fc_gx          = fc .* gx{j}(:);
    if s.do_filterlearning
      % grad wrt. filter coefficients
      fx          = s.vec(train.filter_circ_conv(x,j,s));
      z_minus_fx  = z{j}(:) - fx;
      g_filt(:,j) = cliques_x_old*fc_gx + cliques_c*z_minus_fx - cliques_x*fc;
      if s.unitnorm_filter
        g_filt(:,j) = g_filt(:,j) - s.filter_basis' * s.f{j}(:) * (fc' * (fx_old{j}(:).*gx{j}(:) + z_minus_fx - fx));
        g_filt(:,j) = g_filt(:,j) / fnorms(j);
      end
    end
    % c for previous stage
    if do_c_prev
      c_prev = c_prev + train.filter_circ_corr(s.rs(fc_gx),j,s);
    end
  end
  % c for previous stage
  if do_c_prev
    c_prev = real(ifft2( fft2(train.fix_bndry(c_prev,k,alpha,s,true)) ./ bottom ));
  end

  GRAD.weights = g_weights(:);
  if s.do_filterlearning, GRAD.filters = g_filt(:); end
  g = misc.struct2vec(GRAD);

end
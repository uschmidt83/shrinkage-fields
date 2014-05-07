%CSF_PREDICT - Denoising and deconvolution inference for (cascades of) shrinkage fields.
%   OUT = CSF_PREDICT(M,Y,K) returns a cell array of denoising/deconvolution results for
%   each stage of the shrinkage field cascade M. Y is the corrupt input image, and K
%   the blur kernel (point spread function) when deconvolution is performed.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function out = csf_predict(m,y,k)
  %% check for GPU
  use_gpu = false;
  try
    % note: first run on GPU always takes longer than subsequent runs
    gpu = gpuDevice; if ~gpu.DeviceSupported, error; end %#ok
    use_gpu  = true;
    transfer = @(d) gpuArray(single(d));
    retain   = @gather;
    fprintf('Using GPU[%d] ''%s'' (free memory: %.0fMB)\n', gpu.Index, gpu.Name, gpu.FreeMemory/1024^2)
  catch
    [transfer,retain] = deal(@(d)d);
  end

  %% setup
  tic
  y = transfer(y);
  m.shrink = rbfmix.from_struct(m.shrink,use_gpu); % create @rbfmix objects (computes LUTs)
  if m.is_deblurring
    assert(exist('k','var') ~= 0,  'kernel not provided')
    assert(all(mod(size(k),2)==1), 'kernel must be odd-sized')
    kdims   = size(k);
    k_otf   = transfer(psf2otf(k,size(y)+kdims-1));
    k_alpha = train.edgetaper_alpha(transfer(k),size(y)+kdims-1);
    %
    bndry = (kdims-1)/2;
    pad   = @(x) taper(padarray(x,bndry,'replicate','both'),k,k_otf,k_alpha,3);
    y     = pad(y);
    Kty   = real(ifft2(conj(k_otf).*fft2(y))); % same as imfilter(y,k,'same','circular','corr');
    KtK_  = abs(k_otf).^2;
  else
    bndry = [10,10];
    pad   = @(x) padarray(x,bndry,'replicate','both');
    y     = pad(y);
    Kty   = y;
    KtK_  = 1;
  end
  crop = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
  out = cell(1,m.nstages);
  
  % pre-compute some things
  m.imdims = size(y);
  m = circpadidx(m,transfer);
  m.zeroimg = transfer(zeros(m.imdims));  
  % store filters for all stages
  [f,f_tr] = deal(cell(size(m.f)));
  for i = 1:numel(f)
    f{i}    = transfer(m.f{i});
    f_tr{i} = transfer(flipud(fliplr(m.f{i}))); %#ok
  end

  %% inference
  x = y; % initial result is input image
  fprintf('- Setup   done, elapsed time = %5.2fs\n', toc)
  for i = 1:m.nstages
    [m.f,m.f_tr] = deal(f(:,i),f_tr(:,i));
    x = predict(x, m.lambda(i), m.shrink(:,i), Kty, KtK_, m);
    x = crop(x); out{i} = retain(x); x = pad(x);
    fprintf('- Stage %d done, elapsed time = %5.2fs\n', i, toc)
  end
end


% actual inference, computes Eq. (10) of the paper
function x = predict(x, lambda, shrink, Kty, KtK_, s)
  top    = lambda * Kty;
  bottom = lambda * KtK_;
  for j = 1:s.nfilters
    bottom = bottom + abs(mypsf2otf(j,s)).^2;
    fx  = filter_circ_conv(x,s,j);
    z   = reshape(shrink(j).eval(fx(:)'),s.imdims);
    fz  = filter_circ_corr(z,s,j);
    top = top + fz;
  end
  clear fx z fz % free (GPU) memory
  top = fft2(top);
  x = real(ifft2( top ./ bottom ));
end

% GPU-friendly 2D convolution, results equivalent to imfilter(x,s.f{i},'same','circular',{'conv','corr'});
function x = filter_circ_conv(x,s,j), x = conv2(x(s.circpadidx{j}),s.f{j},   'valid'); end
function x = filter_circ_corr(x,s,j), x = conv2(x(s.circpadidx{j}),s.f_tr{j},'valid'); end

% edge tapering to alleviate artifacts from circular boundary handling
function x = taper(x,k,k_otf,k_alpha,ntapers) %#ok
  for i = 1:ntapers
    % basically equivalent to x = edgetaper(x,k);
    blurred = real(ifft2(fft2(x).*k_otf));
    x = k_alpha.*x + (1-k_alpha).*blurred;
  end
end

% compute circular padding indices for all filters (assume same sizes for all stages)
% used for 2D convolution and mypsf2otf
function s = circpadidx(s,transfer)
  idx = reshape(1:prod(s.imdims),s.imdims);
  s.circpadidx = cell(1,s.nfilters);
  fdims = size(s.f{1});
  if fdims(1) == fdims(2)
    % learned filters (assume that all have same size)
    s.circpadidx(:) = {transfer(padarray(idx,(fdims-1)/2,'circular','both'))};
  else
    % pairwise filters (different sizes)
    for i = 1:s.nfilters
      s.circpadidx{i} = transfer(padarray(idx,(size(s.f{i})-1)/2,'circular','both'));
    end
  end
end

% optimized and GPU-friendly psf2otf, basically equivalent to psf2otf(s.f{j},s.imdims)
function otf = mypsf2otf(j,s)
  fdims = size(s.f{j});
  psf = s.zeroimg;
  idx_circ = s.circpadidx{j}(1:fdims(1),1:fdims(2));
  psf(idx_circ) = s.f{j};
  otf = fft2(psf);
end

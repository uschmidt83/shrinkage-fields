%PRECOMPUTE_MODEL - Compute model-related properties that do not or rarely change during training.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function s = precompute_model(s)

  % index image with circular padding
  fdims = size(s.f{1});
  idx = reshape(1:s.npixels,s.imdims);
  circpadidx = padarray(idx, (fdims-1)/2, 'circular','both');
  % assuming all filters have same size (used for learning)
  cliques_of_circ_conv = flipud(im2col(circpadidx,fdims,'sliding'));

  % psf2otf for all filters (for each stage when s.f is a multi-dimensional array)
  [nfilters,nstages] = size(s.f);
  [f_otf,f_otf_tr] = deal(cell(size(s.f)));
  Filt_ = zeros([s.imdims,nstages]);
  for j = 1:nstages
    for i = 1:nfilters
      f_otf{i,j} = psf2otf(s.f{i,j},s.imdims);
      f_otf_tr{i,j} = conj(f_otf{i,j});
      Filt_(:,:,j) = Filt_(:,:,j) + abs(f_otf{i,j}).^2;
    end
  end
  

  if ~isfield(s,'discard_bndry'), s.discard_bndry = 0; end
  imdims = s.imdims;
  t = s.discard_bndry;
  
  % shorthand for reshaping
  rs = @(x) reshape(x,imdims);
  % reshape to truncated image size
  rst = @(x) reshape(x,imdims-2*t);
  % reshape to column vector
  vec = @(x) reshape(x,[],1);
  
  if t > 0
    if isfield(s,'is_deblurring') && s.is_deblurring
      s.pad_kernel = s.discard_bndry*[1,1];
      s.circpadidx_kernel = padarray(idx,s.pad_kernel,'circular','both');
    end    

    % truncation/cropping matrix T
    [r,c] = ndgrid(1+t:imdims(1)-t, 1+t:imdims(2)-t);
    ind_int = sub2ind(imdims, r(:), c(:));
    d = zeros(s.imdims); d(ind_int) = 1;
    T = spdiags(d(:),0,s.npixels,s.npixels);
    T = T(ind_int,:);

    % padding matrix P that replicates boundary pixels
    % (also called (zero-flux) Neumann boundary condition)
    idximg = reshape(1:prod(imdims-2*t),imdims-2*t);
    pad_idximg = padarray(idximg,[t t],'replicate','both');
    P = sparse((1:s.npixels)',pad_idximg(:),ones(s.npixels,1),s.npixels,prod(imdims-2*t));

    % first truncation, then padding
    PT = P*T;
  else
    [T,P,PT] = deal(1);
  end

  % add things to the struct
  s = misc.struct_put(s, f_otf, f_otf_tr, Filt_, rs, rst, vec, cliques_of_circ_conv, T, P, PT);
end
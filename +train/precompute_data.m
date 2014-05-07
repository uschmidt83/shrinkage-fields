%PRECOMPUTE_DATA - Compute data-related properties that do not change during training.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function s = precompute_data(s)

  is_deblurring = ~isempty(s.ks);
  if is_deblurring
    s.KtK_ = zeros([s.imdims,s.nimages]);
  else
    k_otf_tr = 1;
    s.ks = repmat({1},1,s.nimages);
    [s.KtK_,s.alphas] = deal(ones(1,1,s.nimages));
  end

  s.Kty_ = zeros([s.imdims,s.nimages]);
  for i = 1:s.nimages
    if is_deblurring
      k_otf  = psf2otf(s.ks{i}, s.imdims);
      k_otf_tr = conj(k_otf);
      s.KtK_(:,:,i) = abs(k_otf).^2;
    end
    y = reshape(s.Y(:,i),s.imdims);
    s.Kty_(:,:,i) = k_otf_tr.*fft2(y);
  end

end
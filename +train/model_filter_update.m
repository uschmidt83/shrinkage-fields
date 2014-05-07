%MODEL_FILTER_UPDATE - Recompute things when model filters are updated during training.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [s,fnorms,fraw] = model_filter_update(s, filters)
  s.Filt_ = 0; e = 0;
  fdims   = size(s.f{1});
  f_numel = size(s.filter_basis,2);
  fnorms  = zeros(1,s.nfilters);
  fraw    = zeros(f_numel,s.nfilters);
  for i = 1:s.nfilters
    fraw(:,i) = filters(e+1:e+f_numel);
    s.f{i} = reshape(s.filter_basis * fraw(:,i), fdims);
    if s.unitnorm_filter
      fnorms(i) = sqrt( s.f{i}(:)' * s.f{i}(:) );
      s.f{i} = s.f{i} / fnorms(i);
    end
    s.f_otf{i} = psf2otf(s.f{i},s.imdims);
    s.f_otf_tr{i} = conj(s.f_otf{i});
    s.Filt_ = s.Filt_ + abs(s.f_otf{i}).^2;
    e = e + f_numel;
  end  
end

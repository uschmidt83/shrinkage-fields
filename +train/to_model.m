%TO_MODEL - Convert learned model to be used by CSF_PREDICT

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function model = to_model(filename)
  load(filename)
  assert(exist('static','var') && exist('learning','var'))
  if ~isfield(static,'pos'), static.pos = train.pos_exp; end %#ok

  model = struct;
  for str = {'nfilters','sigma','is_deblurring','nstages'}
    model.(char(str)) = static.(char(str));
  end
  model.lambda = zeros(1,model.nstages);
  model.shrink = repmat(static.shrink,model.nfilters,model.nstages);

  if static.do_filterlearning
    fdims   = size(static.f{1});
    model.f = cell(model.nfilters,model.nstages);
  else
    model.f = repmat(static.f(:),1,model.nstages);
  end

  % first entry contains initial parameters
  THETA = learning.THETA(2:end);

  for j = 1:static.nstages
    model.lambda(j)   = static.pos.from_raw(THETA(j).lambda);
    model.shrink(:,j) = rbfmix.update(model.shrink(:,j),THETA(j).weights);
    if static.do_filterlearning
      filters = static.filter_basis * reshape(THETA(j).filters,[],model.nfilters);
      if static.unitnorm_filter
        filters = bsxfun(@rdivide, filters, sqrt(sum(filters.^2,1)));
      end
      for i = 1:model.nfilters
        model.f{i,j} = reshape(filters(:,i),fdims);
      end
    end
  end
end
%CSF_VISUALIZE - Visualize (cascades of) shrinkage fields.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function csf_visualize(m)

  %% show shrinkage functions
  m.shrink = rbfmix.from_struct(m.shrink);
  MAX = 255; R = -MAX:1e-1:MAX;
  figure('name','Shrinkage functions'), clf
  for i = 1:m.nstages
    V = {}; for k = 1:m.nfilters, V = [V,{R},{m.shrink(k,i).eval(R)}]; end %#ok
    subplot(2,ceil(m.nstages/2),i), plot(R,R,':k',V{:}); axis equal
    axis([-MAX  MAX -300  300])
    % \lambda refers to Eq. (10) of the paper
    title(sprintf('stage %d, \\lambda = %g',i,m.lambda(i)),'interpreter','tex')
  end

  %% show filters
  if m.nfilters > 2
    figure('name','Filters'), colormap(gray(256))
    fdims = size(m.f{1});
    I = cell2mat(m.f');
    I = misc.separate_blocks(I,fdims);
    imagesc(I), axis image off, colorbar %#ok
    title(sprintf('%d filters of size %dx%d for %d stages', m.nfilters, fdims(1), fdims(2), m.nstages))
  end

end
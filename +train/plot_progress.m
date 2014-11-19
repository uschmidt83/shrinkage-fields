%PLOT_PROGRESS - Plot shrinkage functions and filters during training.
%   See also CSF_VISUALIZE.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function plot_progress(ss,shrinkage,THETA)
  s = ss(1);
  if s.do_plot
    MAX = 255; R = -MAX:1e-1:MAX;
    nstages = numel(THETA);
    for i = 1:nstages
      if numel(ss) > 1, s = ss(i); end
      lambda = s.pos.from_raw(THETA(i).lambda);
      V = {}; for k = 1:s.nfilters, V = [V,{R},{shrinkage(k,i).eval(R)}]; end %#ok

      m = 2+s.do_filterlearning; subplot(nstages,m,1+(i-1)*m)
      plot(R,R,':k',V{:}), axis tight equal
      title(sprintf('\\lambda = %g', lambda),'interpreter','tex');
      if nstages > 1, ylabel(sprintf('stage %d', i)), end
      
      if s.do_filterlearning
        fdims = size(s.f{1});
        nrows = round(sqrt(s.nfilters));
        while mod(s.nfilters,nrows) ~= 0, nrows = nrows - 1; end
        I = cell2mat(reshape(s.f,[],nrows)');
        I = misc.separate_blocks(I,fdims);
        subplot(nstages,3,2+(i-1)*3), imagesc(I), axis image off
        title(sprintf('%d filters of size %dx%d', s.nfilters, fdims(1), fdims(2)))
      end
    end
    drawnow
  end
end

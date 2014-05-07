%SEPARATE_BLOCKS - Enlarge 2D image by inserting a margin between blocks of a given size.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function I = separate_blocks(I,blockdims,pad,pad_val)
  if ~exist('pad','var'),     pad     = 1;   end
  if ~exist('pad_val','var'), pad_val = nan; end

  blockfunc = @(b) padarray(b.data,[pad,pad],pad_val,'pre');
  I = blockproc(I,blockdims,blockfunc);
  I = padarray(I,[pad,pad],pad_val,'post');
end
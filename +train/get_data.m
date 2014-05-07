%GET_DATA - Load and generate training data.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function data = get_data(p,s)

  assert(s.nimages <= 20);
  assert(all([128,128]==s.imdims) && s.npixels == prod(s.imdims), 'size must be 128x128 pixels in this example')

  [X,Y] = deal(zeros(s.npixels,s.nimages));
  if s.is_deblurring
    ks = cell(s.nimages,1);
    alphas = zeros([s.imdims,s.nimages]);
  else
    [ks,alphas] = deal([]);
  end

  for i = 1:s.nimages
    fprintf('Loading image %02d/%02d\r',i,s.nimages);
    img = double(imread(sprintf('%s/images/image_%02d.png',p,i)));
    X(:,i) = img(:);

    if s.is_deblurring
      k = dlmread(sprintf('%s/kernels/image_%02d.dlm',p,i),'\t'); k = k/sum(k(:));
      blurred = conv2(img,k,'valid');
      blurred = blurred + s.sigma * randn(size(blurred));
      blurred = double(uint8(blurred));

      % suboptimal, but simplifies training: pad all kernels with 0s to have the same size (s.k_sz_max)
      kdims = size(k); assert(all(mod(s.k_sz_max-kdims,2)==0))
      k = padarray(k,(s.k_sz_max-kdims)/2,0,'both'); ks{i} = k;
      % crop blurred image to correct for kernel padding
      excess = (size(k)-kdims)/2;
      blurred = blurred(1+excess(1):end-excess(1),1+excess(2):end-excess(2));
      % padarray & edgetaper
      alphas(:,:,i) = train.edgetaper_alpha(k,s.imdims);
      blurred = train.fix_bndry(s.T'*blurred(:),ks{i},alphas(:,:,i),s);
      Y(:,i) = blurred(:);
    else
      img = img + s.sigma * randn(size(img));
      img = double(uint8(img));
      Y(:,i) = s.PT * img(:); % truncate and pad corrupted images
    end    
  end
  fprintf('\n')

  data = struct;
  for str = {'nimages','imdims','npixels','sigma'}, data.(char(str)) = s.(char(str)); end
  [data.X,data.Y,data.ks,data.alphas] = deal(X,Y,ks,alphas);

end
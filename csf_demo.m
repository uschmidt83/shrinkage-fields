%CSF_DEMO - Demonstrate denoising and deconvolution with learned shrinkage field models.
%
%   See also CSF_PREDICT, CSF_VISUALIZE.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function csf_demo

  % load your own learned model like this:
  % name = 'my model'; model = train.to_model('model.mat');
  
  %% load ground truth test image
  x_gt = double(rgb2gray(imread('office_4.jpg')));
  figure, colormap(gray(256))

  %% image denoising (1)
  name = 'csf_3x3'; load(sprintf('models/table1/sigma15/%s.mat',name));
  fprintf('Image denoising (sigma=%g) with %s model (%d stages):\n', model.sigma, name, model.nstages)
  y = x_gt + model.sigma*randn(size(x_gt));
  x = csf_predict(model,y);
  show_results(x_gt,y,x,model,name);
  fprintf('Press any key to continue...\n'); pause; fprintf('\n');

  %% image deconvolution
  name = 'csf_pw'; load('models/table3/csf_pw.mat');
  fprintf('Image deconvolution (sigma=%g) with %s model (%d stages):\n', model.sigma, name, model.nstages)
  k = dlmread('data/kernels/image_08.dlm','\t');
  y = conv2(x_gt,k,'valid');
  b = (size(k)-1)/2; % adjust x_gt to have same size as blurred image
  x_gt = x_gt(1+b(1):end-b(1),1+b(2):end-b(2));
  y = y + model.sigma*randn(size(y));
  x = csf_predict(model,y,k);
  y(1:3*size(k,1),1:3*size(k,2)) = imresize(k*255/max(k(:)),3,'nearest'); % embed blur kernel for visualization
  show_results(x_gt,y,x,model,name);
  fprintf('Press any key to continue...\n'); pause; fprintf('\n');

  %% image denoising (2)
  name = 'csf_7x7'; load(sprintf('models/table2/%s.mat',name));
  fprintf('Image denoising (sigma=%g, 8-bit quantization) with %s model (%d stages):\n', model.sigma, name, model.nstages)
  y = x_gt + model.sigma*randn(size(x_gt));
  y = double(uint8(y)); % 8-bit quantization
  x = csf_predict(model,y);
  show_results(x_gt,y,x,model,name);
  fprintf('Press any key to continue...\n'); pause; fprintf('\n');

  %% show learned model
  name = 'csf_3x3'; load(sprintf('models/table2/%s.mat',name));
  fprintf('Visualization of %s model (%d stages)\n', name, model.nstages)
  close
  csf_visualize(model);

end

function show_results(x_gt,y,x,model,name)
  nstages = numel(x);
  psnrs = cellfun(@(v)psnr(v,x_gt),x);
  subplot(121), imagesc(y,[0,255]), axis image on
  title(sprintf('Input (\\sigma=%g), PSNR = %.2fdB', model.sigma, psnr(y,x_gt)))
  subplot(122), imagesc(x{end},[0,255]), axis image on
  title(sprintf('Output (%s) stage %d, PSNR = %.2fdB', name, nstages, psnrs(end)),'interpreter','none')
end

function f = psnr(x,y)
  mse = mean((x(:)-y(:)).^2);
  f = 20*log10(255) - 10*log10(mse);
end

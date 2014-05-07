%GET_MINIMIZER - Find suitable optimization algorithm.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function minimizer = get_minimizer(s, cost_func, theta0)

  if exist('minFunc') == 2 %#ok
    % 'minFunc' by Mark Schmidt (http://www.di.ens.fr/~mschmidt/Software/minFunc.html)
    options = struct('MaxIter',s.maxoptiters);
    if s.do_joint_training, minimizer = @()  minFunc(cost_func,theta0,options);
    else                    minimizer = @(U) minFunc(cost_func,theta0,options,U); end
  elseif exist('minimize') == 2 %#ok
    % 'minimize' by Carl Edward Rasmussen (http://www.gatsby.ucl.ac.uk/~edward/code/minimize/)
    if s.do_joint_training, minimizer = @()  minimize(theta0,cost_func,s.maxoptiters);
    else                    minimizer = @(U) minimize(theta0,cost_func,s.maxoptiters,U); end
  elseif exist('fminunc') == 2 %#ok
    % 'fminunc' from Matlab's optimization toolbox (http://www.mathworks.de/help/optim/ug/fminunc.html)
    options = optimset('MaxIter',s.maxoptiters, 'Display','iter', 'GradObj','on', 'LargeScale','off');
    if s.do_joint_training, minimizer = @()  fminunc(cost_func,theta0,options);
    else                    minimizer = @(U) fminunc(cost_func,theta0,options,U); end
  else
    error('No suitable minimization algorithm found.')
  end

end
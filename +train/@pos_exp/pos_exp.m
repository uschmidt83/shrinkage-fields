%POS_EXP - Helper functions for positivity constraint: pos <- exp(raw)
%   See also POS_EXPIDENT.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

classdef pos_exp

  methods (Static)
    function pos = from_raw(raw)
      pos = exp(raw);
    end
    function raw = to_raw(pos)
      assert(pos>0)
      raw = log(pos);
    end
    function g = d_raw(pos)
      assert(pos>0)
      g = pos;
    end
  end

end

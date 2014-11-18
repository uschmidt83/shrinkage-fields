%POS_EXPIDENT - Helper functions for positivity constraint: pos <- exp(raw-1) if raw < 1, else pos <- raw
%   Has not been used for experiments in the paper, but often leads to better results than using POS_EXP.
%   See also POS_EXP.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

classdef pos_expident

  methods (Static)
    function pos = from_raw(raw)
      pos = raw;
      if pos < 1
        pos = exp(pos-1);
      end
    end
    function raw = to_raw(pos)
      assert(pos>0)
      raw = pos;
      if raw < 1
        raw = log(raw)+1;
      end
    end
    function g = d_raw(pos)
      assert(pos>0)
      if pos < 1
        g = pos;
      else
        g = 1;
      end
    end
  end

end

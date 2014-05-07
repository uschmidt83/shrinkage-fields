%GET_FILTERS - Get filter bank.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function s = get_filters(s)

  switch s.fbank
    case 'pw'
      s.f = {[1 -1 0];[1 -1 0]'};
    case 'dct'
      n = s.fbank_sz;
      % dct filters without DC component
      assert(mod(n,2)==1, 'filters must be odd-sized')
      N = n^2;
      s.f = cell(N-1,1);
      s.filter_basis = zeros(N,N-1);
      for i = 2:N
        d = zeros(n,n); d(i) = 1;
        s.f{i-1} = idct2(d');
        s.filter_basis(:,i-1) = s.f{i-1}(:);
      end
    otherwise
      error('unsupported fbank: %s', s.fbank)
  end
  s.nfilters = numel(s.f);

end
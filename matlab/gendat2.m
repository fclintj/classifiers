function x = gendat2(class,N)

% The means for the two different classes
%  ┌────────────────────────┐
%  │   Content Functions    │
%  └────────────────────────┘
m0 = [-0.132  0.320 1.672 2.230  1.217 -0.819  3.629  0.8210  1.808 0.1700
      -0.711 -1.726 0.139 1.151 -0.373 -1.573 -0.243 -0.5220 -0.511 0.5330];

randn(2,1)m1 = [-1.169 0.813 -0.859 -0.608 -0.832 2.015 0.173 1.432  0.743 1.0328
      2.065  2.441  0.247  1.806  1.286 0.928 1.923 0.1299 1.847 -0.052];

x = [];
for i=1:N
  idx = floor(rand*10 + 1);
  if(class == 0)
	m = m0(:,idx);
  elseif(class==1)
	m = m1(:,idx);
  end
  x = [x m + randn(2,1)/sqrt(5)];
end
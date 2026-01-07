function [value,x_min,domain] = optim_test_fncs(x, nbr)
% optim_test_fncs
%
% [value,x_min,domain] = optim_test_fncs(x, nbr=1)
%
% Test functions for optimization, see
%   https://en.wikipedia.org/wiki/Test_functions_for_optimization
% Input to the function are
%    x - Evaluation point(s). One value per row in a Npts-by-n matrix.
%  nbr - Which test function to use, see detials.
%
% Return values:
%  value - function value(s) at x as a Npts-by-1 vector. Values outside of
%          domain are set to nan.
%  x_min - minima point for reference, a 1-by-n vector
% domain - Domain of the function as a 2-by-n vector with min and max 
%          values for each x value.
%
% Details:
%  Implements the first 20 test functions for single-objective optimization 
%  on (e.g. excluding the 21st function, the Shekel function):
%   https://en.wikipedia.org/wiki/Test_functions_for_optimization
% The n is the dimension of the function domain (i.e. size of the input x), 
% and nbr is which function to evaluate, according to:
%   1) Rastrigin function
%   2) Ackley function
%   3) Sphere function
%   4) Rosenbrock function
%   5) Beale function
%   6) Goldstein–Price function
%   7) Booth function
%   8) Bukin function N.6
%   9) Matyas function
%  10) Lévi function N.13
%  11) Himmelblau's function
%  12) Three-hump camel function
%  13) Easom function
%  14) Cross-in-tray function
%  15) Eggholder function
%  16) Hölder table function
%  17) McCormick function
%  18) Schaffer function N. 2
%  19) Schaffer function N. 4
%  20) Styblinski–Tang function

if nargin<2 || isempty(nbr), nbr=1; end
n = size(x,2);
%sanity checks
if ~isscalar(nbr) || round(nbr)~=nbr
  error('nbr should be an integer scalar')
end

%function evaluation
switch nbr
  case 1  %   1) Rastrigin function
    A = 10;
    value = A*n + sum(x.^2 - 10*cos(2*pi*x),2);
    domain = repmat([-5.12;5.12],[1 n]);
    x_min = zeros(1,n);

  case 2  %   2) Ackley function
    [x,n] = check_equal_2(x,n,nbr);
    value = -20*exp(-0.2*sqrt(0.5*sum(x.^2,2))) ...
      - exp(0.5*sum(cos(2*pi*x),2)) + exp(1) + 20;
    domain = repmat([-5;5],[1 n]);
    x_min = zeros(1,n);
  
  case 3  %   3) Sphere function
    value = sum(x.^2,2);
    domain = repmat([-Inf;Inf],[1 n]);
    x_min = zeros(1,n);
  
  case 4  %   4) Rosenbrock function
    [x,n] = check_less_2(x,n,nbr);
    value = sum(100*(x(:,2:end)-x(:,1:end-1)).^2 + (1-x(:,1:end-1)).^2,2);
    domain = repmat([-Inf;Inf],[1 n]);
    x_min = ones(1,n);

  case 5  %   5) Beale function
    [x,n] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_1 = x(:,2);
    value = (1.5-x_1+x_1.*y_1).^2 ...
      + (2.25-x_1+x_1.*y_1.^2).^2 ...
      + (2.625-x_1+x_1.*y_1.^3).^2;
    domain = repmat([-4.5;4.5],[1 n]);
    x_min = [3 0.5];

  case 6  %   6) Goldstein–Price function
    [x,n] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_1 = x(:,2);
    value = (1+(x_1+y_1+1).^2 .* ...
      (19-14*x_1+3*x_1.^2-14*y_1+6*x_1.*y_1+3*y_1.^2)) .* ...
      (30 + (2*x_1-3*y_1).^2 .* ...
      (18-32*x_1+12*x_1.^2+48*y_1-36*x_1.*y_1+27*y_1.^2));
    domain = repmat([-2;2],[1 n]);
    x_min = [0 -1];

  case 7  %   7) Booth function
    [x,n] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_1 = x(:,2);
    value = (x_1+2*y_1-7).^2 + (2*x_1+y_1-5).^2;
    domain = repmat([-10;10],[1 n]);
    x_min = [1 3];

  case 8  %   8) Bukin function N.6
    [x,~] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_1 = x(:,2);
    value = 100*sqrt(abs(y_1-0.01*x_1.^2)) + 0.01*abs(x_1+10);
    domain = [-15 -3; -5 3];
    x_min = [-10 1];

  case 9  %   9) Matyas function
    [x,n] = check_equal_2(x,n,nbr);
    value = 0.26.*sum(x.^2,2) - 0.48*prod(x,2);
    domain = repmat([-10;10],[1 n]);
    x_min = zeros(1,n);

  case 10 %  10) Lévi function N.13
    [x,n] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_1 = x(:,2);
    value = sin(3*pi*x_1).^2 + ...
      (x_1-1).^2 .* (1+sin(3*pi*y_1).^2) + ...
      (y_1-1).^2 .* (1+sin(2*pi*y_1).^2);
    domain = repmat([-10;10],[1 n]);
    x_min = [1 1];

  case 11 %  11) Himmelblau's function
    [x,n] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_1 = x(:,2);
    value = (x_1.^2+y_1-11).^2 + (x_1+y_1.^2-7).^2;
    domain = repmat([-5;5],[1 n]);
    x_min = [3 2;
      -2.805118 3.131312;
      -3.779310 -3.283186;
      3.584428 -1.848126];

  case 12 %  12) Three-hump camel function
    [x,n] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_1 = x(:,2);
    value = 2*x_1.^2 - 1.05*x_1.^4 + 1/6*x_1.^6 + x_1.*y_1 + y_1.^2;
    domain = repmat([-5;5],[1 n]);
    x_min = zeros(1,n);

  case 13 %  13) Easom function
    [x,n] = check_equal_2(x,n,nbr);
    value = -prod(cos(x),2).*exp(-sum((x-pi).^2,2));
    domain = repmat([-100;100],[1 n]);
    x_min = pi*ones(1,n);

  case 14 %  14) Cross-in-tray function
    [x,n] = check_equal_2(x,n,nbr);
    d = sqrt(sum(x.^2,2))/pi;
    sin_prod = prod(sin(x),2);
    value = -0.0001*(abs(sin_prod.*exp(abs(100-d)))+1).^0.1;
    domain = repmat([-10;10],[1 n]);
    x0=1.34941;
    x_min = [1 1;-1 1;1 -1;-1 -1]*x0;

  case 15 %  15) Eggholder function
    [x,n] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_47 = x(:,2)+47;
    d1 = sqrt(abs(x_1/2+y_47));
    d2 = sqrt(abs(x_1-y_47));
    value = -y_47.*sin(d1) - x_1.*sin(d2);
    domain = repmat([-512;512],[1 n]);
    x_min = [512 404.2319];

  case 16 %  16) Hölder table function
    [x,n] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_1 = x(:,2);
    d = sqrt(sum(x.^2,2))/pi;
    value = -abs(sin(x_1).*cos(y_1).*exp(abs(1-d)));
    domain = repmat([-10;10],[1 n]);
    x_min = [8.05502 9.66459];
    x_min = [1 1;-1 1;1 -1;-1 -1].*x_min;

  case 17 %  17) McCormick function
    [x,~] = check_equal_2(x,n,nbr);
    x_1 = x(:,1);
    y_1 = x(:,2);
    value = sin(x_1+y_1) + (x_1-y_1).^2 - 1.5*x_1 + 2.5*y_1 + 1;
    domain = [-1.5 -3;4 4];
    x_min = [-0.54719 -1.54719];

  case 18 %  18) Schaffer function N. 2
    [x,n] = check_equal_2(x,n,nbr);
    x_2 = x(:,1).^2;
    y_2 = x(:,2).^2;
    value = 0.5 + (sin(x_2-y_2).^2-0.5)./(1+0.001*(x_2+y_2)).^2;
    domain = repmat([-100;100],[1 n]);
    x_min = zeros(1,n);

  case 19 %  19) Schaffer function N. 4
    [x,n] = check_equal_2(x,n,nbr);
    x_2 = x(:,1).^2;
    y_2 = x(:,2).^2;
    value = 0.5 + (cos(sin(abs(x_2-y_2))).^2-0.5)./(1+0.001*(x_2+y_2)).^2;
    domain = repmat([-100;100],[1 n]);
    x0 = 1.25313;
    x_min = [0 1;0 -1;1 0;-1 0]*x0;

  case 20 %  20) Styblinski–Tang function
    value = 0.5*sum(x.^4-16*x.^2+5*x,2);
    domain = repmat([-5;5],[1 n]);
    x_min = -2.903534*ones(1,n);

  otherwise
    error('Unknwon function, nbr should be 1 to 20')
end

%adjust values outside of domain to nan
I = any(x<domain(1,:) | x>domain(2,:),2);
value(I) = nan;

%several functions require n=2, warn for these
  function [x,n] = check_equal_2(x,n,nbr)
    if n>2
      warning('n=2 required for function %u, truncating input to first two columns', nbr)
      n = 2;
      x = x(:,1:2);
    elseif n==1
      error('n=2 required for function %u, current n=1', nbr)
    end
  end

%several functions require n>=2, warn for these
  function [x,n] = check_less_2(x,n,nbr)
    if n==1
      error('n>=2 required for function %u, current n=1', nbr)
    end
  end
end
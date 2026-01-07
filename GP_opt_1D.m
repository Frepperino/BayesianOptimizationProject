function [xy, par_est]=GP_opt_1D(fnc, x_bnd, n_init, beta, nu, sigma2_eps, n_it, plotF)
% GP_opt_1D Example function for Gaussian Process optimization in 1D
%
% [xy, par_est] = GP_opt_1D(fnc, x_bnd, n_init=10, beta=3, ...
%                           sigma2_eps=1e-3, nu=2.5, n_itt=50, plotF=1)
%
% fnc - Objective function taking one input parameter.
% x_bnd - Limits on x
% n_init - Number of initial points taken as
%          x_init =linspace(x_bnd(1). x_bnd(2), n_init)
% beta - Acquisition parameter
% nu - Fixed nu in the Matern covariance
% n_it - Number of iterations.
% plotF - Plot optimization progress.
%
% Returns:
%  xy - matrix with all evaluations points and function values.
%  par_est - final parameter estimate for the Gaussian Process.

%default values
if nargin<3 || isempty(n_init), n_init=10; end
if nargin<4 || isempty(beta), beta=3; end
if nargin<5 || isempty(nu), nu=2.5; end
if nargin<6 || isempty(sigma2_eps), sigma2_eps=1e-3; end
if nargin<7 || isempty(n_it), n_it=50; end
if nargin<8 || isempty(plotF), plotF=1; end

%n_init initial points on a regular grid
x_init = linspace(x_bnd(1), x_bnd(2), n_init)';

%rough grid for optimization and plotting, ca 1000 points in total
x_grid = linspace(x_bnd(1), x_bnd(2), 1e3)';
if plotF
  %compute value of loss-function on grid for illustration
  f_grid = fnc(x_grid);
end

%compute value of loss-function for initial x-values
xy = zeros(n_init+n_it,2);
j=n_init;
xy(1:n_init,:) = [x_init fnc(x_init)];


%initial parameter estimate
par_est = [];

for k=1:n_it
    %fit GP with fixed nu
    par_est = covest_ml(xy(1:j,:), [0 0 nu sigma2_eps], par_est);

    %reconstruction on grid
    [y_rec,y_var,S_kk,iS_y] = reconstruct(xy(1:j,:), x_grid, par_est);

    %find minimum on grid of loss function
    [~,Imin]=min(y_rec-beta*sqrt(y_var));
    x_min = x_grid(Imin,:);

    %find minima, using grid minima as starting point
    x_min = fminsearch(@(x) acquisition(x,x_bnd,beta,xy(1:j,:),par_est,S_kk,iS_y), x_min);

    %plots
    if plotF
        plot(x_grid, f_grid,'-c', ...
          xy(1:j,1),xy(1:j,end),'*r', ...
          x_grid,y_rec,'-k', ...
          x_grid,y_rec+[-2 2].*sqrt(y_var),':k',...
          x_grid,y_rec-beta*sqrt(y_var),'--r')
        xline(x_min)
        title(k)
        if plotF==1
            pause(0.1)
        else
          pause
        end
    end

    %updated function evaluations
    j = j+1;
    xy(j,:) = [x_min fnc(x_min)];
end

%% Distance matrix given rho
function D=distance_matrix(u,v,rho)
D = abs(u-v')/rho;

%% Matern covariance
function r=matern_covariance(dist,sigma2,nu)
%compute distances with effect from nu
dist_nu = sqrt(2*nu)*dist;

%deal with 0 values
dpos = (dist>0);
r = zeros(size(dist));
r(~dpos) = sigma2;

%% Special cases of nu={0.5, 1.5, and 2.5}
switch nu
  case 0.5
    d = dist(dpos);
    B = sigma2.*exp(-d);
  case 1.5
    d = sqrt(3)*dist(dpos);
    B = sigma2.*(1+d).*exp(-d);
  case 2.5
    d = sqrt(5)*dist(dpos);
    B = sigma2.*(1+d+(d.^2)/3).*exp(-d);
  otherwise
    % Less sensitive to large distances:
    B = log(besselk(nu,dist_nu(dpos)));
    B = exp(log(sigma2)-gammaln(nu)-(nu-1)*log(2) + ...
      nu.*log(dist_nu(dpos)) + B);
    %sanity check, large nu values will need replacement for small dunique
    %values
    if any(isinf(B))
      dist = dist(dpos);
      B(isinf(B)) = sigma2*exp(-0.5*dist(isinf(B)).^2);
    end
end
r(dpos) = B;

%% Maximum Likelihood Estimation
function par_est = covest_ml(xy, par_fixed, par_start)
%initial parameters
if isempty(par_start)
  par_start = zeros(size(par_fixed));
  %rough estimate of field variance
  par_start(2) = var(xy(:,2));
  %and nugget
  par_start(4) = par_start(2)/5;
  %rough estimate of range
  par_start(1) = max(max(distance_matrix(xy(:,1),xy(:,1),1)))/5;
else
  par_start = reshape(par_start,1,[]);
end

%fixed parameters?
if isempty(par_fixed)
  par_fixed = zeros(size(par_start));
end

%define loss function that accounts for fixed parameters
neg_log_like = @(par) covest_ml_loss(xy, exp(par), par_fixed);
%minimize loss function (negative log-likelihood)
par = fminsearch(neg_log_like, log(par_start(par_fixed==0)));

%extract estimated parameters
par_fixed(par_fixed==0) = exp(par);
par_est=par_fixed;


%% loss function for the covariance function
function nlogf = covest_ml_loss(xy, par, par0)
%merge parameters
par0(par0==0) = par;
par=par0;
%extract parameters
rho = par(1);
sigma2 = par(2);
nu = par(3);
sigma2_eps = par(4);
%compute distance matrix
D = distance_matrix(xy(:,1),xy(:,1),rho);
%compute covariance function
Sigma_yy = matern_covariance(D,sigma2,nu) + sigma2_eps*eye(size(D));
%compute Choleskey factor
[R,p] = chol(Sigma_yy);
%Choleskey fail -> return large value
if p~=0
	nlogf = realmax/2;
	return; 
end
nlogf = sum(log(diag(R))) + 0.5*sum((xy(:,2)'/R).^2);

%% Function computing reconstructions
function [y_rec,y_var,S_kk,iS_y] = reconstruct(xy,x0,par,S_kk, iS_y)

if nargin<4 || isempty(S_kk)
    S_kk = distance_matrix(xy(:,1),xy(:,1),par(1));
    S_kk = matern_covariance(S_kk, par(2), par(3));
    S_kk = S_kk + par(4)*eye(size(S_kk));
end
if nargin<5 || isempty(iS_y)
    iS_y = S_kk\xy(:,2);
end

S_uk = distance_matrix(x0,xy(:,1),par(1));
S_uk = matern_covariance(S_uk, par(2), par(3));

y_rec = S_uk * iS_y;
y_var = par(2)+par(4) - sum((S_uk/S_kk).*S_uk,2);
y_var = max(y_var,0);

%% acquisition function, used to determine next point
function l=acquisition(x, x_bnd, beta, xy, par, S_kk, iS_y)
  %check bounds
  if x<x_bnd(1) || x_bnd(2)<x
      l = realmax/2;
      return
  end
  [y_rec,y_var] = reconstruct(xy,x,par,S_kk,iS_y);
  l = y_rec-beta*sqrt(y_var);
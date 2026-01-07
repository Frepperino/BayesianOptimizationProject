function [xyz, par_est]=GP_opt_2D(true_fnc, x_bnd, y_bnd, n_init, beta, nu, sigma2_eps, n_it, plotF)
% GP_opt_2D
%
% [xyz, par_est] = GP_opt_2D(true_fnc, x_bnd, y_bnd, n_init=[5, 5], beta=3, ...
%                           sigma2_eps=1e-3, nu=2.5, n_itt=50, plotF=1)
%
% true_fnc - Objective function taking one input parameter.
% x_bnd - Limits on x
% n_init - Number of initial points taken as
%          x_init =linspace(x_bnd(1). x_bnd(2), n_init)
% beta - Acquisition parameter
% nu - Fixed nu in the Matern covariance
% n_it - Number of iterations.
% plotF - Plot optimization progress.
%
% Returns:
%  xyz - matrix with all evaluations points and function values.
%  par_est - final parameter estimate for the Gaussian Process.

%default values
if nargin<4 || isempty(n_init), n_init=[5, 5]; end
if nargin<5 || isempty(beta), beta=3; end
if nargin<6 || isempty(nu), nu=2.5; end
if nargin<7 || isempty(sigma2_eps), sigma2_eps=1e-3; end
if nargin<8 || isempty(n_it), n_it=50; end
if nargin<9 || isempty(plotF), plotF=1; end

%n_init initial points on a regular grid
x_init = linspace(x_bnd(1), x_bnd(2), n_init(1));
y_init = linspace(y_bnd(1), y_bnd(2), n_init(2));

[X_init, Y_init] = meshgrid(x_init,y_init); 

%rough grid for optimization and plotting, ca 1000 points in total
% S: temporarily reduce
x_grid = linspace(x_bnd(1), x_bnd(2), 1e2)';
y_grid = linspace(y_bnd(1), y_bnd(2), 1e2)';

[X_grid,Y_grid] = meshgrid(x_grid,y_grid); 

if plotF
  %compute value of loss-function on grid for illustration
  f_grid = true_fnc(X_grid, Y_grid);
end

%compute value of loss-function for initial x-values
% S: get total number of inital points to fill up our list of evaluated
% points. xyz is our expansion of xy from the 1D.
total_n_init = n_init(1)*n_init(2);
xyz = zeros(total_n_init+n_it,3);
j=total_n_init;
% S: We column stack all the points from the matrix that true_fnc returns.
f_init = true_fnc(X_init, Y_init)
xyz(1:total_n_init,:) = [X_init(:), Y_init(:) f_init(:)];

% S: Uncomment this to plot the true function and the initial points for
% debugging.
% surf(X_grid, Y_grid, f_grid)
% xlabel("X")
% ylabel("Y")
% hold on
% for i=1:numel(f_init)
%     row = xyz(i,:)
%     plot3(row(1),row(2),row(3),'or', 'MarkerFaceColor', 'r');
% end

%initial parameter estimate
par_est = [];

for k=1:n_it
    %fit GP with fixed nu
    par_est = covest_ml(xyz(1:j,:), 'bomboclat', [0 0 nu sigma2_eps], par_est);

    %reconstruction on grid.
    % S: We column stack the grid to fit into the reconstuct function.
    xy0 = [X_grid(:), Y_grid(:)];
    [y_rec,y_var,S_kk,iS_y] = reconstruct(xyz(1:j,:), xy0, par_est);
    %S: y_rec and y_var are now in column stacked format so we reshape them.
    y_rec = reshape(y_rec, size(X_grid));
    y_var = reshape(y_var, size(X_grid));

    %find minimum on grid of loss function
    % S: Since we are in 2D we now do min over all dimensions.
    [~,Imin]=min(y_rec-beta*sqrt(y_var), [], 'all');
    % S: Imin is a linear index, so we need to do
    [row, col] = ind2sub(size(X_grid), Imin);
    xy_min = [X_grid(row, col), Y_grid(row,col)];

    %find minima, using grid minima as starting point
    xy_min = fminsearch(@(xy) acquisition(xy,x_bnd,y_bnd,beta,xyz(1:j,:),par_est,S_kk,iS_y), xy_min);

    %plots
    if plotF
        xyz(1:j,:) % Printing points so far.
        h_true = surf(X_grid, Y_grid, f_grid, 'FaceColor', 'b','FaceAlpha', 0.5);
        hold on
        h_pts = plot3(xyz(1:j,1),xyz(1:j,2),xyz(1:j,3), '*', 'Color', [255/255, 165/255, 0/255]);
        h_rec = surf(X_grid, Y_grid, y_rec,'FaceColor', 'r');
        zlims = zlim;   % current z-axis limits
        h_latest = plot3([xy_min(1) xy_min(1)], [xy_min(2) xy_min(2)], zlims, '-', 'Color', [255/255, 165/255, 0/255] ,'LineWidth', 2);
        h_acq = surf(X_grid, Y_grid, y_rec - beta*sqrt(y_var), 'FaceColor', 'g', 'FaceAlpha', 0.25);
        h_var = surf(X_grid, Y_grid, y_rec+2.*sqrt(y_var),'FaceColor', 'k','FaceAlpha', 0.25);
        surf(X_grid, Y_grid, y_rec-2.*sqrt(y_var),'FaceColor', 'k','FaceAlpha', 0.25);
        legend([h_true, h_pts, h_rec, h_latest, h_acq, h_var], ...
       {'True function', ...
        'Evaluated points', ...
        'Reconstructed surface', ...
        'Latest observation', ...
        'Acquisition surface', ...
        'Reconstruction variance'})
        xlabel("Dim 1")
        ylabel("Dim 2")
        title(sprintf("Iteration %d", k))
        if plotF==1
            pause(0.1)
        else
          pause
        end
        hold off
    end

    %updated function evaluations
    j = j+1;
    xyz(j,:) = [xy_min(1), xy_min(2) true_fnc(xy_min(1), xy_min(2))];
end

%% Distance matrix given rho
% S: this is made for 2D now. I have not validated.
function D=distance_matrix(p1,p2,rho)
dx = p1(:,1) - p2(:,1).';
dy = p1(:,2) - p2(:,2).';
D = sqrt(dx.^2 + dy.^2)/rho;

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
function par_est = covest_ml(xyz, covf, par_fixed, par_start)

%%%% Not used yet, will be used when expanding to other covariance
%%%% functions.
%covf = validatestring(covf, {'matern', 'cauchy', 'exponential', ...
%	'gaussian', 'spherical'}, 'covest_ml', 'covf', 3);

% S: Since we assume a zero mean gaussian process, s2 is simply the
% variance, without adjusting for any mean.
% s2 = var(xyz(:,3));

%initial parameters
if isempty(par_start)
  par_start = zeros(size(par_fixed));
  %rough estimate of field variance
  par_start(2) = var(xyz(:,3));
  %and nugget
  par_start(4) = par_start(2)/5;
  %rough estimate of range
  % S: distance_matrix is changed. I have not validated.
  par_start(1) = max(max(distance_matrix(xyz(:,1:2),xyz(:,1:2),1)))/5;
else
  par_start = reshape(par_start,1,[]);
end

%fixed parameters?
if isempty(par_fixed)
  par_fixed = zeros(size(par_start));
end

%define loss function that accounts for fixed parameters
neg_log_like = @(par) covest_ml_loss(xyz, exp(par), par_fixed);
%minimize loss function (negative log-likelihood)
par = fminsearch(neg_log_like, log(par_start(par_fixed==0)));

%extract estimated parameters
par_fixed(par_fixed==0) = exp(par);
par_est=par_fixed;

%% Setting up values for the desired covariance function (COMPLETE LATER WHEN EXANDING TO MORE COV FUNCS)
function [covf, par_fixed, par_start] = covest_init(covf, par_fixed, par_start, d_max, s2)

switch covf
	case 'cauchy'
		n_pars = 4;
		covf = @(d,x,I,J) cauchy_covariance(d,x(1),x(2),x(3),I,J);
		range_par = @(range, x) range/sqrt(10^(1/x)-1);
	case 'exponential'
		n_pars = 3;
		covf = @(d,x,I,J) exponential_covariance(d,x(1),x(2),I,J);
		range_par = @(range) 2/range;

	case 'gaussian'
		n_pars = 3;
		covf = @(d,x,I,J) gaussian_covariance(d,x(1),x(2),I,J);
		range_par = @(range) range;
	case 'matern'
		n_pars = 4;
		covf = @(d,x,I,J) matern_covariance(d,x(1),x(2),x(3),I,J);
		range_par = @(range, nu) sqrt(8*nu)/range;
	case 'spherical'
		n_pars = 3;
		covf = @(d,x,I,J) spherical_covariance(d,x(1),x(2),I,J);
		range_par = @(range) range/0.73;
	otherwise
		error('Unknown covariance function.')
end

% default par_fixed if not specified
if isempty(par_fixed), par_fixed=zeros(n_pars,1); end

%ensure that par_fixed is column vector
par_fixed = par_fixed(:);

%check size of par_fixed
if length(par_fixed)~=n_pars, error('par0 should be of length %d', n_pars); end

%default for par_start
if isempty(par_start)
	par_start = zeros(size(par_fixed));
	%rough estimate of field variance
    % change from s2 here
	par_start(2) = s2;
	%nugget should be one fifth of total variance
	par_start(end) = par_start(1)/5;
	%range should be about half of max distance
	if n_pars==3 %(exp, gaus, spher)
		par_start(1) = range_par(d_max/2);
	else %(matern, cauchy) first set defautl for nu,x
		if par_start(3)==0, par_start(3)=1; end
		par_start(1) = range_par(d_max/2, par_start(3));
	end
end
%ensure that par_start is column vector
par_start = par_start(:);
%check size of par_start
if length(par_start)~=n_pars
	error('par_start should be of length %d', n_pars);
end

%% loss function for the covariance function
function nlogf = covest_ml_loss(xyz, par, par0)
%merge parameters
par0(par0==0) = par;
par=par0;
%extract parameters
rho = par(1);
sigma2 = par(2);
nu = par(3);
sigma2_eps = par(4);
%compute distance matrix
D = distance_matrix(xyz(:,1:2),xyz(:,1:2),rho);
%compute covariance function
Sigma_yy = matern_covariance(D,sigma2,nu) + sigma2_eps*eye(size(D));
%compute Choleskey factor
[R,p] = chol(Sigma_yy);
%Choleskey fail -> return large value
if p~=0
	nlogf = realmax/2;
	return; 
end
nlogf = sum(log(diag(R))) + 0.5*sum((xyz(:,3)'/R).^2);

%% Function computing reconstructions
%S: changed x0 to two be a Mx2 list of the positions, and changed xy to xyz
function [y_rec,y_var,S_kk,iS_y] = reconstruct(xyz,xy0,par,S_kk, iS_y)

if nargin<4 || isempty(S_kk)
    S_kk = distance_matrix(xyz(:,1:2),xyz(:,1:2),par(1));
    S_kk = matern_covariance(S_kk, par(2), par(3));
    S_kk = S_kk + par(4)*eye(size(S_kk));
end
if nargin<5 || isempty(iS_y)
    iS_y = S_kk\xyz(:,3);
end

S_uk = distance_matrix(xy0,xyz(:,1:2),par(1));
S_uk = matern_covariance(S_uk, par(2), par(3));

y_rec = S_uk * iS_y;
y_var = par(2)+par(4) - sum((S_uk/S_kk).*S_uk,2);
y_var = max(y_var,0);

% S: remember the y_rec and y_var are column stacked at this point. Which
% doesnt matter for the acquistion function below because it only calls for
% a scalar xy0 anyway.

%% acquisition function, used to determine next point
function l=acquisition(xy, x_bnd, y_bnd, beta, xyz, par, S_kk, iS_y)
  % ADD LOGIC FOR y_bnd
  %check bounds
  if xy(1)<x_bnd(1) || x_bnd(2)<xy(1) || xy(2)<y_bnd(1) || y_bnd(2)<xy(2)
      l = realmax/2;
      return
  end
  [y_rec,y_var] = reconstruct(xyz,xy,par,S_kk,iS_y);
  l = y_rec-beta*sqrt(y_var);
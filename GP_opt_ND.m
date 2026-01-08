function [vals, par_est]=GP_opt_ND(true_fnc, n_params, bnds, init_points, grid_resolution, beta, nu, sigma2_eps, n_it, plotF)
% GP_opt_ND
%
% [vals, par_est] = GP_opt_ND(true_fnc, params, bnds, n_init, grid_resolution=1e1, beta=3, ...
%                           sigma2_eps=1e-3, nu=2.5, n_itt=50, plotF=1)
%
% true_fnc - Objective function taking one input parameter.
% n_params - Number of params/dimensions
% bnds - n_params x 2 matrix of each dimensions bounds. I.e. bnds(k,1) is
% the lower bound of dimension k and bnds(k,2) is the upper.
% init_points - N x n_params matrix of initial points.
% grid_resolution - Resolution of the optimization grid
%          axes{i} = linspace(bnds(i,1), bnds(i,2), grid_resolution);
% beta - Acquisition parameter
% nu - Fixed nu in the Matern covariance
% n_it - Number of iterations.
% plotF - Plot optimization progress.
%
% Returns:
%  vals - (n_init^n_params) x (n_params) matrix with all evaluations points and function values.
%  par_est - final parameter estimate for the Gaussian Process.

%default values
if nargin<5 || isempty(grid_resolution), grid_resolution=1e1; end
if nargin<6 || isempty(beta), beta=3; end
if nargin<7 || isempty(nu), nu=2.5; end
if nargin<8 || isempty(sigma2_eps), sigma2_eps=1e-3; end
if nargin<9 || isempty(n_it), n_it=50; end
if nargin<10 || isempty(plotF), plotF=1; end

%rough grid for optimization and plotting
% S: we store each dimensions grid in a row each.
axes = cell(1, n_params);
for i=1:n_params
  axes{i} = linspace(bnds(i,1), bnds(i,2), grid_resolution);
end
gr = cell(1,n_params);
[gr{:}] = ndgrid(axes{:});
grids = zeros(numel(gr{1}), n_params);
for i = 1:n_params
    grids(:,i) = gr{i}(:);
end


%compute value of loss-function for initial x-values
% S: vals is our ND extension of xy from the 1D script. we store all params
% + the value in each row.
n_init_points = size(init_points,1);
vals = zeros(n_init_points+n_it,n_params+1);
f_init = true_fnc(init_points);
vals(1:n_init_points,:) = [init_points(:,:), f_init];
j=n_init_points;

%initial parameter estimate
par_est = [];

if plotF
    if n_params==1 || n_params==2
        f_grid = true_fnc(grids);
        figSurface = figure;
    end

    figPaths = figure;
    title("Parameter paths")
    ylabel("Value")
    % S: I think it might just be cleaner to not plot the first values, or plot
    % them on the same xline maybe?
    xlabel("Point sampled (NOTE THIS IS ONLY ITERATION AFTER n_init_points)")
    hold on
    for i=1:n_params
        plot(1:j,vals(1:j,i), 'o', 'MarkerSize', 5);
    end
    hold off
    
    figBest = figure;
    title("Best function value")
    ylabel("Value")
    xlabel("Point sampled (NOTE THIS IS ONLY ITERATION AFTER n_init_points)")
    plot(1:j,vals(1:j,end), 'o', 'MarkerSize', 5);
end

for k=1:n_it
    %fit GP with fixed nu
    par_est = covest_ml(vals(1:j,:), 'bomboclat', [0 0 nu sigma2_eps], par_est);

    %reconstruction on grid.
    % S: grids is defined as (n_params) x (points), which is the transpose
    % of how we want it in reconstruct.
    [y_rec,y_var,S_kk,iS_y] = reconstruct(vals(1:j,:), grids, par_est);

    %find minimum on grid of loss function
    [~,Imin]=min(y_rec-beta*sqrt(y_var));
    param_min = grids(Imin, 1:end);

    %find minima, using grid minima as starting point
    param_min = fminsearch(@(param) acquisition(param,bnds,beta,vals(1:j,:),par_est,S_kk,iS_y), param_min);

    %plots
    if plotF
        vals(1:j,:) % Printing points so far.

        figure(figBest);
        hold on
        plot(n_init_points:j,cummin(vals(n_init_points:j,end)));
        hold off

        figure(figPaths);
        hold on
        for i=1:n_params
            plot(1:j,vals(1:j,i), 'MarkerSize', 5);
        end
        hold off
        
        if n_params==1
            plot(grids(:,1), f_grid,'-c', ...
            vals(1:j,1),vals(1:j,end),'*r', ...
            grids(:,1),y_rec,'-k', ...
            grids(:,1),y_rec+[-2 2].*sqrt(y_var),':k',...
            grids(:,1),y_rec-beta*sqrt(y_var),'--r')
            xline(param_min)
            title(k)
        elseif n_params==2
            figure(figSurface);
            size(grids)
            size(f_grid)
            X_grid = reshape(grids(:,1), [grid_resolution, grid_resolution]);
            Y_grid = reshape(grids(:,2), [grid_resolution, grid_resolution]);
            size(X_grid)
            f_grid = reshape(f_grid, size(X_grid));
            h_true = surf(X_grid, Y_grid, f_grid, 'FaceColor', 'b','FaceAlpha', 0.5);
            hold on
            h_pts = plot3(vals(1:j,1),vals(1:j,2),vals(1:j,3), '*', 'Color', [255/255, 165/255, 0/255]);
            y_recs = reshape(y_rec, size(X_grid));
            y_vars = reshape(y_var, size(X_grid));
            h_rec = surf(X_grid, Y_grid, y_recs,'FaceColor', 'r');
            zlims = zlim;   % current z-axis limits
            h_latest = plot3([param_min(1) param_min(1)], [param_min(2) param_min(2)], zlims, '-', 'Color', [255/255, 165/255, 0/255] ,'LineWidth', 2);
            h_acq = surf(X_grid, Y_grid, y_recs - beta*sqrt(y_vars), 'FaceColor', 'g', 'FaceAlpha', 0.25);
            h_var = surf(X_grid, Y_grid, y_recs+2.*sqrt(y_vars),'FaceColor', 'k','FaceAlpha', 0.25);
            surf(X_grid, Y_grid, y_recs-2.*sqrt(y_vars),'FaceColor', 'k','FaceAlpha', 0.25);
            legend([h_true, h_pts, h_rec, h_latest, h_acq, h_var], ...
           {'True function', ...
            'Evaluated points', ...
            'Reconstructed surface', ...
            'Latest observation', ...
            'Acquisition surface', ...
            'Reconstruction variance'})
            xlabel("Dim 1")
            ylabel("Dim 2")
        end

        if plotF==1
            pause(0.1)
        else
          pause
        end
        hold off
    end

    %updated function evaluations
    j = j+1;
    vals(j,:) = [param_min true_fnc(param_min)];
end

%% Distance matrix given rho
% S: remade for all ND. ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
% S: not yet validated
function D=distance_matrix(p1,p2,rho)
p1s = p1 ./ rho;
p2s = p2 ./ rho;

D2 = sum(p1s.^2, 2) + sum(p2s.^2, 2).' - 2*(p1s*p2s.');
D  = sqrt(max(D2,0)); % Just for safety in case negative term gives floating point negative.

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
function par_est = covest_ml(vals, covf, par_fixed, par_start)

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
  par_start(2) = var(vals(:,end));
  %and nugget
  par_start(4) = par_start(2)/5;
  %rough estimate of range
  % S: distance_matrix is changed but it is still only 2D by pairwise
  % points.
  par_start(1) = max(max(distance_matrix(vals(:,1:end-1),vals(:,1:end-1),1)))/5;
else
  par_start = reshape(par_start,1,[]);
end

%fixed parameters?
if isempty(par_fixed)
  par_fixed = zeros(size(par_start));
end

%define loss function that accounts for fixed parameters
neg_log_like = @(par) covest_ml_loss(vals, exp(par), par_fixed);
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
function nlogf = covest_ml_loss(vals, par, par0)
%merge parameters
par0(par0==0) = par;
par=par0;
%extract parameters
rho = par(1);
sigma2 = par(2);
nu = par(3);
sigma2_eps = par(4);
%compute distance matrix
D = distance_matrix(vals(:,1:end-1),vals(:,1:end-1),rho);
%compute covariance function
Sigma_yy = matern_covariance(D,sigma2,nu) + sigma2_eps*eye(size(D));
%compute Choleskey factor
[R,p] = chol(Sigma_yy);
%Choleskey fail -> return large value
if p~=0
	nlogf = realmax/2;
	return; 
end
nlogf = sum(log(diag(R))) + 0.5*sum((vals(:,end)'/R).^2);

%% Function computing reconstructions
%S: changed x0 to grid, and xy to vals.
function [y_rec,y_var,S_kk,iS_y] = reconstruct(vals,grid,par,S_kk, iS_y)

if nargin<4 || isempty(S_kk)
    S_kk = distance_matrix(vals(:,1:end-1),vals(:,1:end-1),par(1));
    S_kk = matern_covariance(S_kk, par(2), par(3));
    S_kk = S_kk + par(4)*eye(size(S_kk));
end
if nargin<5 || isempty(iS_y)
    iS_y = S_kk\vals(:,end);
end

S_uk = distance_matrix(grid,vals(:,1:end-1),par(1));
S_uk = matern_covariance(S_uk, par(2), par(3));

y_rec = S_uk * iS_y;
y_var = par(2)+par(4) - sum((S_uk/S_kk).*S_uk,2);
y_var = max(y_var,0);

%% acquisition function, used to determine next point
function l=acquisition(param, bnds, beta, vals, par, S_kk, iS_y)
  %check bounds
  % S: if any parameter is less than its lower bound or higher bound. The
  % transpose is because of bnds row corresponds to each column in param.
  if any(param < bnds(:,1).') || any(param > bnds(:,2).')
      l = realmax/2;
      return
  end
  [y_rec,y_var] = reconstruct(vals,param,par,S_kk,iS_y);
  l = y_rec-beta*sqrt(y_var);
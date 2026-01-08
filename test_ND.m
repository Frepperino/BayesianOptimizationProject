clear
%% Same as test_2D script, to compare
function val = true_func(param_vals)
val = sin(0.5*param_vals(:,1)) .* cos(0.5*param_vals(:,2));
end

n_params=2;
bnds = [[-5, 5]; [-6, 6]];
init_points = [[-5, 6]; [5, 6]];

GP_opt_ND(@true_func, n_params,bnds, init_points, 1e2, [], [], [], [], 2);

%% Same as example_1D_opt.m, to compare
fnc = @(t)  sin(2*t)+.1*t.^2;
n_params = 1;
bnds = [[-5, 5]];
init_points = [[-5]; [0]; [5]];

grid_resolution=1e3;
%run Bayesian optimzation on this function
%[vals, par_est] = GP_opt_ND(fnc, n_params, bnds, init_points, grid_resolution, [], [], [], [], 2);

%run Bayesian optimzation on this function - larger beta
[vals, par_est] = GP_opt_ND(fnc, n_params, bnds, init_points, grid_resolution, 10, [], [], [], 2);


%% Otherwise / sandbox:

n_params=3;
bnds = [[-5, 5]; [-6, 6]; [-3, 3]];
init_points = [[-5, 6, 1]; [5, 6, 0]];

GP_opt_ND(@true_func, n_params,bnds, init_points, 1e2, [], [], [], [], 2);
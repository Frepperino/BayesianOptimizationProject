clear

function Z = true_func(X, Y)
Z = sin(0.5*X) .* cos(0.5*Y);
end

grid_resolution=1e2;

[xyz, par_est] = GP_opt_2D(@true_func, [-5, 5], [-6, 6], [2, 1], grid_resolution, [], [], [], [], 2);
xyz




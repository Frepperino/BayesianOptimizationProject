clc
clear

function Z = true_func(X, Y)
Z = sin(0.5*X) .* cos(0.5*Y);
end

[xyz, par_est] = GP_opt_2D(@true_func, [-5, 5], [-6, 6], [3, 5], [], [], [], [], 2);
xyz



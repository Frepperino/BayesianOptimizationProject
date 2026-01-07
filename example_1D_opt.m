%define a non linear function with multiple minima
fnc = @(t)  sin(2*t)+.1*t.^2;

%run Bayesian optimzation on this function
[xy, par_est] = GP_opt_1D(fnc, [-5 5], 3, [], [], [], [], 2);

%run Bayesian optimzation on this function - larger beta
[xy, par_est] = GP_opt_1D(fnc, [-5 5], 3, 10, [], [], [], 2);

% Define the function
f = @(x) sqrt(abs(25*x.^2 - 1));

% Define x values for dense plotting and evaluation
x_vals = linspace(-1, 1, 1000);
f_vals = f(x_vals);

% Number of points for interpolation
n_values = [6, 11, 16];

% Initialize the figure for the function and its interpolations
figure;
hold on;
plot(x_vals, f_vals, 'k-', 'LineWidth', 2, 'DisplayName', 'Original Function');

% Initialize figure for error plots
figure_error = figure;
hold on;

% Compute and plot interpolation and error for each n
for n = n_values
    % Evenly spaced interpolation points
    x_inter = linspace(-1, 1, n);
    y_inter = f(x_inter);

    % Interpolation using built-in function (linear interpolation)
    s = interp1(x_inter, y_inter, x_vals, 'linear');

    % Plot interpolation
    figure(1); % Switch to the function plot
    plot(x_vals, s, 'DisplayName', ['Interpolation n=', num2str(n)]);

    % Calculate and plot error
    error_vals = abs(f_vals - s);
    figure(2); % Switch to the error plot
    plot(x_vals, error_vals, 'DisplayName', ['Error n=', num2str(n)]);
end

% Configure the plot for the function and its interpolations
figure(1);
xlabel('x');
ylabel('f(x) / s(x)');
title('Function and Its Linear Interpolations');
legend show;
hold off;

% Configure the plot for the error
figure(2);
xlabel('x');
ylabel('Error |f(x) - s(x)|');
title('Error between the function and interpolations');
legend show;
hold off;
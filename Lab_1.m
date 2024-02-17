yalmip('clear');
clc;

a = sdpvar(6,1);
b = sdpvar(6,1);
c = sdpvar(6,1);

% polynomial:
p = @(x) a(1) + a(2)*x + a(3)*x.^2 + a(4)*x.^3 + a(5)*x.^4 + a(6)*x.^5;
p1 = @(x) b(1) + b(2)*x + b(3)*x.^2 + b(4)*x.^3 + b(5)*x.^4 + b(6)*x.^5;
p2 = @(x) c(1) + c(2)*x + c(3)*x.^2 + c(4)*x.^3 + c(5)*x.^4 + c(6)*x.^5;

x = linspace(0, 10, 101);

Constraints = [];
Constraints_1 = [];
Constraints_2 = [];
for i = 1:length(x)
    Constraints = [Constraints, 0 <= p(x(i)) <= 5];
    Constraints_1 = [Constraints_1, 0 <= p1(x(i)) <= 5];
    Constraints_2 = [Constraints_2, 0 <= p2(x(i)) <= 5];
end
Constraints = [Constraints, p(0) <= 1, p(3) >= 4, p(7) <= 1, 2 <= p(10) <= 3];
Constraints_1 = [Constraints_1, p1(0) <= 1, p1(3) >= 4, p1(7) <= 1, 2 <= p1(10) <= 3];
Constraints_2 = [Constraints_2, p2(0) <= 1, p2(3) >= 4, p2(7) <= 1, 2 <= p2(10) <= 3];

Objective = a(6)^2 + a(5)^2 + a(4)^2;
Objective_1 = b(6)^2;
Objective_2 = c(6)^2 + c(5)^2 + c(4)^2 + c(3)^2 + c(2)^2 + c(1)^2;

% solver options
options = sdpsettings('solver','sdpt3');

% optimization problem solver
sol = optimize(Constraints,Objective,options)
sol_1 = optimize(Constraints_1,Objective_1,options)
sol_2 = optimize(Constraints_2,Objective_2,options)
%disp(estimated_coefficients);

y = value(p(x));
y1 = value(p1(x));
y2 = value(p2(x));

% initialize noisy values
z = zeros(size(x));
v = zeros(size(x));
w = zeros(size(x));
noise_variance = 0.3;

%---------------------------------------------------%
%         Comment one of this noise scheme !
%---------------------------------------------------%

%Noise 1 (Gaussian noise):
for i = 1:length(x)
   w = sqrt(noise_variance) * randn; 
   z(i) = y(i) + w; 
end

%---------------------------------------------------%
% Noise 2:
% for i = 1:length(x)
%     if rand() < 0.1
%         % Генерация значений в диапазоне [-20, -10] U [10, 20]
%        v(i) = 10 + 10 * rand();
%        if rand() < 0.5
%            v(i) = -v(i);
%        end
%     end
%     w = sqrt(noise_variance) * randn; 
%     z(i) = y(i) + w + v(i);
%     
% end
%---------------------------------------------------%

%plot_1
figure;
plot(x, y2, 'g', 'LineWidth', 2); 
hold on;
plot(x, y, 'b', 'LineWidth', 2); 
hold on;
plot(x, y1, 'r', 'LineWidth', 2); 
grid on;
xlabel('x');
ylabel('Value');
legend('p1: Obj = a(6)^2 + a(5)^2 + a(4)^2 + a(3)^2 + a(2)^2 + a(1)^2 ','p2: Obj = a(6)^2 + a(5)^2 + a(4)^2', 'p3: Obj = a(6)^2', 'Location','southwest');
title('p1 - p2 - p3');

%plot_2
figure;
plot(x, y, 'b', 'LineWidth', 2);
hold on;
plot(x, z, 'r', 'LineWidth', 2); % noize mc
grid on;
xlabel('x');
ylabel('Value');
legend('Original Polynomial', 'Noisy Polynomial');
title('y = p(x)');

%1 LEAST SQUARE METHOD (LSM)  - got the best result 
coeffs_lsm = polyfit(x,z,5);
y_lsm = polyval(coeffs_lsm, x);

%2 Chebyshev aproximation (CHEB)
init_coeffs_cheb = zeros(1, 6);
cheb_error_func = @(coeffs) max(abs(polyval(coeffs, x) - z_values));
coeffs_cheb = fminsearch(cheb_error_func, init_coeffs_cheb);
y_cheb = polyval(coeffs_cheb, x);

%3 Minimizing the sum of error modules (Least abs err LAE)
error_func = @(coeffs) sum(abs(polyval(coeffs, x) - z));
init_coeffs_LAE = zeros(1, 6);
coeffs_LAE = fminsearch(error_func, init_coeffs_LAE);
y_lae = polyval(coeffs_LAE, x);

%4 Min the sum of penalty function: phi(t) = sqrt(abs(t)) - PF
penalty_func = @(coeffs) sum(abs(polyval(coeffs, x) - z).^0.5);
init_coeffs_PF = zeros(1, 6);
coeffs_PF = fminsearch(penalty_func, init_coeffs_PF);
y_pf = polyval(coeffs_PF, x);

%plot_3
figure;
plot(x, y, 'k', 'LineWidth', 2); 
hold on;

plot(x, y_lsm, 'g', 'LineWidth', 2);
plot(x, y_cheb, 'm', 'LineWidth', 2); 
plot(x, y_lae, 'c', 'LineWidth', 2); 
plot(x, y_pf , 'r', 'LineWidth', 2); 
%plot(x, z, 'k--', 'LineWidth', 1); 

legend('Original', 'Least Square Method', 'Chebyshev aproximation', 'Least Abs Error', 'Penalty Function', 'Noise');
grid on;
xlabel('x');
ylabel('Values');
title('comparison of methods');








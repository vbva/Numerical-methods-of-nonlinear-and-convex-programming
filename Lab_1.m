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
   w(i) = sqrt(noise_variance) * randn(); 
   z(i) = y2(i) + w(i); 
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
%     z(i) = y2(i) + w + v(i);
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
plot(x, y2, 'b', 'LineWidth', 2);
hold on;
plot(x, z, 'r', 'LineWidth', 2); % noize mc
grid on;
xlabel('x');
ylabel('Value');
legend('Original Polynomial', 'Noisy Polynomial');
title('y = p(x)');

%1 LEAST SQUARE METHOD (LSM)  - got the best result 
d = sdpvar(6,1);
p_1 = d(1) + d(2)*x + d(3)*x.^2 + d(4)*x.^3 + d(5)*x.^4 + d(6)*x.^5;

residuals_lsm = z - p_1;
Objective_lsm = norm(residuals_lsm, 2);
optimize(Constraints, Objective_lsm);
y_lsm = value(p_1);
error_lsm = max(abs(y_lsm - y2))

%2 Chebyshev aproximation (CHEB)
d_2 = sdpvar(6,1);
p_2 = d_2(1) + d_2(2)*x + d_2(3)*x.^2 + d_2(4)*x.^3 + d_2(5)*x.^4 + d_2(6)*x.^5;

residuals_cheb = z - p_2;
Objective_cheb = norm(residuals_cheb, inf);
optimize(Constraints, Objective_cheb);
y_cheb = value(p_2);
error_cheb = max(abs(y_cheb-y2))

%3 Minimizing the sum of error modules (Least abs err LAE)
d_3 = sdpvar(6,1);
p_3 = d_3(1) + d_3(2)*x + d_3(3)*x.^2 + d_3(4)*x.^3 + d_3(5)*x.^4 + d_3(6)*x.^5;

residuals_lae = z - p_3;
Objective_lae = norm(residuals_lae, 1);
optimize(Constraints, Objective_lae);
y_lae = value(p_3);
error_lae = max(abs(y_lae-y2))

%4 Min the sum of penalty function: phi(t) = sqrt(abs(t))
X  = value(c)    
a_0 = 1.1 * X;
opts = optimset('TolX', 1e-16,'MaxFunEvals', 10000, 'MaxIter', 10000, 'TolFun', 1e-16);
[x_pen, f_val] = fminunc(@poly_value, a_0, opts);
A5 = x_pen(6); B5 = x_pen(5); C5 = x_pen(4); D5 = x_pen(3);E5 = x_pen(2);F5 = x_pen(1);
f4 = @(x)A5 * x.^5 + B5 * x.^4 + C5 * x.^3 + D5 * x.^2 + E5 * x + F5;
error_pf = max(abs(f4(x)-y2))
%plot_3
figure;
plot(x, y2, 'k', 'LineWidth', 2); 
hold on;

plot(x, y_lsm, 'g', 'LineWidth', 2);
plot(x, y_cheb, 'm', 'LineWidth', 2); 
plot(x, y_lae, 'c', 'LineWidth', 2); 
plot(x, f4(x) , 'r', 'LineWidth', 2); 
plot(x, z, 'k--', 'LineWidth', 1); 

legend('Original', 'Least Square Method', 'Chebyshev aproximation', 'Least Abs Error', 'Penalty Function', 'Noise');
grid on;
xlabel('x');
ylabel('Values');
title('comparison of methods');




function func_value = poly_value(a)
    global z;
    A = [];
    t_vec = @(tt) [ tt^0; tt^1; tt^2; tt^3; tt^4; tt^5 ];
    n = length(z);

    for i = 0 : 1 : n-1
        A = [ A; abs(t_vec(0.1 * i)' * a' - z(i+1))^(0.5) ];
    end
    func_value = sum(A);
end





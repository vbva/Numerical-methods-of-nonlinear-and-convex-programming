% Построить два набора случайных точек на плоскости, используя
% параметрическую формулу:
% x = x_c + a*r*cos(phi)*cos(alpha) - b*r*sin(phi)*sin(alpha)
% y = y_c + a*r*cos(phi)*sin(alpha) + b*r*sin(phi)*cos(alpha)

% phi = [0, 2pi], r = [0,1] - random
% #1: a = 3, b = 1, alpha = 0, x_c = -2, y_c = 1
% #2: a = 4, b = 1, alpha = pi/4, x_c = 2, y_c = -1
% num of points (N): 500

%  -    task 1   -
% a1 = 3;
% a2 = 4;
% b1 = 1;
% b2 = 1;

%  -    task 2   -
a1 = 3;
a2 = 4;
b1 = 2.5;
b2 = 2;
%  ---------------
N = 500;
alpha1 = 0;
alpha2 = pi/4;
xc1 = -2;
xc2 = 2;
yc1 = 1;
yc2 = -1;
r = rand(N, 1); % in the interval (0,1)
phi = 2*pi*rand(N, 1);

% #1 points set :
x1 = xc1 + a1.*r.*cos(phi).*cos(alpha1) - b1.*r.*sin(phi).*sin(alpha1);
y1 = yc1 + a1.*r.*sin(alpha1).*cos(phi) + b1.*r.*sin(phi).*cos(alpha1);

% #2 points set :
x2 = xc2 + a2.*r.*cos(phi).*cos(alpha2) - b2.*r.*sin(phi).*sin(alpha2);
y2 = yc2 + a2.*r.*cos(phi).*sin(alpha2) + b2.*r.*sin(phi).*cos(alpha2);

X = [x1; x2]; % array
Y = [y1; y2];
data = [X,Y]; % matrix

%Support Vector Machine (SVM)

labels = [ones(N, 1); -ones(N, 1)];

%define optimization task:
w = sdpvar(2, 1); % weight vector (orientation of hyperplane)
b = sdpvar(1);    % bias

%decision function: f(x) = x * w + b 
%optimal hyperplane: x * w + b == 0
% - define in which side og hyperplane a new data point lies
% -  +/- f(x) - indicator of predicted class of x

constraints = [labels.*(data*w + b) >= 1]; 
%if label = +1, then f(x) >= +1
%if label = -1, then f(x) <= -1

objective = 0.5*norm(w)^2; % minimize norm(w) ~ maximizing margin classes
options = sdpsettings('solver', 'sdpt3');
optimize(constraints, objective, options);

w_opt = value(w);
b_opt = value(b);

% draw the sets:
scatter(x1, y1, 'b.');
hold on;
scatter(x2, y2, 'r.');
x_line = linspace(min(X), max(X), 100);
y_line = (-w_opt(1)*x_line - b_opt)/w_opt(2); %w_opt(1)*x + w_opt(2)*y + b_opt = 0
plot(x_line, y_line, '--k', 'LineWidth', 1);
title('sets of points');
legend('set 1', 'set 2', 'dividing line');
xlabel('X');
ylabel('Y');
grid on;
hold off;









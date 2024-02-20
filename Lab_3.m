load('calib.mat');
Q = calib.Q;
T_real = calib.T;

%init_par =[d1, a1, alpha1, 
                % d2, a2, alpha2,
                % d3, a3, alpha3,
                % q1, q2, q3]
                  
init_par = [300, 0, pi/2, ...
                  0, 400, 0, ...
                  0, 500, 0, ...
                  0, 0, 0];
              
%boundaries: angle +- 0.01rad, lenght +- 10mm
lb = [290, -0.01, pi/2-0.01, -10, 390, -0.01, -10, 490, -0.01, -10, -0.01, -0.01];
ub = [310, 0.01, pi/2+0.01, 10, 410, 0.01, 10, 510, 0.01, 10, 0.01, 0.01];

options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

[optimized_params, fval] = fmincon(@(params) obj_func(params, Q, T_real),...
                                    init_par,[], [], [], [], lb, ub, [], options);

disp('Optimal parameters:');
disp(optimized_params);

%Параметры DH (по паспорту):
%_____________________
% theta1 = 0.2268;
% theta2 = 2.3512;
% theta3 = 2.2005;
% 
% d1 = 300;
% d2 = 0;
% d3 = 0;
% 
% alpha1 = pi/2;
% alpha2 = 0;
% alpha3 = 0;
% 
% a1 = 0;
% a2 = 400;
% a3 = 500;
%_____________________
%Прямая кинематики:
% T_kinematic = T_dh(theta1, d1, alpha1, a1) *  ... 
%     T_dh(theta2, d2, alpha2, a2) * ...
%     T_dh(theta3, d3, alpha3, a3);
% 
% d = T_kinematic(1:3, 4);
% R = T_kinematic(1:3, 1:3);

%----------------------------------------------------------------%
function total_diff = obj_func(params, Q, T_real)
    total_diff = 0;
    d1 = params(1);
    a1 = params(2);
    alpha1 = params(3);

    d2 = params(4);
    a2 = params(5);
    alpha2 = params(6);

    d3 = params(7);
    a3 = params(8);
    alpha3 = params(9);
    
    for i = 1:size(Q, 1)
        T_kinematic = T_dh(Q(i, 1), d1, alpha1, a1) *  ... 
                      T_dh(Q(i, 2), d2, alpha2, a2) * ...
                      T_dh(Q(i, 3), d3, alpha3, a3);
        %d = T_kinematic(1:3, 4);
        %R = T_kinematic(1:3, 1:3);
                  
        diff = norm(T_kinematic - T_real(:, :, i), 'fro');
        total_diff = total_diff + diff;
    end
end
%R_z(theta) - матрица поворота на угол theta отн-но оси Z
%T_z(d) - матрица смещения на величину d вдоль оси Z
function Rz = rot_z(theta) 
    Rz = [cos(theta) -sin(theta) 0 0;
          sin(theta) cos(theta) 0 0;
          0 0 1 0;
          0 0 0 1];
end

function Tz = trans_z(d)
    Tz = [1 0 0 0;
          0 1 0 0;
          0 0 1 d;
          0 0 0 1];
end

function Rx = rot_x(alpha)
    Rx = [1 0 0 0;
          0 cos(alpha) -sin(alpha) 0;
          0 sin(alpha) cos(alpha) 0;
          0 0 0 1];
end

function Tx = trans_x(a)
    Tx = [1 0 0 a;
          0 1 0 0;
          0 0 1 0;
          0 0 0 1];
end

%Матрциа  представления Денавита-Хартенберга (T_dh)
function T = T_dh(theta, d, alpha, a)
    T = rot_z(theta)*trans_z(d)*rot_x(alpha)*trans_x(a);
end



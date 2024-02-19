%handwritten digit recognition
%dataset: MNIST

load('mnist.mat');

num0 = 6;
num1 = 4;

ind0_train = find(training.labels == num0);
ind1_train = find(training.labels == num1);
X_train = reshape(training.images(:,:,ind0_train), [], numel(ind0_train))';
X_train = [X_train; reshape(training.images(:,:,ind1_train), [], numel(ind1_train))'];
Y_train = [ones(numel(ind0_train), 1); -ones(numel(ind1_train), 1)];

% test dataset
ind0_test = find(test.labels == num0);
ind1_test = find(test.labels == num1);

half0_test = floor(length(ind0_test) / 2);
half1_test = floor(length(ind1_test) / 2);

% additional dataset
X_train_add = [reshape(test.images(:,:,ind0_test(1:half0_test)), [], half0_test)'; 
               reshape(test.images(:,:,ind1_test(1:half1_test)), [], half1_test)'];
Y_train_add = [ones(half0_test, 1); -ones(half1_test, 1)];

% total dataset
X_train_full = [X_train; X_train_add];
Y_train_full = [Y_train; Y_train_add];

% SVM
w = sdpvar(size(X_train_full, 2), 1);
b = sdpvar(1);
constraints = [Y_train_full.*(X_train_full * w + b) >= 1];
objective = 0.5*norm(w)^2;
options = sdpsettings('solver', 'sdpt3');
optimize(constraints, objective, options);
w_opt = value(w);
b_opt = value(b);


X_test = [reshape(test.images(:,:,ind0_test(half0_test+1:end)), [], numel(ind0_test(half0_test+1:end)))';
          reshape(test.images(:,:,ind1_test(half1_test+1:end)), [], numel(ind1_test(half1_test+1:end)))'];
Y_test = [ones(numel(ind0_test(half0_test+1:end)), 1); -ones(numel(ind1_test(half1_test+1:end)), 1)];


predictions = sign(X_test * w_opt + b_opt);

% Error:
accuracy = sum(predictions == Y_test) / numel(Y_test);
disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);

% Vizualization:
numImagesToShow = 10;
figure;
colormap gray;

for i = 1:numImagesToShow
    % 1-st number:
    subplot(2, numImagesToShow, i);
    img = test.images(:,:,ind0_test(half0_test + i));
    imshow(img);
    predLabel = predictions(i);

    if predLabel == 1
        title(sprintf('actual: %d\npredicted: %d', num0, num0));
    else
        title(sprintf('actual: %d\npredicted: error', num0, num0));
    end
    
    % 2-nd number:
    subplot(2, numImagesToShow, numImagesToShow + i);
    img = test.images(:,:,ind1_test(half1_test + i));
    imshow(img);
    predLabel = predictions(half0_test + i);

    if predLabel == -1
        title(sprintf('actual: %d\npredicted: %d', num1, num1));
    else
        title(sprintf('actual: %d\npredicted: error', num1, num1));
    end
    
end

% disp('Коэффициенты гиперплоскости (w_opt):');
% disp(w_opt);
% 
% disp('Смещение гиперплоскости (b_opt):');
% disp(b_opt);



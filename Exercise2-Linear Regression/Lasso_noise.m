function [] = Lasso_noise(X, y, X_test, y_test)


%This script is designed for Exercise 2 Question 3

[sz1,~] = size(X);
extra_one = ones(sz1,1);
X = [X extra_one];
irrel_col = randn([sz1 1000]);
X = [X irrel_col];

[sz3,~] = size(X_test);
extra_one_test = ones([sz3 1]);
irrel_col = randn([sz3 1000]);
X_test = [X_test irrel_col];
X_test = [X_test extra_one_test];
X_test = X_test .* 1/2;

[~,sz2] = size(X);

k = 11;
mean_square_error_train = zeros(1,k);
mean_square_error_valid = zeros(1,k);
mean_square_error_test = zeros(1,k);
non_zero_precent_train = zeros(1,k);
non_zero_precent_valid = zeros(1,k);

for i=1:k
    lambda_star = i * 10;
    lambda_matrix = lambda_star * ones(sz2,sz2);
    w = (X.'*X+lambda_matrix)\(X.'*y);
    
    mean_square_error_train(i) = (norm(X*w - y)^2)/sz1;
    non_zero_precent_train(i) = nnz(w)*1.0/sz2;
    for j=1:10
        X_valid_train = [X(1:(j-1)*30,:); X(j*30+1:300,:)];
        X_valid_test = X((j-1)*30+1:j*30,:);
        y_valid_train = [y(1:(j-1)*30,:); y(j*30+1:300,:)];
        y_valid_test = y((j-1)*30+1:j*30,:);
        
        w_valid = (X_valid_train.'*X_valid_train+lambda_matrix)\(X_valid_train.'*y_valid_train);
        mean_square_error_valid(i) = mean_square_error_valid(i) + (norm(X_valid_test*w_valid - y_valid_test)^2)/30;
        non_zero_precent_valid(i) = non_zero_precent_valid(i) + nnz(w_valid)*1.0/sz2;
    end
    
    mean_square_error_valid(i) = mean_square_error_valid(i)/10;
    non_zero_precent_valid(i) = non_zero_precent_valid(i)/10;
   
    mean_square_error_test(i) = ((norm(X_test*w - y_test))^2)/sz1;
end

all_lambda = [0;10;20;30;40;50;60;70;80;90;100];
res = table(all_lambda,mean_square_error_train.',mean_square_error_valid.',mean_square_error_test.');
res.Properties.VariableNames = {'Lambda' 'Training_MSE' 'Valid_MSE' 'Test_MSE' };
res

w = w(15:1014,1);
p = nnz(w)*1.0/1000;
p



%getTheDataset();

%X= dlmread("dataset.csv");
%size(X);
%y=dlmread("winner.csv");
%size(X);

%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 10;  % 20x20 Input Images of Digits
hidden_layer_size = 10;   % 25 hidden units
num_labels = 2;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)







%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

%initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
%initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_Theta1=dlmread("savedTheta1.csv");
initial_Theta2=dlmread("savedTheta2.csv");

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
%checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
%checkNNGradients(lambda);
fprintf( "Backpropagation done!" );


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 1);

%  You should also try different values of lambda
lambda = 0.5;

% Create "short hand" for the cost function to be minimized

X=dlmread("dataset.csv");
y=dlmread("winner.csv");
size(X);
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);



% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
nn_params=initial_nn_params;
for i =1:5

    [nn_params, cost] = fmincg(costFunction, nn_params, options);
end;

fprintf("Done with the cost!" );

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

S_id=11;
Toss_win=1;
Toss_decision=0;

Innings=1;
Ovr=19;
ball=1;
runs=6;
wicket=0;

q=[1 2 S_id Toss_win Toss_decision Innings Ovr ball runs wicket];

csvwrite("savedTheta1.csv",Theta1);
csvwrite("savedTheta2.csv",Theta2);
size(Theta1);

[pred,percent] = predict(Theta1, Theta2, X,q);
pred;
percent;
%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf("\nWin Prediction!\nTeam 1:      %f\nTeam 2:      %f\n",percent*100,(1-percent)*100);

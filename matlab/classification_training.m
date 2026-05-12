clc; clear;

% Load the clean signal data from different files
load('clean_10mm_9features.mat');  % clean_10mm
load('clean_18mm_9features.mat');  % clean_18mm
load('clean_20mm_9features.mat');  % clean_20mm
load('clean_25mm_9features.mat');  % clean_25mm

% Data preprocessing - Normalize features to [0,1] range
clean_10mm = normalize(clean_10mm', 'range');
clean_18mm = normalize(clean_18mm', 'range');
clean_20mm = normalize(clean_20mm', 'range');
clean_25mm = normalize(clean_25mm', 'range');

% Combine all clean signals into one dataset
clean_signals_combined = [clean_10mm; clean_18mm; clean_20mm; clean_25mm];

% Get number of samples for each class
num_samples_10mm = size(clean_10mm, 1);
num_samples_18mm = size(clean_18mm, 1);
num_samples_20mm = size(clean_20mm, 1);
num_samples_25mm = size(clean_25mm, 1);

% Display dataset information
fprintf('Dataset dimensions:\n');
fprintf('10mm samples: %d\n', num_samples_10mm);
fprintf('18mm samples: %d\n', num_samples_18mm);
fprintf('20mm samples: %d\n', num_samples_20mm);
fprintf('25mm samples: %d\n', num_samples_25mm);

% Create target labels (1 for 10mm, 2 for 18mm, 3 for 20mm, 4 for 25mm)
labels_combined = [ones(num_samples_10mm, 1);
                  2 * ones(num_samples_18mm, 1);
                  3 * ones(num_samples_20mm, 1);
                  4 * ones(num_samples_25mm, 1)];

% Convert to one-hot encoding for neural network
num_classes = 4;
targets = zeros(num_classes, length(labels_combined));
for i = 1:length(labels_combined)
    targets(labels_combined(i), i) = 1;
end

% K-fold cross validation parameters
K = 4;
cv = cvpartition(length(labels_combined), 'KFold', K);

% Initialize array to store models and accuracies
saved_models = cell(K, 1);
accuracy_clean = zeros(K, 1);

% Store training indices for each fold
fold_indices = struct('train', cell(1, K), 'test', cell(1, K));

for fold = 1:K
    fprintf('Training Fold %d/%d\n', fold, K);

    % Get train and test indices for the current fold
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    % Store indices for later use in testing
    fold_indices(fold).train = find(trainIdx);
    fold_indices(fold).test = find(testIdx);

    % Prepare data for this fold
    train_data = clean_signals_combined(trainIdx, :)';
    train_targets = targets(:, trainIdx);

    % Create a Pattern Recognition Network
    hiddenLayerSize = 15;
    net = patternnet(hiddenLayerSize);

    % Configure network parameters
    net.trainFcn = 'trainscg';
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:size(train_data, 2);  % Use all training data
    net.divideParam.valInd = [];  % No validation set
    net.divideParam.testInd = []; % No test set during training

    % Train the Network
    [net, tr] = train(net, train_data, train_targets);

    % Test on validation set
    test_data = clean_signals_combined(testIdx, :)';
    test_targets = targets(:, testIdx);

    yTest = net(test_data);

    % Convert predictions to indices
    [~, predictionTest] = max(yTest);
    [~, actualTest] = max(test_targets);

    % Calculate accuracy for the test set
    accuracy_clean(fold) = sum(predictionTest == actualTest) / numel(actualTest);

    % Save the model for this fold
    saved_models{fold} = net;

    % Display results for current fold
    fprintf('Accuracy for Fold %d: %.2f%%\n', fold, accuracy_clean(fold) * 100);

    % Display confusion matrix for current fold
    C = confusionmat(actualTest, predictionTest);
    fprintf('Confusion Matrix for Fold %d:\n', fold);
    disp(C);
end

% Calculate and display average accuracy and standard deviation
mean_accuracy = mean(accuracy_clean);
std_accuracy = std(accuracy_clean);
fprintf('Training Results:\n');
fprintf('Average Test Accuracy: %.2f%% (±%.2f%%)\n', mean_accuracy*100, std_accuracy*100);
fprintf('Standard Deviation of Accuracies: %.2f%%\n', std_accuracy * 100);



% Save all models and related information
final_save = struct();
final_save.models = saved_models;
final_save.accuracy = accuracy_clean;
final_save.mean_accuracy = mean_accuracy;
final_save.std_accuracy = std_accuracy;
final_save.fold_indices = fold_indices;
final_save.feature_normalization = 'range';  % Save normalization method used
final_save.input_size = size(clean_signals_combined, 2);
save('trained_models.mat', '-struct', 'final_save');


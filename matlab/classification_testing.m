%% 4 types
% Load the trained models and data
load('trained_models.mat');

% Load the clean signal data
load('clean_10mm_9features.mat');
load('clean_18mm_9features.mat');
load('clean_20mm_9features.mat');
load('clean_25mm_9features.mat');

% Load noisy and denoised data (adjust filenames as needed)
load('noisy_minus10dB_10mm_9features.mat');
load('noisy_minus10dB_18mm_9features.mat');
load('noisy_minus10dB_20mm_9features.mat');
load('noisy_minus10dB_25mm_9features.mat');

load('savgol_wavelet_minus10dB_10mm_9features.mat');
load('savgol_wavelet_minus10dB_18mm_9features.mat');
load('savgol_wavelet_minus10dB_20mm_9features.mat');
load('savgol_wavelet_minus10dB_25mm_9features.mat');

% Normalize and combine clean data
clean_10mm = normalize(clean_10mm', 'range');
clean_18mm = normalize(clean_18mm', 'range');
clean_20mm = normalize(clean_20mm', 'range');
clean_25mm = normalize(clean_25mm', 'range');
savgol_wavelet_combined = [clean_10mm; clean_18mm; clean_20mm; clean_25mm];

% Normalize and combine noisy data
noisy_10mm = normalize(noisy_minus10dB_10mm', 'range');
noisy_18mm = normalize(noisy_minus10dB_18mm', 'range');
noisy_20mm = normalize(noisy_minus10dB_20mm', 'range');
noisy_25mm = normalize(noisy_minus10dB_25mm', 'range');
noisy_signals_combined = [noisy_10mm; noisy_18mm; noisy_20mm; noisy_25mm];

% % Normalize and combine denoised data
% denoised_10mm = normalize(savgol_wavelet_minus10dB_10mm', 'range');
% denoised_18mm = normalize(savgol_wavelet_minus10dB_18mm', 'range');
% denoised_20mm = normalize(savgol_wavelet_minus10dB_20mm', 'range');
% denoised_25mm = normalize(savgol_wavelet_minus10dB_25mm', 'range');
% denoised_signals_combined = [denoised_10mm; denoised_18mm; denoised_20mm; denoised_25mm];

% Normalize and combine denoised data
denoised_10mm = normalize(savgol_wavelet_minus10dB_10mm', 'range');
denoised_18mm = normalize(savgol_wavelet_minus10dB_18mm', 'range');
denoised_20mm = normalize(savgol_wavelet_minus10dB_20mm', 'range');
denoised_25mm = normalize(savgol_wavelet_minus10dB_25mm', 'range');
denoised_signals_combined = [denoised_10mm; denoised_18mm; denoised_20mm; denoised_25mm];

% Create labels
num_samples_10mm = size(clean_10mm, 1);
num_samples_18mm = size(clean_18mm, 1);
num_samples_20mm = size(clean_20mm, 1);
num_samples_25mm = size(clean_25mm, 1);

labels_combined = [ones(num_samples_10mm, 1);
                  2 * ones(num_samples_18mm, 1);
                  3 * ones(num_samples_20mm, 1);
                  4 * ones(num_samples_25mm, 1)];

% Initialize arrays for storing results
K = length(models);
accuracy_values_clean = zeros(1, K);
accuracy_values_noisy = zeros(1, K);
accuracy_values_denoised = zeros(1, K);

% Initialize aggregated predictions and actual values
aggregated_pred_clean = [];
aggregated_actual = [];
aggregated_pred_noisy = [];
aggregated_pred_denoised = [];

for fold = 1:K
    fprintf('\nTesting Fold %d/%d\n', fold, K);

    % Get the model and indices for current fold
    net = models{fold};
    testIdx = fold_indices(fold).test;

    % Test on clean data
    test_data_clean = savgol_wavelet_combined(testIdx, :)';
    test_labels = labels_combined(testIdx);

    yTest_clean = net(test_data_clean);
    [~, pred_clean] = max(yTest_clean);

    % Test on noisy data
    test_data_noisy = noisy_signals_combined(testIdx, :)';
    yTest_noisy = net(test_data_noisy);
    [~, pred_noisy] = max(yTest_noisy);

    % Test on denoised data 
    test_data_denoised = denoised_signals_combined(testIdx, :)';
    yTest_denoised = net(test_data_denoised);
    [~, pred_denoised] = max(yTest_denoised);

    % Calculate accuracies
    accuracy_clean = sum(pred_clean == test_labels') / numel(test_labels) * 100;
    accuracy_noisy = sum(pred_noisy == test_labels') / numel(test_labels) * 100;
    accuracy_denoised = sum(pred_denoised == test_labels') / numel(test_labels) * 100;

    % Store accuracies
    accuracy_values_clean(fold) = accuracy_clean/100;
    accuracy_values_noisy(fold) = accuracy_noisy/100;
    accuracy_values_denoised(fold) = accuracy_denoised/100;

    % Display results for current fold - Clean Data
    fprintf('Accuracy for Fold %d: %.2f%%\n', fold, accuracy_clean);
    fprintf('Confusion Matrix for Fold %d:\n', fold);
    conf_mat_clean = confusionmat(test_labels, pred_clean);
    disp(conf_mat_clean);
    fprintf('\n');

    % Display results for Noisy Data
    fprintf('Noisy Data - Fold %d/%d\n', fold, K);
    fprintf('Accuracy for Fold %d: %.2f%%\n', fold, accuracy_noisy);
    fprintf('Confusion Matrix for Fold %d:\n', fold);
    conf_mat_noisy = confusionmat(test_labels, pred_noisy);
    disp(conf_mat_noisy);
    fprintf('\n');

    % Display results for Denoised Data
    fprintf('Denoised Data - Fold %d/%d\n', fold, K);
    fprintf('Accuracy for Fold %d: %.2f%%\n', fold, accuracy_denoised);
    fprintf('Confusion Matrix for Fold %d:\n', fold);
    conf_mat_denoised = confusionmat(test_labels, pred_denoised);
    disp(conf_mat_denoised);
    fprintf('\n');

    % Aggregate predictions and actual values
    aggregated_pred_clean = [aggregated_pred_clean, pred_clean];
    aggregated_actual = [aggregated_actual, test_labels'];
    aggregated_pred_noisy = [aggregated_pred_noisy, pred_noisy];
    aggregated_pred_denoised = [aggregated_pred_denoised, pred_denoised];
end

% Display final test results
fprintf('Test Results:\n');
fprintf('Average Test Accuracy (Clean): %.2f%% (±%.2f%%)\n', ...
    mean(accuracy_values_clean)*100, std(accuracy_values_clean)*100);
fprintf('Standard Deviation of Accuracies: %.2f%%\n\n', std(accuracy_values_clean)*100);

fprintf('Average Test Accuracy (Noisy): %.2f%% (±%.2f%%)\n', ...
    mean(accuracy_values_noisy)*100, std(accuracy_values_noisy)*100);
fprintf('Standard Deviation of Accuracies: %.2f%%\n\n', std(accuracy_values_noisy)*100);

fprintf('Average Test Accuracy (Denoised): %.2f%% (±%.2f%%)\n', ...
    mean(accuracy_values_denoised)*100, std(accuracy_values_denoised)*100);
fprintf('Standard Deviation of Accuracies: %.2f%%\n', std(accuracy_values_denoised)*100);

% Create and plot overall confusion matrices
figure('Position', [100, 100, 400, 400]);

conf_matrix_clean = confusionmat(aggregated_actual, aggregated_pred_clean);
confusionchart(conf_matrix_clean);
title('Confusion Matrix - Clean Data');

subplot(1,3,2);
conf_matrix_noisy = confusionmat(aggregated_actual, aggregated_pred_noisy);
confusionchart(conf_matrix_noisy);
title('Confusion Matrix - Noisy Data');

subplot(1,3,3);
conf_matrix_denoised = confusionmat(aggregated_actual, aggregated_pred_denoised);
confusionchart(conf_matrix_denoised);
title('Confusion Matrix - Denoised Data');

%% One types of signal
% clc; clear;
% 
% % Load the trained models and data
% load('trained_models.mat');
% 
% % Load noisy data
% load('cnn_ABCDEF_minus10dB_10mm_9features.mat');
% load('cnn_ABCDEF_minus10dB_18mm_9features.mat');
% load('cnn_ABCDEF_minus10dB_20mm_9features.mat');
% load('cnn_ABCDEF_minus10dB_25mm_9features.mat');
% 
% % Normalize and combine noisy data
% cnn_ABCDEF_10mm = normalize(cnn_ABCDEF_minus10dB_10mm', 'range');
% cnn_ABCDEF_18mm = normalize(cnn_ABCDEF_minus10dB_18mm', 'range');
% cnn_ABCDEF_20mm = normalize(cnn_ABCDEF_minus10dB_20mm', 'range');
% cnn_ABCDEF_25mm = normalize(cnn_ABCDEF_minus10dB_25mm', 'range');
% cnn_ABCDEF_signals_combined = [cnn_ABCDEF_10mm; cnn_ABCDEF_18mm; cnn_ABCDEF_20mm; cnn_ABCDEF_25mm];
% 
% % Create labels
% num_samples_10mm = size(cnn_ABCDEF_10mm, 1);
% num_samples_18mm = size(cnn_ABCDEF_18mm, 1);
% num_samples_20mm = size(cnn_ABCDEF_20mm, 1);
% num_samples_25mm = size(cnn_ABCDEF_25mm, 1);
% 
% labels_combined = [ones(num_samples_10mm, 1);
%                   2 * ones(num_samples_18mm, 1);
%                   3 * ones(num_samples_20mm, 1);
%                   4 * ones(num_samples_25mm, 1)];
% 
% % Initialize arrays for storing results
% K = length(models);
% accuracy_values = zeros(1, K);
% 
% % Initialize aggregated predictions and actual values
% aggregated_pred = [];
% aggregated_actual = [];
% 
% for fold = 1:K
%     fprintf('Testing Fold %d/%d\n', fold, K);
% 
%     % Get the model and indices for current fold
%     net = models{fold};
%     testIdx = fold_indices(fold).test;
% 
%     % Test on cnn_ABCDEF data
%     test_data = cnn_ABCDEF_signals_combined(testIdx, :)';
%     test_labels = labels_combined(testIdx);
% 
%     yTest = net(test_data);
%     [~, pred] = max(yTest);
% 
%     % Calculate accuracy
%     accuracy = sum(pred == test_labels') / numel(test_labels) * 100;
% 
%     % Store accuracy
%     accuracy_values(fold) = accuracy/100;
% 
%     % Display results for current fold
%     fprintf('Accuracy for Fold %d: %.2f%%\n', fold, accuracy);
%     fprintf('Confusion Matrix for Fold %d:\n', fold);
%     conf_mat = confusionmat(test_labels, pred);
%     disp(conf_mat);
%     fprintf('\n');
% 
%     % Aggregate predictions and actual values
%     aggregated_pred = [aggregated_pred, pred];
%     aggregated_actual = [aggregated_actual, test_labels'];
% end
% 
% % Display final test results
% fprintf('Test Results:\n');
% fprintf('Average Test Accuracy: %.2f%% (±%.2f%%)\n', ...
%     mean(accuracy_values)*100, std(accuracy_values)*100);
% fprintf('Standard Deviation of Accuracies: %.2f%%\n', std(accuracy_values)*100);
% 
% % Create and plot overall confusion matrix
% figure('Position', [100, 100, 400, 400]);
% conf_matrix = confusionmat(aggregated_actual, aggregated_pred);
% confusionchart(conf_matrix);
% title('Confusion Matrix - Denoised Data (CNN-Wavelet)');
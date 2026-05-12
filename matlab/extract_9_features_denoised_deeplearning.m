%% Extracted Feature (One type, one noise level)
% clc; clear;
% 
% %% Load noisy signals
% load('C:\Users\cyx02\Downloads\Documents\FYP\Classification\noisy_minus10dB_25mm.mat');  % matrix: [200 x 2001]
% 
% %% Load trained CNN model
% load('C:\Users\cyx02\Downloads\Documents\FYP\Classification\cnn_wavelet.mat');  % model: net
% 
% num_samples = size(noisy_data_minus10dB_25mm, 1);  % should be 200
% feature_matrix = zeros(9, num_samples);  % 9 features x 200 samples
% 
% for i = 1:num_samples
%     % Extract one signal and preprocess
%     signal = noisy_data_minus10dB_25mm(i, :).';  % column vector
% 
%     % Ensure length is exactly 2001
%     desired_length = 2001;
%     if length(signal) < desired_length
%         signal(end+1:desired_length) = 0;
%     elseif length(signal) > desired_length
%         signal = signal(1:desired_length);
%     end
% 
%     % Normalize if required
%     signal = normalize(signal);
% 
%     % Reshape to match CNN input
%     input_signal = reshape(signal, [2001, 1, 1]);
% 
%     % Denoise using CNN
%     denoised_data = predict(net, input_signal);
%     denoised_data = denoised_data';  % now [1 x 2001]
% 
%     % Feature extraction
%     RMS_denoised = rms(denoised_data);
%     VAR_denoised = var(denoised_data);
%     SD_denoised = std(denoised_data);
%     WF_denoised = RMS_denoised / mean(denoised_data);
%     K_denoised = kurtosis(denoised_data);
%     SK_denoised = skewness(denoised_data);
%     MAX_PSP_denoised = max(pspectrum(denoised_data));
% 
%     % MDF and MNF: Split into 4 segments
%     split_cols = floor(size(denoised_data, 2) / 4);
%     d1 = denoised_data(:, 1:split_cols);
%     d2 = denoised_data(:, split_cols+1:2*split_cols);
%     d3 = denoised_data(:, 2*split_cols+1:3*split_cols);
%     d4 = denoised_data(:, 3*split_cols+1:end);
% 
%     MDF = mean([medfreq(d1), medfreq(d2), medfreq(d3), medfreq(d4)]);
%     MNF = mean([meanfreq(d1), meanfreq(d2), meanfreq(d3), meanfreq(d4)]);
% 
%     % Combine into feature vector
%     feature_vector = [RMS_denoised; VAR_denoised; SD_denoised; WF_denoised; ...
%                       K_denoised; SK_denoised; MAX_PSP_denoised; MDF; MNF];
% 
%     % Store in feature matrix
%     cnn_wavelet_minus10dB_25mm(:, i) = feature_vector;
% end
% 
% %% Save to .mat file
% save('C:\Users\cyx02\Downloads\Documents\FYP\Classification\cnn_wavelet_minus10dB_25mm_9features.mat', ...
%      'cnn_wavelet_minus10dB_25mm');
% fprintf('DONE EXTRACTED');
% 
%% Extracted Feature (4 type tgt, one noise level)

% clc; clear;
% 
% %% Define file paths and names
% base_path = 'C:\Users\cyx02\Downloads\Documents\FYP\Classification\';
% 
% % Input files
% input_files = {
%     'noisy_minus10dB_10mm.mat', 
%     'noisy_minus10dB_18mm.mat', 
%     'noisy_minus10dB_20mm.mat', 
%     'noisy_minus10dB_25mm.mat'
% };
% 
% % Variable names in each .mat file (assuming they follow the same pattern)
% variable_names = {
%     'noisy_data_minus10dB_10mm',
%     'noisy_data_minus10dB_18mm', 
%     'noisy_data_minus10dB_20mm',
%     'noisy_data_minus10dB_25mm'
% };
% 
% % Output files
% output_files = {
%     'wavelet_cnn_ABCDEF_minus10dB_10mm_9features.mat',
%     'wavelet_cnn_ABCDEF_minus10dB_18mm_9features.mat',
%     'wavelet_cnn_ABCDEF_minus10dB_20mm_9features.mat',
%     'wavelet_cnn_ABCDEF_minus10dB_25mm_9features.mat'
% };
% 
% % Output variable names
% output_variable_names = {
%     'wavelet_cnn_ABCDEF_minus10dB_10mm',
%     'wavelet_cnn_ABCDEF_minus10dB_18mm',
%     'wavelet_cnn_ABCDEF_minus10dB_20mm',
%     'wavelet_cnn_ABCDEF_minus10dB_25mm'
% };
% 
% %% Load trained CNN model
% load([base_path 'wavelet_cnn_ABCDEF.mat']); % model: net
% 
% %% Process each file
% for file_idx = 1:length(input_files)
%     fprintf('Processing file %d/%d: %s\n', file_idx, length(input_files), input_files{file_idx});
% 
%     % Load noisy signals
%     load([base_path input_files{file_idx}]); % This will load the variable with the corresponding name
% 
%     % Get the data using dynamic field referencing
%     eval(['noisy_data = ' variable_names{file_idx} ';']);
% 
%     num_samples = size(noisy_data, 1); % should be 200
%     feature_matrix = zeros(9, num_samples); % 9 features x 200 samples
% 
%     for i = 1:num_samples
%         % Extract one signal and preprocess
%         signal = noisy_data(i, :).'; % column vector
% 
%         % Ensure length is exactly 2001
%         desired_length = 2001;
%         if length(signal) < desired_length
%             signal(end+1:desired_length) = 0;
%         elseif length(signal) > desired_length
%             signal = signal(1:desired_length);
%         end
% 
%         % Normalize if required
%         signal = normalize(signal);
% 
%         % Reshape to match CNN input
%         input_signal = reshape(signal, [2001, 1, 1]);
% 
%         % Denoise using CNN
%         denoised_data = predict(net, input_signal);
%         denoised_data = denoised_data'; % now [1 x 2001]
% 
%         % Feature extraction
%         RMS_denoised = rms(denoised_data);
%         VAR_denoised = var(denoised_data);
%         SD_denoised = std(denoised_data);
%         WF_denoised = RMS_denoised / mean(denoised_data);
%         K_denoised = kurtosis(denoised_data);
%         SK_denoised = skewness(denoised_data);
%         MAX_PSP_denoised = max(pspectrum(denoised_data));
% 
%         % MDF and MNF: Split into 4 segments
%         split_cols = floor(size(denoised_data, 2) / 4);
%         d1 = denoised_data(:, 1:split_cols);
%         d2 = denoised_data(:, split_cols+1:2*split_cols);
%         d3 = denoised_data(:, 2*split_cols+1:3*split_cols);
%         d4 = denoised_data(:, 3*split_cols+1:end);
% 
%         MDF = mean([medfreq(d1), medfreq(d2), medfreq(d3), medfreq(d4)]);
%         MNF = mean([meanfreq(d1), meanfreq(d2), meanfreq(d3), meanfreq(d4)]);
% 
%         % Combine into feature vector
%         feature_vector = [RMS_denoised; VAR_denoised; SD_denoised; WF_denoised; ...
%                          K_denoised; SK_denoised; MAX_PSP_denoised; MDF; MNF];
% 
%         % Store in feature matrix
%         feature_matrix(:, i) = feature_vector;
%     end
% 
%     % Save to .mat file with dynamic variable naming
%     eval([output_variable_names{file_idx} ' = feature_matrix;']);
%     save([base_path output_files{file_idx}], output_variable_names{file_idx});
% 
%     fprintf('Completed processing: %s -> %s\n', input_files{file_idx}, output_files{file_idx});
% end
% 
% fprintf('ALL FILES EXTRACTED SUCCESSFULLY!\n');
clear all;close all; clc;
%% Wavelet-CNN (A, B, C, D, E, F) 
% % Type A: Sparse PD pulses (4 pulses with gaps)
% % Type B: Spike-dense signal (realistic sharp pulses with gaps)
% % Type C: 10mm (very sparse PD events)
% % Type D: 18mm (sparse PD events)
% % Type E: 20mm (moderate-high frequency PD events)
% % Type F: 25mm (high frequency, complex PD events)
% 
% %% 1. Signal Simulation with Types A, B, C, D, E, F
% fs = 1000e6;                  % 1 GHz sampling
% t_total = 2e-6;               % 2 μs duration
% t = 0:1/fs:t_total;           % Time vector
% signal_length = length(t);
% num_samples = 3000;           % Increased for 6 types (200 per type)
% 
% clean_signals = zeros(num_samples, signal_length);
% noisy_signals = zeros(num_samples, signal_length);
% 
% for i = 1:num_samples
%     % Cycle through Type A, Type B, Type C, Type D, Type E, Type F
%     signal_type = mod(i-1, 6) + 1;  % Types 1,2,3,4,5,6 correspond to A,B,C,D,E,F
% 
%     if signal_type == 1  % Type A: Sparse PD pulses (4 pulses with gaps)
%         clean_signal = zeros(size(t));
%         start_times = [0.2e-6, 0.6e-6, 1.2e-6, 1.6e-6]; % Clear time gaps
% 
%         for k = 1:length(start_times)
%             A = 10 + rand()*10;
%             fc = 25e6 + rand()*10e6; % Higher freq helps sharpen
%             tau = 0.01e-6 + rand()*0.03e-6; % Very short pulse
%             pulse_t = t - start_times(k);
%             pulse_t = pulse_t(pulse_t >= 0);
% 
%             % Generate a short pulse with fewer points
%             pulse_duration = 0.05e-6; % 50 ns duration
%             pulse_t = pulse_t(pulse_t <= pulse_duration);
%             pulse = A * exp(-pulse_t/tau) .* sin(2*pi*fc*pulse_t);
% 
%             % Insert pulse at the correct position
%             start_idx = find(t >= start_times(k), 1);
%             pulse_len = length(pulse);
%             if start_idx + pulse_len - 1 <= length(clean_signal)
%                 clean_signal(start_idx:start_idx + pulse_len - 1) = pulse;
%             end
%         end
% 
%     elseif signal_type == 2  % Type B: Spike-dense signal (realistic sharp pulses with gaps)
%         clean_signal = zeros(size(t));
%         num_spikes = 20 + randi(10); % Fewer, more realistic pulses
%         spike_len = 2; % Length of each biphasic spike
% 
%         for s = 1:num_spikes
%             start_idx = randi([1, signal_length - spike_len]);
%             amp = 0.5 + 0.5*rand(); % Amplitude
%             direction = (-1)^randi([0 1]); % Flip polarity randomly
% 
%             % Biphasic spike: [positive, negative] or [negative, positive]
%             clean_signal(start_idx) = direction * amp;
%             clean_signal(start_idx + 1) = -direction * amp;
%         end
% 
%     elseif signal_type == 3  % Type C: 10mm
%         clean_signal = zeros(size(t));
% 
%         % Very sparse PD events with random locations
%         num_events = 5 + randi(5);  % 5-10 events (very sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Clear, distinct bipolar spikes
%                 amplitude = 2.5 + rand() * 2;  % 2.5-4.5 amplitude
%                 polarity = (-1)^randi([0 1]);  % Random polarity
% 
%                 % Sharp bipolar pulse
%                 spike_width = 3 + randi(3);  % 3-6 samples wide
% 
%                 if start_idx + spike_width - 1 <= length(clean_signal)
%                     % Main spike
%                     clean_signal(start_idx) = polarity * amplitude;
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.8;
%                     end
% 
%                     % Decay tail
%                     for j = 2:spike_width-1
%                         if start_idx + j <= length(clean_signal)
%                             clean_signal(start_idx + j) = polarity * amplitude * 0.3 * exp(-(j-1));
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 4  % Type D: 18mm
%         clean_signal = zeros(size(t));
% 
%         % Sparse PD events with random locations
%         num_events = 55 + randi(10);  % 55-65 events (sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Mixed event types with random characteristics
%                 event_type = rand();
%                 amplitude = 2 + rand() * 3;  % 2-5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 if event_type < 0.7  % 70% - Sharp bipolar spikes
%                     spike_width = 2 + randi(4);
% 
%                     if start_idx + spike_width - 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         if start_idx + 1 <= length(clean_signal)
%                             clean_signal(start_idx + 1) = -polarity * amplitude * 0.7;
%                         end
% 
%                         % Add some oscillatory tail
%                         for j = 2:spike_width-1
%                             if start_idx + j <= length(clean_signal)
%                                 clean_signal(start_idx + j) = polarity * amplitude * 0.2 * sin(j);
%                             end
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 5  % Type E: 20mm
%         clean_signal = zeros(size(t));
% 
%         % Moderate to high frequency PD events with random locations
%         num_events = 120 + randi(30);  % 120-150 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 2 + rand() * 4;  % 2-6 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.6  % 60% - Sharp spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 40% - Multi-frequency transients
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                 end
%             end
%         end
% 
%     else  % Type F: 25mm
%         clean_signal = zeros(size(t));
% 
%         % 25mm events with random locations throughout
%         num_events = 250 + randi(80);  % 250-330 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 3 + rand() * 4.2;  % 3-7.2 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.4  % 40% - Quick spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 60% - Complex multi-component events
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                 end
%             end
%         end
%     end
% % 
%     % Keep signals in a reasonable range but preserve relative amplitudes
%     max_amplitude = max(abs(clean_signal));
%     if max_amplitude > 0
%         if max_amplitude > 5
%             clean_signal = clean_signal * (5 / max_amplitude);
%         end
%     end
% 
%     % Normalize clean signal
%     clean_signal = clean_signal / (max(abs(clean_signal)) + eps);  % Avoid division by zero
% 
%     % Add Noise (matching your original noise model)
%     white_noise = 0.08*randn(size(clean_signal));
%     powerline_noise = 0.025*sin(2*pi*50e6*t) + 0.015*sin(2*pi*150e6*t);
%     narrowband = 0.03*sin(2*pi*80e6*t + rand()*2*pi);
%     impulse_noise = zeros(size(clean_signal));
%     spike_pos = randperm(length(clean_signal), 15);
%     impulse_noise(spike_pos) = 0.4*(0.2 + 0.8*rand(1,15));
%     noise = white_noise + powerline_noise + narrowband + impulse_noise;
% 
%     % Adjust SNR
%     current_snr = 10*log10(var(clean_signal) / (var(noise) + eps));
%     desired_snr = -10 + rand()*8;
%     noise = noise * 10^((current_snr-desired_snr)/20);
%     noisy_signal = clean_signal + noise;
% 
%     clean_signals(i,:) = clean_signal;
%     noisy_signals(i,:) = noisy_signal;
% end
% 
% % Plot Type A-F samples
% figure('Position', [100, 100, 1600, 1200]);
% type_names = {'Type A: Sparse PD pulses', 'Type B: Spike-dense signal', ...
%               'Type C: 10mm', 'Type D: 18mm', 'Type E: 20mm', 'Type F: 25mm'};
% 
% for type_idx = 1:6
%     subplot(6,2,(type_idx-1)*2+1);
%     plot(t, clean_signals(type_idx,:));
%     title([type_names{type_idx} ' (Clean)']);
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     subplot(6,2,(type_idx-1)*2+2);
%     plot(t, noisy_signals(type_idx,:));
%     title([type_names{type_idx} ' (Noisy)']);
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% end
% 
% %% 2. Define Optimal wavelet selection function
% function wname = selectOptimalWavelet(signal)
%     % Test different wavelets and select the one with lowest reconstruction error
%     wavelets = {'sym4', 'sym6', 'db4', 'db6', 'coif3', 'bior4.4'};
%     min_error = inf;
%     best_wavelet = 'sym6'; % Default
% 
%     for w = 1:length(wavelets)
%         wname = wavelets{w};
%         try
%             % Level 5 decomposition (conservative)
%             [c, l] = wavedec(signal, 5, wname);
%             % Simple reconstruction without thresholding for testing
%             rec = waverec(c, l, wname);
%             error = mean((signal - rec).^2);
% 
%             if error < min_error
%                 min_error = error;
%                 best_wavelet = wname;
%             end
%         catch
%             % Skip if wavelet causes error
%             continue;
%         end
%     end
%     wname = best_wavelet;
% end
% 
% %% 3. Wavelet Preprocessing
% wavelet_processed_signals = zeros(size(noisy_signals));
% 
% for i = 1:size(noisy_signals, 1)
%     sig = noisy_signals(i,:);
% 
%     % 1. Select optimal wavelet for this specific signal type
%     wname = selectOptimalWavelet(sig);
% 
%     % 2. Use a conservative decomposition level
%     level = 5;
%     [c, l] = wavedec(sig, level, wname);
% 
%     % 3. Adaptive thresholding strategy using BayesShrink
%     c_den = c;
% 
%     % Process each level with adaptive BayesShrink thresholding
%     for j = 1:level
%         % Get detail coefficients at this level
%         d = detcoef(c, l, j);
% 
%         % Skip if all zeros or very small
%         if all(abs(d) < 1e-10)
%             continue;
%         end
% 
%         % Estimate noise using robust median estimator
%         sigma = median(abs(d))/0.6745;
% 
%         % If sigma is too small, use a minimum value
%         if sigma < 1e-10
%             sigma = std(d) / 10;
%         end
% 
%         % Calculate variance of signal without noise
%         var_signal = max(0, var(d) - sigma^2);
% 
%         % BayesShrink threshold
%         if var_signal > 0
%             thr = (sigma^2) / sqrt(var_signal);
%         else
%             % Conservative universal threshold if signal variance estimation fails
%             thr = sigma * sqrt(2*log(length(d)));
%         end
% 
%         % Scale threshold based on level (more conservative for low frequencies)
%         level_factor = 1.0 - 0.15*j;  % Lower levels get less thresholding
%         thr = thr * level_factor;
% 
%         % Find level indices
%         start_idx = sum(l(1:(j))) + 1;
%         end_idx = sum(l(1:(j+1)));
%         idx = start_idx:end_idx;
% 
%         % Apply soft thresholding
%         c_den(idx) = wthresh(c(idx), 's', thr);
%     end
% 
%     % 4. Handle approximation coefficients separately
%     approx_coef = appcoef(c, l, wname);
%     if ~isempty(approx_coef) && any(approx_coef)
%         sigma_a = std(approx_coef) / 10;  % Very conservative threshold
%         approx_idx = 1:l(1);
%         c_den(approx_idx) = wthresh(c(approx_idx), 's', sigma_a);
%     end
% 
%     % 5. Reconstruction with improved coefficients
%     sig_denoised = waverec(c_den, l, wname);
% 
%     % 6. Post-wavelet cleanup - remove artifacts
%     diff_signal = abs(sig_denoised - sig);
%     mean_diff = mean(diff_signal);
%     std_diff = std(diff_signal);
% 
%     % Identify potential artifacts
%     artifact_threshold = mean_diff + 3*std_diff;
%     artifact_indices = find(diff_signal > artifact_threshold);
% 
%     % For each potential artifact, smooth it out
%     for idx = artifact_indices'
%         sig_denoised(idx) = sig(idx);
%     end
% 
%     wavelet_processed_signals(i,:) = sig_denoised;
% end
% 
% % Calculate SNR after wavelet pre-processing
% wavelet_preproc_snr = zeros(num_samples, 1);
% for i = 1:num_samples
%     clean_power = mean(clean_signals(i,:).^2);
%     noise_power = mean((clean_signals(i,:) - wavelet_processed_signals(i,:)).^2);
%     wavelet_preproc_snr(i) = 10*log10(clean_power/noise_power);
% end
% 
% fprintf('Average SNR after wavelet pre-processing: %.2f dB\n', mean(wavelet_preproc_snr));
% 
% %% 4. Enhanced Data Preparation with Stratified Sampling for 6 Types
% clean_mean = mean(clean_signals(:));
% clean_std = std(clean_signals(:));
% wavelet_mean = mean(wavelet_processed_signals(:));
% wavelet_std = std(wavelet_processed_signals(:));
% 
% clean_signals_norm = (clean_signals - clean_mean) / clean_std;
% wavelet_signals_norm = (wavelet_processed_signals - wavelet_mean) / wavelet_std;
% 
% % Stratified split for 6 signal types
% samples_per_type = num_samples / 6;
% train_indices = [];
% val_indices = [];
% test_indices = [];
% 
% for type_idx = 1:6
%     % Get indices for this type
%     type_indices = ((type_idx-1)*samples_per_type + 1):(type_idx*samples_per_type);
% 
%     % Random permutation within type
%     type_perm = type_indices(randperm(length(type_indices)));
% 
%     % Split ratios
%     train_size = floor(0.7 * samples_per_type);
%     val_size = floor(0.15 * samples_per_type);
% 
%     % Assign indices
%     train_indices = [train_indices, type_perm(1:train_size)];
%     val_indices = [val_indices, type_perm(train_size+1:train_size+val_size)];
%     test_indices = [test_indices, type_perm(train_size+val_size+1:end)];
% end
% 
% % Reshape for network
% X_train = reshape(wavelet_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% Y_train = reshape(clean_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% X_val = reshape(wavelet_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% Y_val = reshape(clean_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% X_test = reshape(wavelet_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% Y_test = reshape(clean_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% 
% % Save test data for comparison
% X_test_orig_noisy = reshape(noisy_signals(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% 
% %% 5. CNN Architecture
% layers = [
%     imageInputLayer([signal_length,1,1], 'Name', 'input')
% 
%     % Encoder with residual-like connections
%     convolution2dLayer([7,1], 64, 'Padding','same', 'Name','conv1')
%     batchNormalizationLayer('Name','bn1')
%     leakyReluLayer(0.1, 'Name','lrelu1')
% 
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','conv2')
%     batchNormalizationLayer('Name','bn2')
%     leakyReluLayer(0.1, 'Name','lrelu2')
% 
%     % Multi-scale feature extraction
%     convolution2dLayer([3,1], 128, 'Padding','same', 'Name','conv3a')
%     batchNormalizationLayer('Name','bn3a')
%     leakyReluLayer(0.1, 'Name','lrelu3a')
% 
%     % Dilated convolutions 
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',2, 'Name','dilated1')
%     batchNormalizationLayer('Name','bn_dilated1')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated1')
%     dropoutLayer(0.1, 'Name','dropout1')
% 
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',4, 'Name','dilated2')
%     batchNormalizationLayer('Name','bn_dilated2')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated2')
%     dropoutLayer(0.1, 'Name','dropout2')
% 
%     % Decoder path 
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','decode1')
%     batchNormalizationLayer('Name','bn_decode1')
%     leakyReluLayer(0.1, 'Name','lrelu_decode1')
% 
%     convolution2dLayer([3,1], 32, 'Padding','same', 'Name','decode2')
%     batchNormalizationLayer('Name','bn_decode2')
%     leakyReluLayer(0.1, 'Name','lrelu_decode2')
% 
%     % Final output
%     convolution2dLayer([3,1], 1, 'Padding','same', 'Name','output')
%     regressionLayer()
% ];
% 
% %% 6. Training Configuration 
% options = trainingOptions('adam', ...
%     'MaxEpochs', 300, ...                   
%     'MiniBatchSize', 24, ...                % Slightly reduced for better gradients
%     'InitialLearnRate', 3e-4, ...           % Optimized learning rate
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.6, ...         % More gradual reduction
%     'LearnRateDropPeriod', 25, ...          
%     'L2Regularization', 1.5e-4, ...         % Slightly increased regularization
%     'GradientThreshold', 0.8, ...           % Better gradient clipping
%     'ValidationData', {X_val, Y_val}, ...   
%     'ValidationFrequency', 150, ...          
%     'ValidationPatience', 80, ...           
%     'Shuffle', 'every-epoch', ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');
% 
% %% 7. Train Network
% net = trainNetwork(X_train, Y_train, layers, options);
% 
% %% 6. Plot Results - Random Samples of Type A-F (FIXED)
% % Find Type A-F samples in test set
% type_a_test_indices = find(mod(test_indices-1, 6) == 0);  % Type A samples
% type_b_test_indices = find(mod(test_indices-1, 6) == 1);  % Type B samples
% type_c_test_indices = find(mod(test_indices-1, 6) == 2);  % Type C samples
% type_d_test_indices = find(mod(test_indices-1, 6) == 3);  % Type D samples
% type_e_test_indices = find(mod(test_indices-1, 6) == 4);  % Type E samples
% type_f_test_indices = find(mod(test_indices-1, 6) == 5);  % Type F samples
% 
% % Store all test indices for each type
% all_type_test_indices = {type_a_test_indices, type_b_test_indices, type_c_test_indices, ...
%                          type_d_test_indices, type_e_test_indices, type_f_test_indices};
% type_labels = {'A', 'B', 'C', 'D', 'E', 'F'};
% type_names_short = {'Sparse PD pulses', 'Spike-dense signal', '10mm', '18mm', '20mm', '25mm'};
% 
% % Plot one random sample from each type
% for type_idx = 1:6
%     type_test_indices_current = all_type_test_indices{type_idx};
%     if ~isempty(type_test_indices_current)
%         % Select a random position from this type's test indices
%         random_position = type_test_indices_current(randi(length(type_test_indices_current)));
% 
%         % Extract signals (already in original scale)
%         clean_signal = squeeze(Y_test_orig(:,1,1,random_position));
%         noisy_signal = squeeze(X_test_noisy_orig(:,1,1,random_position));
%         wavelet_signal = squeeze(X_test_wavelet_orig(:,1,1,random_position));
%         denoised_signal = squeeze(Y_pred_orig(:,1,1,random_position));
% 
%         % Ensure vectors are the same orientation as t
%         clean_signal = clean_signal(:)';      % Convert to row vector
%         noisy_signal = noisy_signal(:)';      % Convert to row vector
%         wavelet_signal = wavelet_signal(:)';  % Convert to row vector
%         denoised_signal = denoised_signal(:)'; % Convert to row vector
% 
%         % Create plot for this type
%         figure;
%         subplot(4,1,1); plot(t, clean_signal); 
%         title(['Type ' type_labels{type_idx} ' - Clean (' type_names_short{type_idx} ')']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
% 
%         subplot(4,1,2); plot(t, noisy_signal); 
%         title(['Type ' type_labels{type_idx} ' - Noisy']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
% 
%         subplot(4,1,3); plot(t, wavelet_signal); 
%         title(['Type ' type_labels{type_idx} ' - Wavelet Only']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
% 
%         subplot(4,1,4); plot(t, denoised_signal); 
%         title(['Type ' type_labels{type_idx} ' - Wavelet-CNN']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
%     end
% end
% %% 8. Comprehensive Evaluation
% Y_pred = predict(net, X_test);
% 
% % Convert back to original scale
% Y_pred_orig = Y_pred * clean_std + clean_mean;
% Y_test_orig = Y_test * clean_std + clean_mean;
% X_test_wavelet_orig = X_test * wavelet_std + wavelet_mean;
% X_test_noisy_orig = X_test_orig_noisy;
% 
% % Overall performance metrics
% clean_power = mean(Y_test_orig(:).^2);
% noise_power_before = mean((Y_test_orig(:) - X_test_noisy_orig(:)).^2);
% snr_before = 10*log10(clean_power/noise_power_before);
% 
% noise_power_wavelet = mean((Y_test_orig(:) - X_test_wavelet_orig(:)).^2);
% snr_wavelet = 10*log10(clean_power/noise_power_wavelet);
% 
% noise_power_wavelet_cnn = mean((Y_test_orig(:) - Y_pred_orig(:)).^2);
% snr_wavelet_cnn = 10*log10(clean_power/noise_power_wavelet_cnn);
% 
% cc_wavelet = corrcoef(Y_test_orig(:), X_test_wavelet_orig(:));
% cc_wavelet = cc_wavelet(1,2);
% cc_wavelet_cnn = corrcoef(Y_test_orig(:), Y_pred_orig(:));
% cc_wavelet_cnn = cc_wavelet_cnn(1,2);
% 
% % Display overall results
% fprintf('\n=== Denoising Performance ===\n');
% fprintf('                |  Before  | Wavelet | Wavelet_CNN \n');
% fprintf('------------------------------------------------------\n');
% fprintf('MSE             |    -     | %.4f  | %.4f\n', noise_power_wavelet, noise_power_wavelet_cnn);
% fprintf('SNR (dB)        | %6.2f   | %6.2f  | %6.2f\n', snr_before, snr_wavelet, snr_wavelet_cnn);
% fprintf('CC              |    -     | %.4f  | %.4f\n', cc_wavelet, cc_wavelet_cnn);
% fprintf('Improvement (dB)| -        | %6.2f  | %6.2f\n', ...
%         snr_wavelet - snr_before, snr_wavelet_cnn - snr_before);
% 
% %% 9. Evaluate on 10 Random Test Samples
% num_eval_samples = 10;
% random_indices = randperm(length(test_indices), num_eval_samples);
% avg_snr_before = 0;
% avg_mse_wavelet = 0;
% avg_mse_wavelet_cnn = 0;
% avg_snr_wavelet = 0;
% avg_snr_wavelet_cnn = 0;
% avg_cc_wavelet = 0;
% avg_cc_wavelet_cnn = 0;
% figure;
% for idx = 1:num_eval_samples
%     test_idx = random_indices(idx);
%     x = X_test_orig_noisy(:,1,1,test_idx);
%     y_true = Y_test_orig(:,1,1,test_idx);
%     y_wavelet_cnn = Y_pred_orig(:,1,1,test_idx);
%     y_wavelet = X_test_wavelet_orig(:,1,1,test_idx);
%     % Metrics
%     mse_wavelet_cnn_i = mean((y_true - y_wavelet_cnn).^2);
%     mse_wavelet_i = mean((y_true - y_wavelet).^2);
%     snr_wavelet_cnn_i = 10*log10(mean(y_true.^2) / mse_wavelet_cnn_i);
%     snr_wavelet_i = 10*log10(mean(y_true.^2) / mean((y_true - y_wavelet).^2));
%     cc_wavelet_cnn_i = corrcoef(y_true, y_wavelet_cnn); cc_wavelet_cnn_i = cc_wavelet_cnn_i(1,2);
%     cc_wavelet_i = corrcoef(y_true, y_wavelet); cc_wavelet_i = cc_wavelet_i(1,2);
%     % Accumulate for average
%     avg_snr_before = avg_snr_before + snr_before;
%     avg_mse_wavelet_cnn = avg_mse_wavelet_cnn + mse_wavelet_cnn_i;
%     avg_mse_wavelet = avg_mse_wavelet + mse_wavelet_i;
%     avg_snr_wavelet_cnn = avg_snr_wavelet_cnn + snr_wavelet_cnn_i;
%     avg_snr_wavelet = avg_snr_wavelet + snr_wavelet_i;
%     avg_cc_wavelet_cnn = avg_cc_wavelet_cnn + cc_wavelet_cnn_i;
%     avg_cc_wavelet = avg_cc_wavelet + cc_wavelet_i;
%     % Plot
%     subplot(num_eval_samples, 4, (idx-1)*4 + 1);
%     plot(t, x); title(sprintf('Noisy #%d', idx)); ylabel('Amplitude');
%     subplot(num_eval_samples, 4, (idx-1)*4 + 2);
%     plot(t, y_true); title('Clean'); ylabel('Amplitude');
%     subplot(num_eval_samples, 4, (idx-1)*4 + 3);
%     plot(t, y_wavelet); title('Wavelet'); ylabel('Amplitude');
%     subplot(num_eval_samples, 4, (idx-1)*4 + 4);
%     plot(t, y_wavelet_cnn); title('Wavelet_CNN'); ylabel('Amplitude');
% end
% % Take averages
% avg_snr_before = avg_snr_before / num_eval_samples;
% avg_mse_wavelet = avg_mse_wavelet / num_eval_samples;
% avg_mse_wavelet_cnn = avg_mse_wavelet_cnn / num_eval_samples;
% avg_snr_wavelet = avg_snr_wavelet / num_eval_samples;
% avg_snr_wavelet_cnn = avg_snr_wavelet_cnn / num_eval_samples;
% avg_cc_wavelet = avg_cc_wavelet / num_eval_samples;
% avg_cc_wavelet_cnn = avg_cc_wavelet_cnn / num_eval_samples;
% 
% % Display results
% fprintf('\n=== 10-Sample Random Evaluation ===\n');
% fprintf('MSE       | Wavelet: %.4f | Wavelet_CNN: %.4f\n', avg_mse_wavelet, avg_mse_wavelet_cnn);
% fprintf('SNR (dB)  | Wavelet: %.2f  | Wavelet_CNN: %.2f\n', avg_snr_wavelet, avg_snr_wavelet_cnn);
% fprintf('Corr Coef | Wavelet: %.4f | Wavelet_CNN: %.4f\n', avg_cc_wavelet, avg_cc_wavelet_cnn);
% 
% % Compare with overall results
% fprintf('\n=== Comparison with Overall Results ===\n');
% fprintf('                     | Overall Eval | 10-Sample Avg\n');
% fprintf('-----------------------------------------------------\n');
% fprintf('SNR Before (dB)      | %6.2f       | %6.2f\n', snr_before, avg_snr_before);
% fprintf('SNR CNN (dB)         | %6.2f       | %6.2f\n', snr_wavelet, avg_snr_wavelet);
% fprintf('SNR CNN_Wavelet (dB) | %6.2f       | %6.2f\n', snr_wavelet_cnn, avg_snr_wavelet_cnn);
% fprintf('CC CNN               | %6.4f       | %6.4f\n', cc_wavelet, avg_cc_wavelet);
% fprintf('CC CNN_Wavelet       | %6.4f       | %6.4f\n', cc_wavelet_cnn, avg_cc_wavelet_cnn);
% 
% %% 9. Enhanced Type-Specific Analysis
% type_names = {'Type A', 'Type B', 'Type C', 'Type D', 'Type E', 'Type F'};
% type_names_full = {'Type A: Sparse PD pulses', 'Type B: Spike-dense signal', ...
%                    'Type C: 10mm', 'Type D: 18mm', 'Type E: 20mm', 'Type F: 25mm'};
% 
% % Get test indices for each type
% type_a_test = test_indices(mod(test_indices-1, 6) == 0);
% type_b_test = test_indices(mod(test_indices-1, 6) == 1);
% type_c_test = test_indices(mod(test_indices-1, 6) == 2);
% type_d_test = test_indices(mod(test_indices-1, 6) == 3);
% type_e_test = test_indices(mod(test_indices-1, 6) == 4);
% type_f_test = test_indices(mod(test_indices-1, 6) == 5);
% 
% type_test_indices_list = {type_a_test, type_b_test, type_c_test, type_d_test, type_e_test, type_f_test};
% 
% % Initialize arrays to store comprehensive metrics
% num_types = 6;
% snr_before_types = zeros(num_types, 1);
% snr_improvements = zeros(num_types, 2); % [type, method] (wavelet, wavelet-cnn)
% cc_values = zeros(num_types, 2); % [type, method]
% mse_values = zeros(num_types, 2); % [type, method]
% sample_counts = zeros(num_types, 1);
% 
% fprintf('\n=== Enhanced Performance Analysis by Signal Type ===\n');
% fprintf('Signal Type                | Initial SNR (dB) | Samples\n');
% fprintf('----------------------------------------------------------\n');
% 
% for type_idx = 1:num_types
%     type_test = type_test_indices_list{type_idx};
% 
%     if isempty(type_test)
%         fprintf('Warning: No test samples found for %s\n', type_names_full{type_idx});
%         continue;
%     end
% 
%     % Find indices in the test set
%     type_indices_in_test = find(ismember(test_indices, type_test));
%     sample_counts(type_idx) = length(type_indices_in_test);
% 
%     if isempty(type_indices_in_test)
%         continue;
%     end
% 
%     % Extract signals for this type
%     x_noisy_type = X_test_noisy_orig(:,:,:,type_indices_in_test);
%     y_true_type = Y_test_orig(:,:,:,type_indices_in_test);
%     x_wavelet_type = X_test_wavelet_orig(:,:,:,type_indices_in_test);
%     y_pred_type = Y_pred_orig(:,:,:,type_indices_in_test);
% 
%     % Calculate SNR before denoising
%     clean_power_type = mean(y_true_type(:).^2);
%     noise_power_before_type = mean((y_true_type(:) - x_noisy_type(:)).^2);
%     snr_before_types(type_idx) = 10*log10(clean_power_type/noise_power_before_type);
% 
%     % Wavelet-only metrics
%     residual_wavelet_type = y_true_type(:) - x_wavelet_type(:);
%     mse_wavelet_type = mean(residual_wavelet_type.^2);
%     snr_wavelet_type = 10*log10(clean_power_type/mse_wavelet_type);
%     cc_wavelet_type = corrcoef(y_true_type(:), x_wavelet_type(:));
%     cc_wavelet_type = cc_wavelet_type(1,2);
% 
%     % Wavelet-CNN metrics
%     residual_cnn_type = y_true_type(:) - y_pred_type(:);
%     mse_cnn_type = mean(residual_cnn_type.^2);
%     snr_cnn_type = 10*log10(clean_power_type/mse_cnn_type);
%     cc_cnn_type = corrcoef(y_true_type(:), y_pred_type(:));
%     cc_cnn_type = cc_cnn_type(1,2);
% 
%     % Store improvements and metrics
%     snr_improvements(type_idx, 1) = snr_wavelet_type - snr_before_types(type_idx);
%     snr_improvements(type_idx, 2) = snr_cnn_type - snr_before_types(type_idx);
%     cc_values(type_idx, 1) = cc_wavelet_type;
%     cc_values(type_idx, 2) = cc_cnn_type;
%     mse_values(type_idx, 1) = mse_wavelet_type;
%     mse_values(type_idx, 2) = mse_cnn_type;
% 
%     fprintf('%-26s | %8.2f        | %7d\n', type_names_full{type_idx}, ...
%         snr_before_types(type_idx), sample_counts(type_idx));
% end
% 
% fprintf('\n=== SNR Improvement and Correlation by Signal Type ===\n');
% fprintf('Signal Type                |                Wavelet               |            Wavelet-CNN               |\n');
% fprintf('                           |     SNR      |     CC    |    MSE    |     SNR      |      CC     |     MSE     |\n');
% fprintf('-------------------------------------------------------------------------------------------------------------\n');
% 
% for type_idx = 1:num_types
%     if sample_counts(type_idx) > 0
%         fprintf('%-26s | %8.2f dB  | %8.4f  | %8.5f  | %8.2f dB  | %10.4f  | %11.5f |\n', ...
%             type_names_full{type_idx}, snr_improvements(type_idx, 1), cc_values(type_idx, 1), mse_values(type_idx, 1), ...
%             snr_improvements(type_idx, 2), cc_values(type_idx, 2), mse_values(type_idx, 2));
%     end
% end
% 
% % Calculate and display average improvements
% valid_types = sample_counts > 0;
% avg_snr_improvement_wavelet = mean(snr_improvements(valid_types, 1));
% avg_snr_improvement_cnn = mean(snr_improvements(valid_types, 2));
% avg_cc_wavelet_all = mean(cc_values(valid_types, 1));
% avg_cc_cnn_all = mean(cc_values(valid_types, 2));
% 
% fprintf('\n=== Average Performance Across All Types ===\n');
% fprintf('Method       |   Avg SNR    |   Avg CC\n');
% fprintf('-------------------------------------\n');
% fprintf('Wavelet      | %8.2f dB  | %8.4f\n', avg_snr_improvement_wavelet, avg_cc_wavelet_all);
% fprintf('Wavelet-CNN  | %8.2f dB  | %8.4f\n', avg_snr_improvement_cnn, avg_cc_cnn_all);
% fprintf('Additional CNN Gain: %.2f dB\n', avg_snr_improvement_cnn - avg_snr_improvement_wavelet);
% 
% %% 10. Enhanced Visual Comparison - Split into Multiple Figures
% % Find representative samples of each type from the test set
% type_sample_indices = [];
% type_sample_names = {};
% type_labels = {'A', 'B', 'C', 'D', 'E', 'F'};
% 
% for type_idx = 1:6
%     type_test = type_test_indices_list{type_idx};
%     if ~isempty(type_test)
%         % Find first sample of this type in test set
%         type_indices_in_test = find(ismember(test_indices, type_test));
%         if ~isempty(type_indices_in_test)
%             type_sample_indices(end+1) = type_indices_in_test(1);
%             type_sample_names{end+1} = type_names_full{type_idx};
%         end
%     end
% end
% 
% % Split into 2 figures: Types A-C and Types D-F
% types_per_figure = 3;
% methods = {'Wavelet', 'Wavelet-CNN'};
% 
% for fig_num = 1:2
%     figure('Position', [100 + (fig_num-1)*50, 100 + (fig_num-1)*50, 1800, 900]);
% 
%     start_type = (fig_num-1) * types_per_figure + 1;
%     end_type = min(fig_num * types_per_figure, length(type_sample_indices));
% 
%     for local_type = 1:(end_type - start_type + 1)
%         s = start_type + local_type - 1;
%         idx = type_sample_indices(s);
% 
%         % Extract signals
%         noisy_signal = squeeze(X_test_noisy_orig(:,1,1,idx));
%         clean_signal = squeeze(Y_test_orig(:,1,1,idx));
%         wavelet_signal = squeeze(X_test_wavelet_orig(:,1,1,idx));
%         wavelet_cnn_signal = squeeze(Y_pred_orig(:,1,1,idx));
% 
%         % Convert to row vectors for plotting
%         noisy_signal = noisy_signal(:)';
%         clean_signal = clean_signal(:)';
%         wavelet_signal = wavelet_signal(:)';
%         wavelet_cnn_signal = wavelet_cnn_signal(:)';
% 
%         % 1. Original signals (noisy vs clean)
%         subplot(types_per_figure, 3, (local_type-1)*3 + 1);
%         plot(t, noisy_signal, 'r'); hold on;
%         plot(t, clean_signal, 'g');
%         title([type_sample_names{s} ': Original Signals']);
%         legend('Noisy', 'Clean');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
% 
%         % 2. All denoised methods vs clean
%         subplot(types_per_figure, 3, (local_type-1)*3 + 2);
%         plot(t, clean_signal, 'g', 'LineWidth', 2); hold on;
%         plot(t, wavelet_signal, 'b', 'LineWidth', 1);
%         plot(t, wavelet_cnn_signal, 'm', 'LineWidth', 1);
%         title([type_sample_names{s} ': All Methods']);
%         legend('Clean', 'Wavelet', 'Wavelet-CNN');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
% 
%         % 3. Zoomed-in section for detail
%         subplot(types_per_figure, 3, (local_type-1)*3 + 3);
% 
%         % Define zoom ranges based on signal type
%         if contains(type_sample_names{s}, 'Type A')  % Sparse PD pulses
%             zoom_range = round(0.15e-6 * fs):round(0.35e-6 * fs);
%         elseif contains(type_sample_names{s}, 'Type B')  % Spike-dense
%             zoom_range = round(0.4e-6 * fs):round(0.6e-6 * fs);
%         elseif contains(type_sample_names{s}, 'Type C')  % 10mm - very sparse
%             zoom_range = round(0.2e-6 * fs):round(0.4e-6 * fs);
%         elseif contains(type_sample_names{s}, 'Type D')  % 18mm - sparse  
%             zoom_range = round(0.55e-6 * fs):round(0.75e-6 * fs);
%         elseif contains(type_sample_names{s}, 'Type E')  % 20mm - moderate-high
%             zoom_range = round(0.4e-6 * fs):round(0.6e-6 * fs);
%         else  % Type F - 25mm - very high frequency
%             zoom_range = round(0.3e-6 * fs):round(0.5e-6 * fs);
%         end
% 
%         % Ensure zoom range is within bounds
%         zoom_range = zoom_range(zoom_range >= 1 & zoom_range <= length(t));
% 
%         plot(t(zoom_range), clean_signal(zoom_range), 'g', 'LineWidth', 2); hold on;
%         plot(t(zoom_range), wavelet_signal(zoom_range), 'b', 'LineWidth', 1);
%         plot(t(zoom_range), wavelet_cnn_signal(zoom_range), 'm', 'LineWidth', 1);
%         title([type_sample_names{s} ': Zoomed Detail']);
%         legend('Clean', 'Wavelet', 'Wavelet-CNN');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
%     end
% 
%     % Add figure title
%     if fig_num == 1
%         sgtitle('Signal Type Comparison - Types A, B, C', 'FontSize', 16, 'FontWeight', 'bold');
%     else
%         sgtitle('Signal Type Comparison - Types D, E, F', 'FontSize', 16, 'FontWeight', 'bold');
%     end
% end
% 
% 
% %% 13. Enhanced Save Results with Comprehensive Data
% try
%     % Create a comprehensive results structure
%     results = struct();
%     results.net = net;
%     results.Y_pred_orig = Y_pred_orig;
%     results.X_test_wavelet_orig = X_test_wavelet_orig;
%     results.X_test_noisy_orig = X_test_noisy_orig;
%     results.Y_test_orig = Y_test_orig;
% 
%     % Overall metrics
%     results.overall_metrics = struct();
%     results.overall_metrics.snr_before = snr_before;
%     results.overall_metrics.snr_wavelet = snr_wavelet;
%     results.overall_metrics.snr_cnn = snr_cnn;
%     results.overall_metrics.cc_wavelet = cc_wavelet;
%     results.overall_metrics.cc_cnn = cc_cnn;
%     results.overall_metrics.noise_power_wavelet = noise_power_wavelet;
%     results.overall_metrics.noise_power_cnn = noise_power_cnn;
% 
%     % Type-specific metrics
%     results.type_metrics = struct();
%     results.type_metrics.type_names = type_names;
%     results.type_metrics.type_names_full = type_names_full;
%     results.type_metrics.snr_before_types = snr_before_types;
%     results.type_metrics.snr_improvements = snr_improvements;
%     results.type_metrics.cc_values = cc_values;
%     results.type_metrics.mse_values = mse_values;
%     results.type_metrics.sample_counts = sample_counts;
% 
%     % Averages
%     results.averages = struct();
%     results.averages.avg_snr_improvement_wavelet = avg_snr_improvement_wavelet;
%     results.averages.avg_snr_improvement_cnn = avg_snr_improvement_cnn;
%     results.averages.avg_cc_wavelet = avg_cc_wavelet_all;
%     results.averages.avg_cc_cnn = avg_cc_cnn_all;
% 
%     % Signal parameters
%     results.signal_params = struct();
%     results.signal_params.fs = fs;
%     results.signal_params.t_total = t_total;
%     results.signal_params.num_samples = num_samples;
%     results.signal_params.signal_length = signal_length;
% 
%     % Training parameters
%     results.training_params = struct();
%     results.training_params.max_epochs = options.MaxEpochs;
%     results.training_params.batch_size = options.MiniBatchSize;
%     results.training_params.learning_rate = options.InitialLearnRate;
% 
%     results.test_indices = test_indices;
% 
%     % Save the comprehensive results
%     save('wavelet_cnn_ABCDEF_v3_results.mat', 'results');
%     fprintf('Comprehensive results saved to wavelet_cnn_ABCDEF_v3_results.mat\n');
% 
%     % Save just the network
%     save('wavelet_cnn_ABCDEF_v3.mat', 'net');
% 
%     % Save detailed performance metrics separately
%     performance_summary = struct();
%     performance_summary.type_names = type_names_full;
%     performance_summary.snr_before = snr_before_types;
%     performance_summary.snr_improvements = snr_improvements;
%     performance_summary.correlation_coefficients = cc_values;
%     performance_summary.mse_values = mse_values;
%     performance_summary.sample_counts = sample_counts;
%     performance_summary.avg_snr_improvement_wavelet = avg_snr_improvement_wavelet;
%     performance_summary.avg_snr_improvement_cnn = avg_snr_improvement_cnn;
%     performance_summary.avg_cc_wavelet = avg_cc_wavelet_all;
%     performance_summary.avg_cc_cnn = avg_cc_cnn_all;
%     performance_summary.additional_cnn_gain = avg_snr_improvement_cnn - avg_snr_improvement_wavelet;
% 
%     save('wavelet_cnn_performance_summary_ABCDEF.mat', 'performance_summary');
% 
%     % Verify saves
%     fileInfo1 = dir('wavelet_cnn_ABCDEF_v3_results.mat');
%     fileInfo2 = dir('wavelet_cnn_ABCDEF_v3.mat');
%     fileInfo3 = dir('wavelet_cnn_performance_summary_ABCDEF.mat');
% 
%     if ~isempty(fileInfo1) && fileInfo1.bytes > 0
%         fprintf('Comprehensive results file saved successfully (%d bytes)\n', fileInfo1.bytes);
%     end
% 
%     if ~isempty(fileInfo2) && fileInfo2.bytes > 0
%         fprintf('Network file saved successfully (%d bytes)\n', fileInfo2.bytes);
% 
%         % Test loading
%         testLoad = load('wavelet_cnn_ABCDEF_v3.mat');
%         if isfield(testLoad, 'net')
%             fprintf('Verified: Network can be loaded successfully\n');
%         end
%     end
% 
%     if ~isempty(fileInfo3) && fileInfo3.bytes > 0
%         fprintf('Performance summary saved successfully (%d bytes)\n', fileInfo3.bytes);
%     end
% 
% catch ME
%     fprintf('ERROR saving files: %s\n', ME.message);
%     % Try alternative location
%     try
%         save(fullfile(pwd, 'backup_wavelet_cnn_ABCDEF_final.mat'), 'net');
%         fprintf('Network saved to backup location\n');
%     catch
%         fprintf('Failed to save to backup location as well\n');
%     end
% end
% 

%% CNN_Wavelet 
%% --------------CNN + WAVELET (TYPE A B）------------------
% 
% % 1. Enhanced Signal Simulation with Types A and B Only
% fs = 1000e6;                  % 1 GHz sampling
% t_total = 2e-6;               % 2 μs duration
% t = 0:1/fs:t_total;           % Time vector
% signal_length = length(t);
% num_samples = 1000;
% 
% clean_signals = zeros(num_samples, signal_length);
% noisy_signals = zeros(num_samples, signal_length);
% 
% for i = 1:num_samples
%     % Alternate between Type A and Type B samples
%     if mod(i, 2) == 1  % Type A: Sparse PD pulses (4 pulses with gaps)
%         clean_signal = zeros(size(t));
%         start_times = [0.2e-6, 0.6e-6, 1.2e-6, 1.6e-6];  % Clear time gaps
% 
%         for k = 1:length(start_times)
%             A = 10 + rand()*10;
%             fc = 25e6 + rand()*10e6;                    % Higher freq helps sharpen
%             tau = 0.01e-6 + rand()*0.03e-6;             % Very short pulse
%             pulse_t = t - start_times(k);
%             pulse_t = pulse_t(pulse_t >= 0);
% 
%             % Generate a short pulse with fewer points
%             pulse_duration = 0.05e-6;  % 50 ns duration
%             pulse_t = pulse_t(pulse_t <= pulse_duration);
%             pulse = A * exp(-pulse_t/tau) .* sin(2*pi*fc*pulse_t);
% 
%             % Insert pulse at the correct position
%             start_idx = find(t >= start_times(k), 1);
%             pulse_len = length(pulse);
%             if start_idx + pulse_len - 1 <= length(clean_signal)
%                 clean_signal(start_idx:start_idx + pulse_len - 1) = pulse;
%             end
%         end
% 
%     else  % Type B: Spike-dense signal (realistic sharp pulses with gaps)
%         clean_signal = zeros(size(t));
%         num_spikes = 20 + randi(10);  % Fewer, more realistic pulses
%         spike_len = 2;  % Length of each biphasic spike
% 
%         for s = 1:num_spikes
%             start_idx = randi([1, signal_length - spike_len]);
%             amp = 0.5 + 0.5*rand();  % Amplitude
%             direction = (-1)^randi([0 1]);  % Flip polarity randomly
% 
%             % Biphasic spike: [positive, negative] or [negative, positive]
%             clean_signal(start_idx) = direction * amp;
%             clean_signal(start_idx + 1) = -direction * amp;
%         end
%     end
% 
%     % Normalize clean signal
%     clean_signal = clean_signal / max(abs(clean_signal) + eps);  % Avoid division by zero
% 
%     % Noise: White + powerline + narrowband + impulse (with reduced levels)
%     white_noise = 0.08*randn(size(clean_signal));
%     powerline_noise = 0.025*sin(2*pi*50e6*t) + 0.015*sin(2*pi*150e6*t);
%     narrowband = 0.03*sin(2*pi*80e6*t + rand()*2*pi);
%     impulse_noise = zeros(size(clean_signal));
%     spike_pos = randperm(length(clean_signal), 15);
%     impulse_noise(spike_pos) = 0.4*(0.2 + 0.8*rand(1,15));
% 
%     noise = white_noise + powerline_noise + narrowband + impulse_noise;
% 
%     % Adjust SNR (Improved control)
%     current_snr = 10*log10(var(clean_signal) / (var(noise) + eps));
%     desired_snr = -10 + rand()*8;  % 
%     noise = noise * 10^((current_snr-desired_snr)/20);
% 
%     noisy_signal = clean_signal + noise;
% 
%     clean_signals(i,:) = clean_signal;
%     noisy_signals(i,:) = noisy_signal;
% end
% 
% % Plot Type A and B samples
% figure;
% subplot(2,1,1);
% plot(t, clean_signals(1,:));
% title('Type A: Clean Signal (4 Pulses with Gaps)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(2,1,2);
% plot(t, clean_signals(2,:));
% title('Type B: Clean Signal (Simulated Real)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% figure;
% subplot(2,1,1);
% plot(t, noisy_signals(1,:));
% title('Type A : Noisy Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(2,1,2);
% plot(t, noisy_signals(2,:));
% title('Type B : Noisy Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% %% 2. Improved Data Preparation
% % Don't normalize to range, use standardization instead
% clean_mean = mean(clean_signals(:));
% clean_std = std(clean_signals(:));
% noisy_mean = mean(noisy_signals(:));
% noisy_std = std(noisy_signals(:));
% 
% clean_signals_norm = (clean_signals - clean_mean) / clean_std;
% noisy_signals_norm = (noisy_signals - noisy_mean) / noisy_std;
% 
% % Fixed data split with stratification (equal Type A/B in each)
% num_type_a = sum(1:2:num_samples <= num_samples);
% num_type_b = sum(2:2:num_samples <= num_samples);
% 
% % Type indices
% type_a_indices = 1:2:num_samples;
% type_b_indices = 2:2:num_samples;
% 
% type_a_train_size = floor(0.7 * num_type_a);
% type_a_val_size = floor(0.15 * num_type_a);
% type_b_train_size = floor(0.7 * num_type_b);
% type_b_val_size = floor(0.15 * num_type_b);
% 
% % Random permutation within types
% type_a_perm = type_a_indices(randperm(length(type_a_indices)));
% type_b_perm = type_b_indices(randperm(length(type_b_indices)));
% 
% % Split
% train_indices = [type_a_perm(1:type_a_train_size), type_b_perm(1:type_b_train_size)];
% val_indices = [type_a_perm(type_a_train_size+1:type_a_train_size+type_a_val_size), ...
%                type_b_perm(type_b_train_size+1:type_b_train_size+type_b_val_size)];
% test_indices = [type_a_perm(type_a_train_size+type_a_val_size+1:end), ...
%                 type_b_perm(type_b_train_size+type_b_val_size+1:end)];
% 
% % Reshape for network
% X_train = reshape(noisy_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% Y_train = reshape(clean_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% X_val = reshape(noisy_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% Y_val = reshape(clean_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% X_test = reshape(noisy_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% Y_test = reshape(clean_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% 
% %% 3. CNN Architecture
% layers = [
%     imageInputLayer([signal_length,1,1], 'Name', 'input')
% 
%     % Enhanced encoder with residual-like connections
%     convolution2dLayer([7,1], 64, 'Padding','same', 'Name','conv1')
%     batchNormalizationLayer('Name','bn1')
%     leakyReluLayer(0.1, 'Name','lrelu1')
% 
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','conv2')
%     batchNormalizationLayer('Name','bn2')
%     leakyReluLayer(0.1, 'Name','lrelu2')
% 
%     % Multi-scale feature extraction
%     convolution2dLayer([3,1], 128, 'Padding','same', 'Name','conv3a')
%     batchNormalizationLayer('Name','bn3a')
%     leakyReluLayer(0.1, 'Name','lrelu3a')
% 
%     % Dilated convolutions for larger receptive field
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',2, 'Name','dilated1')
%     batchNormalizationLayer('Name','bn_dilated1')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated1')
%     dropoutLayer(0.1, 'Name','dropout1')
% 
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',4, 'Name','dilated2')
%     batchNormalizationLayer('Name','bn_dilated2')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated2')
%     dropoutLayer(0.1, 'Name','dropout2')
% 
%     % Decoder path with improved architecture
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','decode1')
%     batchNormalizationLayer('Name','bn_decode1')
%     leakyReluLayer(0.1, 'Name','lrelu_decode1')
% 
%     convolution2dLayer([3,1], 32, 'Padding','same', 'Name','decode2')
%     batchNormalizationLayer('Name','bn_decode2')
%     leakyReluLayer(0.1, 'Name','lrelu_decode2')
% 
%     % Final output
%     convolution2dLayer([3,1], 1, 'Padding','same', 'Name','output')
%     regressionLayer()
% ];
% 
% %% 4. Training Configuration 
% options = trainingOptions('adam', ...
%     'MaxEpochs', 150, ...                   % Significantly increased
%     'MiniBatchSize', 24, ...                
%     'InitialLearnRate', 2e-4, ...           % Slightly lower initial rate
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.75, ...        % More gradual decay
%     'LearnRateDropPeriod', 50, ...          % Less frequent drops
%     'L2Regularization', 1.5e-4, ...         
%     'GradientThreshold', 0.8, ...           
%     'ValidationData', {X_val, Y_val}, ...   
%     'ValidationFrequency', 120, ...         % Check validation less frequently
%     'ValidationPatience', 60, ...           % Much more patient
%     'Shuffle', 'every-epoch', ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');
% 
% %% 5. Optimal wavelet selection function
% function wname = selectOptimalWavelet(signal)
%     % Test different wavelets and select the one with lowest reconstruction error
%     wavelets = {'sym4', 'sym6', 'db4', 'db6', 'coif3', 'bior4.4'};
%     min_error = inf;
%     best_wavelet = 'sym6'; % Default
% 
%     for w = 1:length(wavelets)
%         wname = wavelets{w};
%         try
%             % Level 5 decomposition (instead of 8 - more conservative)
%             [c, l] = wavedec(signal, 5, wname);
%             % Simple reconstruction without thresholding for testing
%             rec = waverec(c, l, wname);
%             error = mean((signal - rec).^2);
% 
%             if error < min_error
%                 min_error = error;
%                 best_wavelet = wname;
%             end
%         catch
%             % Skip if wavelet causes error
%             continue;
%         end
%     end
%     wname = best_wavelet;
% end
% 
% %% 6. Train Network 
% net = trainNetwork(X_train, Y_train, layers, options);
% 
% %% 7. CNN + Improved Wavelet Post-Processing
% Y_pred_raw = predict(net, X_test);
% 
% % Convert back to original scale
% Y_pred_cnn = Y_pred_raw * clean_std + clean_mean;
% Y_test_orig = Y_test * clean_std + clean_mean; 
% 
% % Improved wavelet denoising parameters and approach
% Y_pred = zeros(size(Y_pred_cnn));
% 
% for i = 1:size(Y_pred_cnn, 4)
%     sig = squeeze(Y_pred_cnn(:,1,1,i));
% 
%     % 1. Select optimal wavelet for this specific signal type
%     wname = selectOptimalWavelet(sig);
% 
%     % 2. Use a more conservative decomposition level (5 instead of 8)
%     level = 5;
%     [c, l] = wavedec(sig, level, wname);
% 
%     % 3. Improved thresholding strategy - using BayesShrink
%     % (more adaptive than the previous approach)
%     c_den = c;
% 
%     % Process each level with adaptive BayesShrink thresholding
%     for j = 1:level
%         % Get detail coefficients at this level
%         d = detcoef(c, l, j);
% 
%         % Skip if all zeros or very small
%         if all(abs(d) < 1e-10)
%             continue;
%         end
% 
%         % Estimate noise using robust median estimator
%         sigma = median(abs(d))/0.6745;
% 
%         % If sigma is too small, use a minimum value
%         if sigma < 1e-10
%             sigma = std(d) / 10;
%         end
% 
%         % Calculate variance of signal without noise
%         var_signal = max(0, var(d) - sigma^2);
% 
%         % BayesShrink threshold
%         if var_signal > 0
%             thr = (sigma^2) / sqrt(var_signal);
%         else
%             % Conservative universal threshold if signal variance estimation fails
%             thr = sigma * sqrt(2*log(length(d)));
%         end
% 
%         % Scale threshold based on level (more conservative for low frequencies)
%         level_factor = 1.0 - 0.15*j;  % Lower levels get less thresholding
%         thr = thr * level_factor;
% 
%         % Find level indices
%         start_idx = sum(l(1:(j))) + 1;
%         end_idx = sum(l(1:(j+1)));
%         idx = start_idx:end_idx;
% 
%         % Apply soft thresholding
%         c_den(idx) = wthresh(c(idx), 's', thr);
%     end
% 
%     % 4. Handle approximation coefficients separately - avoid over-smoothing
%     % Only minimal denoising on approximation coefficients
%     approx_coef = appcoef(c, l, wname);
%     if ~isempty(approx_coef) && any(approx_coef)
%         sigma_a = std(approx_coef) / 10;  % Very conservative threshold
%         approx_idx = 1:l(1);
%         c_den(approx_idx) = wthresh(c(approx_idx), 's', sigma_a);
%     end
% 
%     % 5. Reconstruction with improved coefficients
%     Y_pred(:,1,1,i) = waverec(c_den, l, wname);
% 
%     % 6. Post-wavelet cleanup - remove any introduced artifacts
%     % Find any isolated spikes that weren't in original prediction
%     y_ref = Y_pred(:,1,1,i);
%     y_pred_orig = sig;
% 
%     % Find large differences introduced by wavelet processing
%     diff_signal = abs(y_ref - y_pred_orig);
%     mean_diff = mean(diff_signal);
%     std_diff = std(diff_signal);
% 
%     % Identify potential artifacts (values that deviate significantly from prediction)
%     artifact_threshold = mean_diff + 3*std_diff;
%     artifact_indices = find(diff_signal > artifact_threshold);
% 
%     % For each potential artifact, smooth it out
%     for idx = artifact_indices'
%         % Simple replacement with predicted value
%         y_ref(idx) = y_pred_orig(idx);
%     end
% 
%     Y_pred(:,1,1,i) = y_ref;
% end
% 
% %% 8. Plot Results - Random Samples of Type A and B (FIXED)
% % Find Type A and Type B samples in test set
% type_a_test_indices = find(mod(test_indices-1, 2) == 0);  % Type A samples
% type_b_test_indices = find(mod(test_indices-1, 2) == 1);  % Type B samples
% 
% % Select random samples from each type (CORRECTED LOGIC)
% if ~isempty(type_a_test_indices)
%     % Directly select a random position from Type A test indices
%     type_a_test_position = type_a_test_indices(randi(length(type_a_test_indices)));
% 
%     % Extract signals (convert back to original scale)
%     clean_signal_A = squeeze(Y_test(:,1,1,type_a_test_position)) * clean_std + clean_mean;
%     noisy_signal_A = squeeze(X_test(:,1,1,type_a_test_position)) * noisy_std + noisy_mean;
%     denoised_signal_A = squeeze(Y_pred(:,1,1,type_a_test_position));
% 
%     % Ensure vectors are the same orientation as t
%     clean_signal_A = clean_signal_A(:)';    % Convert to row vector
%     noisy_signal_A = noisy_signal_A(:)';    % Convert to row vector  
%     denoised_signal_A = denoised_signal_A(:)'; % Convert to row vector
% 
%     % Type A Plot
%     figure;
%     subplot(3,1,1); plot(t, clean_signal_A); title('Type A - Clean'); xlabel('Time (s)'); ylabel('Amplitude');
%     subplot(3,1,2); plot(t, noisy_signal_A); title('Type A - Noisy'); xlabel('Time (s)'); ylabel('Amplitude');
%     subplot(3,1,3); plot(t, denoised_signal_A); title('Type A - CNN+Wavelet Denoised'); xlabel('Time (s)'); ylabel('Amplitude');
% end
% 
% if ~isempty(type_b_test_indices)
%     % Directly select a random position from Type B test indices
%     type_b_test_position = type_b_test_indices(randi(length(type_b_test_indices)));
% 
%     % Extract signals (convert back to original scale)
%     clean_signal_B = squeeze(Y_test(:,1,1,type_b_test_position)) * clean_std + clean_mean;
%     noisy_signal_B = squeeze(X_test(:,1,1,type_b_test_position)) * noisy_std + noisy_mean;
%     denoised_signal_B = squeeze(Y_pred(:,1,1,type_b_test_position));
% 
%     % Ensure vectors are the same orientation as t
%     clean_signal_B = clean_signal_B(:)';    % Convert to row vector
%     noisy_signal_B = noisy_signal_B(:)';    % Convert to row vector
%     denoised_signal_B = denoised_signal_B(:)'; % Convert to row vector
% 
%     % Type B Plot
%     figure;
%     subplot(3,1,1); plot(t, clean_signal_B); title('Type B - Clean'); xlabel('Time (s)'); ylabel('Amplitude');
%     subplot(3,1,2); plot(t, noisy_signal_B); title('Type B - Noisy'); xlabel('Time (s)'); ylabel('Amplitude');
%     subplot(3,1,3); plot(t, denoised_signal_B); title('Type B - CNN+Wavelet Denoised'); xlabel('Time (s)'); ylabel('Amplitude');
% end
% 
% %% 9. Evaluation 
% X_test_orig = X_test * noisy_std + noisy_mean;
% 
% clean_power = mean(Y_test_orig(:).^2);
% noise_power = mean((Y_test_orig(:)-X_test_orig(:)).^2);
% snr_before = 10*log10(clean_power/noise_power);
% 
% % CNN+Wavelet output metrics
% residual_cnn_wavelet = Y_test_orig(:) - Y_pred(:);
% mse_cnn_wavelet = mean(residual_cnn_wavelet.^2);
% snr_cnn_wavelet = 10*log10(clean_power/mean(residual_cnn_wavelet.^2));
% cc_cnn_wavelet = corrcoef(Y_test_orig(:), Y_pred(:));
% cc_cnn_wavelet = cc_cnn_wavelet(1,2);
% 
% % Display comprehensive results
% fprintf('=== CNN+Wavelet Denoising Performance ===\n');
% fprintf('                |  Before  | CNN+Wavelet\n');
% fprintf('------------------------------------\n');
% fprintf('MSE             |    -     | %.4f  \n', mse_cnn_wavelet);
% fprintf('SNR (dB)        | %6.2f   | %6.2f  \n',snr_before, snr_cnn_wavelet);
% fprintf('CC              |    -     | %.4f  \n', cc_cnn_wavelet);
% fprintf('Improvement (dB)| -        | %6.2f  \n', snr_cnn_wavelet - snr_before);
% 
% %% 10. Evaluate on 10 Random Test Samples
% num_eval_samples = 10;
% random_indices = randperm(length(test_indices), num_eval_samples);
% avg_snr_before = 0;
% avg_mse_cnn_wavelet = 0;
% avg_snr_cnn_wavelet = 0;
% avg_cc_cnn_wavelet = 0;
% 
% figure;
% for idx = 1:num_eval_samples
%     test_idx = random_indices(idx);
% 
%     x = X_test(:,1,1,test_idx);
%     y_true = Y_test(:,1,1,test_idx);
%     y_cnn_wavelet = Y_pred(:,1,1,test_idx);
% 
%     % Metrics
%     mse_cnn_wavelet_i = mean((y_true - y_cnn_wavelet).^2);
%     snr_cnn_wavelet_i = 10*log10(mean(y_true.^2) / mse_cnn_wavelet_i);
%     cc_cnn_wavelet_i = corrcoef(y_true, y_cnn_wavelet); cc_cnn_wavelet_i = cc_cnn_wavelet_i(1,2);
% 
%     % Accumulate for average
%     avg_snr_before = avg_snr_before + snr_before;
%     avg_mse_cnn_wavelet = avg_mse_cnn_wavelet + mse_cnn_wavelet_i;
%     avg_snr_cnn_wavelet = avg_snr_cnn_wavelet + snr_cnn_wavelet_i;
%     avg_cc_cnn_wavelet = avg_cc_cnn_wavelet + cc_cnn_wavelet_i;
% 
%     % Plot
%     subplot(num_eval_samples, 3, (idx-1)*3 + 1);
%     plot(t, x); title(sprintf('Noisy #%d', idx)); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 3, (idx-1)*3 + 2);
%     plot(t, y_true); title('Clean'); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 3, (idx-1)*3 + 3);
%     plot(t, y_cnn_wavelet); title('CNN+Wavelet'); ylabel('Amplitude');
% end
% 
% % Take averages
% avg_snr_before = avg_snr_before / num_eval_samples;
% avg_mse_cnn_wavelet = avg_mse_cnn_wavelet / num_eval_samples;
% avg_snr_cnn_wavelet = avg_snr_cnn_wavelet / num_eval_samples;
% avg_cc_cnn_wavelet = avg_cc_cnn_wavelet / num_eval_samples;
% 
% % Display results
% fprintf('\n=== 10-Sample Random Evaluation ===\n');
% fprintf('MSE        | CNN+Wavelet: %.4f\n', avg_mse_cnn_wavelet);
% fprintf('SNR (dB)   | CNN+Wavelet: %.2f\n', avg_snr_cnn_wavelet);
% fprintf('Corr Coef  | CNN+Wavelet: %.4f\n', avg_cc_cnn_wavelet);
% 
% % Compare with overall results
% fprintf('\n=== Comparison with Overall Results ===\n');
% fprintf('                     | Overall Eval | 10-Sample Avg\n');
% fprintf('-----------------------------------------------------\n');
% fprintf('SNR Before (dB)      | %6.2f       | %6.2f\n', snr_before, avg_snr_before);
% fprintf('SNR CNN+Wavelet (dB) | %6.2f       | %6.2f\n', snr_cnn_wavelet, avg_snr_cnn_wavelet);
% fprintf('CC CNN+Wavelet       | %6.4f       | %6.4f\n', cc_cnn_wavelet, avg_cc_cnn_wavelet);
% 
% %% 11. Visual comparison of denoised signals for Type A and B (FIXED)
% % Find representative samples of each type from the test set
% type_a_indices = find(mod(test_indices-1, 2) == 0);  % Type A (signal_type = 1)
% type_b_indices = find(mod(test_indices-1, 2) == 1);  % Type B (signal_type = 2)
% 
% % Select the first one of each type (if available) - CORRECTED
% sample_indices = [];
% sample_types = {};
% 
% if ~isempty(type_a_indices)
%     sample_indices(end+1) = type_a_indices(1);  % Use the position directly
%     sample_types{end+1} = 'Type A';
% end
% if ~isempty(type_b_indices)
%     sample_indices(end+1) = type_b_indices(1);  % Use the position directly
%     sample_types{end+1} = 'Type B';
% end
% 
% % Create the comparison figure
% figure('Position', [100, 100, 1200, 600]);
% for s = 1:length(sample_indices)
%     idx = sample_indices(s);
% 
%     % Extract and ensure proper orientation
%     noisy_signal = squeeze(X_test(:,1,1,idx)) * noisy_std + noisy_mean;
%     clean_signal = squeeze(Y_test(:,1,1,idx)) * clean_std + clean_mean;
%     cnn_wavelet_signal = squeeze(Y_pred(:,1,1,idx));
% 
%     % Convert to row vectors to match t
%     noisy_signal = noisy_signal(:)';
%     clean_signal = clean_signal(:)';
%     cnn_wavelet_signal = cnn_wavelet_signal(:)';
% 
%     % Plot original signals
%     subplot(length(sample_indices), 3, (s-1)*3 + 1);
%     plot(t, noisy_signal, 'r'); hold on;
%     plot(t, clean_signal, 'g');
%     title([sample_types{s} ': Original Signals']);
%     legend('Noisy', 'Clean');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     % Plot CNN+Wavelet denoised signal
%     subplot(length(sample_indices), 3, (s-1)*3 + 2);
%     plot(t, clean_signal, 'g'); hold on;
%     plot(t, cnn_wavelet_signal, 'b', 'LineWidth', 1);
%     title([sample_types{s} ': CNN+Wavelet Denoising']);
%     legend('Clean', 'CNN+Wavelet');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     % Plot zoomed-in section for detail
%     subplot(length(sample_indices), 3, (s-1)*3 + 3);
%     % Define zoom ranges based on signal type
%     if contains(sample_types{s}, 'A')  % Type A - very sparse
%         zoom_range = round(0.15e-6 * fs):round(0.35e-6 * fs);
%     else  % Type B - sparse  
%         zoom_range = round(0.55e-6 * fs):round(0.75e-6 * fs);
%     end
% 
%     % Ensure zoom_range is within bounds
%     zoom_range = zoom_range(zoom_range >= 1 & zoom_range <= length(t));
% 
%     plot(t(zoom_range), clean_signal(zoom_range), 'g', 'LineWidth', 2); hold on;
%     plot(t(zoom_range), cnn_wavelet_signal(zoom_range), 'b', 'LineWidth', 1);
%     title([sample_types{s} ': Zoomed Detail']);
%     legend('Clean', 'CNN+Wavelet');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% end
% 
% %% 12. Calculate performance metrics for each signal type
% % Get test indices for each type
% type_a_test = test_indices(mod(test_indices-1, 2) == 0);
% type_b_test = test_indices(mod(test_indices-1, 2) == 1);
% 
% % Initialize arrays to store metrics
% num_types = 2;
% type_names = {'Type A', 'Type B'};
% type_test_indices = {type_a_test, type_b_test};
% 
% snr_before_types = zeros(num_types, 1);
% snr_improvements = zeros(num_types, 1);
% cc_values = zeros(num_types, 1);
% mse_values = zeros(num_types, 1);
% 
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
% 
%     if isempty(type_test)
%         fprintf('Warning: No test samples found for %s\n', type_names{type_idx});
%         continue;
%     end
% 
%     % Find indices in the test set
%     type_indices_in_test = find(ismember(test_indices, type_test));
% 
%     if isempty(type_indices_in_test)
%         continue;
%     end
% 
%     % Extract signals for this type
%     x_type = X_test(:,:,:,type_indices_in_test) * noisy_std + noisy_mean;
%     y_type = Y_test(:,:,:,type_indices_in_test) * clean_std + clean_mean;
%     y_pred_type = Y_pred(:,:,:,type_indices_in_test);
% 
%     % Calculate SNR before denoising
%     clean_power_type = mean(y_type(:).^2);
%     noise_power_type = mean((y_type(:) - x_type(:)).^2);
%     snr_before_types(type_idx) = 10*log10(clean_power_type/noise_power_type);
% 
%     % CNN+Wavelet metrics
%     residual_cnn_wavelet_type = y_type(:) - y_pred_type(:);
%     mse_cnn_wavelet_type = mean(residual_cnn_wavelet_type.^2);
%     snr_cnn_wavelet_type = 10*log10(clean_power_type/mse_cnn_wavelet_type);
%     cc_cnn_wavelet_type = corrcoef(y_type(:), y_pred_type(:));
%     cc_cnn_wavelet_type = cc_cnn_wavelet_type(1,2);
% 
%     % Store improvements and correlation coefficients
%     snr_improvements(type_idx) = snr_cnn_wavelet_type - snr_before_types(type_idx);
%     cc_values(type_idx) = cc_cnn_wavelet_type;
%     mse_values(type_idx) = mse_cnn_wavelet_type;
% end
% 
% % Display comprehensive results by signal type
% fprintf('\n=== CNN+Wavelet Denoising Performance by Signal Type ===\n');
% fprintf('Signal Type | Initial SNR (dB) | Samples\n');
% fprintf('-------------------------------------------\n');
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
%     fprintf('%-11s | %8.2f        | %7d\n', type_names{type_idx}, ...
%         snr_before_types(type_idx), length(type_test));
% end
% 
% fprintf('\n=== SNR Improvement and Correlation by Signal Type ===\n');
% fprintf('Signal Type |           CNN+Wavelet           |\n');
% fprintf('Evaluation  |     SNR      |     CC    |    MSE    |\n');
% fprintf('----------------------------------------------------\n');
% for type_idx = 1:num_types
%     fprintf('%-11s | %8.2f dB  | %8.4f  | %8.5f  |\n', ...
%         type_names{type_idx}, snr_improvements(type_idx), cc_values(type_idx), mse_values(type_idx));
% end
% 
% % Calculate and display average improvements
% avg_snr_improvement_cnn_wavelet = mean(snr_improvements);
% avg_cc_cnn_wavelet_overall = mean(cc_values);
% 
% fprintf('\n=== Average Performance Across All Types ===\n');
% fprintf('Method      |   Avg SNR    |   Avg CC\n');
% fprintf('-------------------------------\n');
% fprintf('CNN+Wavelet | %8.2f dB  | %8.4f\n', avg_snr_improvement_cnn_wavelet, avg_cc_cnn_wavelet_overall);
% 
% %% 13. Save Results with Error Handling
% try
%     % First, verify that the network exists and is valid
%     if ~exist('net', 'var') || isempty(net)
%         error('Network variable is empty or not defined');
%     end
% 
%     % Save the comprehensive results file
%     save('cnn_wavelet_AB_v2_result.mat', 'net', 'Y_pred', ...
%          'mse_cnn_wavelet', 'snr_before', 'snr_cnn_wavelet', 'cc_cnn_wavelet', ...
%          'snr_improvements', 'cc_values', 'mse_values', 'type_names', ...
%          'snr_before_types', 'avg_snr_improvement_cnn_wavelet');
%     fprintf('Comprehensive results saved successfully to cnn_wavelet_AB_v2_result.mat\n');
% 
%     % Save just the network with verification
%     save('cnn_wavelet_AB_v2.mat', 'net');
% 
%     % Verify the saved file
%     fileInfo = dir('cnn_wavelet_AB_v2.mat');
%     if isempty(fileInfo) || fileInfo.bytes == 0
%         error('Failed to save network: File is empty');
%     else
%         fprintf('Network saved successfully to cnn_wavelet_AB_v2.mat (%d bytes)\n', fileInfo.bytes);
% 
%         % Double-check by trying to load it
%         testLoad = load('cnn_wavelet_AB_v2.mat');
%         if ~isfield(testLoad, 'net')
%             error('Saved file does not contain the network variable');
%         else
%             fprintf('Verified: Network was saved correctly and can be loaded\n');
%         end
%     end
% 
%     % Save detailed performance metrics
%     performance_summary = struct();
%     performance_summary.type_names = type_names;
%     performance_summary.snr_before = snr_before_types;
%     performance_summary.snr_improvements = snr_improvements;
%     performance_summary.correlation_coefficients = cc_values;
%     performance_summary.mse_values = mse_values;
%     performance_summary.avg_snr_improvement_cnn_wavelet = avg_snr_improvement_cnn_wavelet;
%     performance_summary.avg_cc_cnn_wavelet = avg_cc_cnn_wavelet_overall;
% 
%     save('performance_summary_wavelet_AB.mat', 'performance_summary');
%     fprintf('Performance summary saved to performance_summary_wavelet_AB.mat\n');
% 
% catch ME
%     fprintf('ERROR saving results: %s\n', ME.message);
%     % Try an alternative save location
%     try
%         alternativePath = fullfile(pwd, 'cnn_wavelet_AB_v2_backup.mat');
%         save(alternativePath, 'net');
%         fprintf('Network saved to alternative location: %s\n', alternativePath);
%     catch
%         fprintf('Failed to save to alternative location as well\n');
%     end
% end
% 
% %% 14. Final Summary
% fprintf('\n=== FINAL SUMMARY ===\n');
% fprintf('Dataset: %d samples with 2 signal types (A, B)\n', num_samples);
% fprintf('Signal length: %d samples (%.1f μs at %.0f MHz)\n', signal_length, t_total*1e6, fs/1e6);
% fprintf('Network: CNN + Wavelet Post-Processing\n');
% fprintf('Training epochs: %d\n', options.MaxEpochs);
% fprintf('\nOverall Performance:\n');
% fprintf('• Average SNR improvement: %.2f dB\n', avg_snr_improvement_cnn_wavelet);
% fprintf('• Average correlation: %.4f\n', avg_cc_cnn_wavelet_overall);
% 
% fprintf('\nBest performing signal type:\n');
% [~, best_cnn_wavelet_idx] = max(snr_improvements);
% fprintf('• CNN+Wavelet: %s (%.2f dB improvement)\n', type_names{best_cnn_wavelet_idx}, snr_improvements(best_cnn_wavelet_idx));
% 
% fprintf('\nFiles saved:\n');
% fprintf('• cnn_wavelet_AB_v2.mat (trained network)\n');
% fprintf('• cnn_wavelet_AB_v2_result.mat (complete results)\n');
% fprintf('• performance_summary_wavelet_AB.mat (performance metrics)\n');
% fprintf('\n=== ANALYSIS COMPLETE ===\n');
% 
% %% ------------------- Add on code for comparison----------------------------------
% % %% 10. Implement All Four Denoising Approaches
% % 
% % % Select one sample from Type A and Type B for detailed comparison
% % type_a_idx = find(mod(test_indices, 2) == 1, 1);
% % type_b_idx = find(mod(test_indices, 2) == 0, 1);
% % sample_indices = [type_a_idx, type_b_idx];
% % 
% % % Initialize arrays to store results
% % sample_types = {'Type A', 'Type B'};
% % methods = {'Only Wavelet', 'Only CNN', 'Wavelet before CNN', 'Wavelet after CNN'};
% % mse_results = zeros(length(methods), length(sample_indices));
% % snr_results = zeros(length(methods), length(sample_indices));
% % cc_results = zeros(length(methods), length(sample_indices));
% % 
% % % Store denoised signals for plotting
% % denoised_signals = cell(length(methods), length(sample_indices));
% % 
% % for s = 1:length(sample_indices)
% %     idx = sample_indices(s);
% %     noisy_signal = squeeze(X_test(:,1,1,idx)) * noisy_std + noisy_mean;
% %     clean_signal = squeeze(Y_test(:,1,1,idx)) * clean_std + clean_mean;
% % 
% %     % Method 1: Only Wavelet
% %     % Select optimal wavelet
% %     wname = selectOptimalWavelet(noisy_signal);
% %     level = 5;
% %     [c, l] = wavedec(noisy_signal, level, wname);
% % 
% %     % Apply BayesShrink thresholding
% %     c_den = c;
% %     for j = 1:level
% %         d = detcoef(c, l, j);
% %         if all(abs(d) < 1e-10)
% %             continue;
% %         end
% %         sigma = median(abs(d))/0.6745;
% %         if sigma < 1e-10
% %             sigma = std(d) / 10;
% %         end
% %         var_signal = max(0, var(d) - sigma^2);
% %         if var_signal > 0
% %             thr = (sigma^2) / sqrt(var_signal);
% %         else
% %             thr = sigma * sqrt(2*log(length(d)));
% %         end
% %         level_factor = 1.0 - 0.15*j;
% %         thr = thr * level_factor;
% %         start_idx = sum(l(1:(j))) + 1;
% %         end_idx = sum(l(1:(j+1)));
% %         idx_range = start_idx:end_idx;
% %         c_den(idx_range) = wthresh(c(idx_range), 's', thr);
% %     end
% % 
% %     % Handle approximation coefficients
% %     approx_coef = appcoef(c, l, wname);
% %     if ~isempty(approx_coef) && any(approx_coef)
% %         sigma_a = std(approx_coef) / 10;
% %         approx_idx = 1:l(1);
% %         c_den(approx_idx) = wthresh(c(approx_idx), 's', sigma_a);
% %     end
% % 
% %     % Reconstruct signal
% %     only_wavelet = waverec(c_den, l, wname);
% %     denoised_signals{1, s} = only_wavelet;
% % 
% %     % Method 2: Only CNN (already done in your code)
% %     only_cnn = squeeze(Y_pred(:,1,1,idx)) * clean_std + clean_mean;
% %     denoised_signals{2, s} = only_cnn;
% % 
% %     % Method 3: Wavelet before CNN
% %     % Apply wavelet denoising first
% %     wavelet_denoised = only_wavelet;
% % 
% %     % Normalize for CNN
% %     wavelet_denoised_norm = (wavelet_denoised - noisy_mean) / noisy_std;
% % 
% %     % Reshape for CNN input
% %     wavelet_denoised_input = reshape(wavelet_denoised_norm, [signal_length,1,1,1]);
% % 
% %     % Apply CNN to wavelet denoised signal
% %     wavelet_then_cnn = predict(net, wavelet_denoised_input);
% %     wavelet_then_cnn = squeeze(wavelet_then_cnn) * clean_std + clean_mean;
% %     denoised_signals{3, s} = wavelet_then_cnn;
% % 
% %     % Method 4: Wavelet after CNN (already in your code)
% %     cnn_then_wavelet = squeeze(Y_refined(:,1,1,idx)) * clean_std + clean_mean;
% %     denoised_signals{4, s} = cnn_then_wavelet;
% % 
% %     % Calculate metrics for all methods
% %     for m = 1:length(methods)
% %         denoised = denoised_signals{m, s};
% %         mse_results(m, s) = mean((clean_signal - denoised).^2);
% %         snr_results(m, s) = 10*log10(mean(clean_signal.^2) / mse_results(m, s));
% %         cc_temp = corrcoef(clean_signal, denoised);
% %         cc_results(m, s) = cc_temp(1,2);
% %     end
% % end
% % 
% % %% 11. Create Comparison Plots
% % 
% % % 1. Visual comparison of denoised signals for Type A and Type B
% % figure('Position', [100, 100, 1200, 800]);
% % 
% % for s = 1:length(sample_indices)
% %     idx = sample_indices(s);
% %     noisy_signal = squeeze(X_test(:,1,1,idx)) * noisy_std + noisy_mean;
% %     clean_signal = squeeze(Y_test(:,1,1,idx)) * clean_std + clean_mean;
% % 
% %     % Plot original signals
% %     subplot(2, 3, (s-1)*3 + 1);
% %     plot(t, noisy_signal, 'b'); hold on;
% %     plot(t, clean_signal, 'r');
% %     title([sample_types{s} ': Original Signals']);
% %     legend('Noisy', 'Clean');
% %     xlabel('Time (s)');
% %     ylabel('Amplitude');
% % 
% %     % Plot all denoised signals
% %     subplot(2, 3, (s-1)*3 + 2);
% %     plot(t, clean_signal, 'r'); hold on;
% %     for m = 1:length(methods)
% %         plot(t, denoised_signals{m, s}, 'LineWidth', 1);
% %     end
% %     title([sample_types{s} ': All Methods']);
% %     legend('Clean', methods{:});
% %     xlabel('Time (s)');
% %     ylabel('Amplitude');
% % 
% %     % Plot zoomed-in section for detail
% %     subplot(2, 3, (s-1)*3 + 3);
% %     % Find a section with interesting features
% %     if s == 1 % Type A
% %         zoom_range = round(0.6e-6 * fs):round(0.8e-6 * fs);
% %     else % Type B
% %         zoom_range = round(0.4e-6 * fs):round(0.6e-6 * fs);
% %     end
% %     plot(t(zoom_range), clean_signal(zoom_range), 'r', 'LineWidth', 2); hold on;
% %     for m = 1:length(methods)
% %         plot(t(zoom_range), denoised_signals{m, s}(zoom_range), 'LineWidth', 1);
% %     end
% %     title([sample_types{s} ': Zoomed Detail']);
% %     legend('Clean', methods{:});
% %     xlabel('Time (s)');
% %     ylabel('Amplitude');
% % end
% % 
% % % 4. Print detailed results table
% % fprintf('SNR Before :%6.2f',snr_before);
% % fprintf('\n=== Detailed Performance Comparison ===\n\n');
% % 
% % for s = 1:length(sample_indices)
% %     fprintf('--- %s Results ---\n', sample_types{s});
% %     fprintf('%-20s | %-10s | %-10s | %-10s\n', 'Method', 'SNR (dB)', 'MSE', 'Corr Coef');
% %     fprintf('----------------------------------------------------\n');
% %     for m = 1:length(methods)
% %         fprintf('%-20s | %10.2f | %10.6f | %10.4f\n', ...
% %             methods{m}, snr_results(m,s), mse_results(m,s), cc_results(m,s));
% %     end
% %     fprintf('\n');
% % end

%% --------------CNN_WAVELET (TYPE A B C D : 10 18 20 25MM)
% % 1. Enhanced Signal Simulation with Types A, B, C, D
% fs = 1000e6;                  % 1 GHz sampling
% t_total = 2e-6;               % 2 μs duration
% t = 0:1/fs:t_total;           % Time vector
% signal_length = length(t);
% num_samples = 2000;
% 
% clean_signals = zeros(num_samples, signal_length);
% noisy_signals = zeros(num_samples, signal_length);
% 
% for i = 1:num_samples
%     % Cycle through Type A, Type B, Type C, and Type D
%     signal_type = mod(i-1, 4) + 1;  % Types 1,2,3,4 correspond to A,B,C,D
% 
%     if signal_type == 1  % Type A: 10mm
%         clean_signal = zeros(size(t));
% 
%         % Very sparse PD events with random locations
%         num_events = 5 + randi(5);  % 3-7 events (very sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Clear, distinct bipolar spikes
%                 amplitude = 2.5 + rand() * 2;  % 2.5-4.5 amplitude
%                 polarity = (-1)^randi([0 1]);  % Random polarity
% 
%                 % Sharp bipolar pulse
%                 spike_width = 3 + randi(3);  % 3-6 samples wide
% 
%                 if start_idx + spike_width - 1 <= length(clean_signal)
%                     % Main spike
%                     clean_signal(start_idx) = polarity * amplitude;
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.8;
%                     end
% 
%                     % Decay tail
%                     for j = 2:spike_width-1
%                         if start_idx + j <= length(clean_signal)
%                             clean_signal(start_idx + j) = polarity * amplitude * 0.3 * exp(-(j-1));
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 2  % Type B: 18mm
%         clean_signal = zeros(size(t));
% 
%         % Sparse PD events with random locations
%         num_events = 55 + randi(10);  % 10-25 events (sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Mixed event types with random characteristics
%                 event_type = rand();
%                 amplitude = 2 + rand() * 3;  % 1.5-4.5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 if event_type < 0.7  % 70% - Sharp bipolar spikes
%                     spike_width = 2 + randi(4);
% 
%                     if start_idx + spike_width - 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         if start_idx + 1 <= length(clean_signal)
%                             clean_signal(start_idx + 1) = -polarity * amplitude * 0.7;
%                         end
% 
%                         % Add some oscillatory tail
%                         for j = 2:spike_width-1
%                             if start_idx + j <= length(clean_signal)
%                                 clean_signal(start_idx + j) = polarity * amplitude * 0.2 * sin(j);
%                             end
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 3  % Type C: 20mm
%         clean_signal = zeros(size(t));
% 
%         % Moderate to high frequency PD events with random locations
%         num_events = 120 + randi(30);  % 40-75 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 2 + rand() * 4;  % 1-5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.8  % 40% - Sharp spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 30% - Multi-frequency transients
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                   end
%                end
%             end
% 
%        else  % Type D: 25mm
%         clean_signal = zeros(size(t));
% 
%         % 25mm events with random locations throughout
%         num_events = 250 + randi(80);  % 100-180 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 3 + rand() * 4.2;  % 0.8-5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.6  % 30% - Quick spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 40% - Complex multi-component events
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                 end
%             end
%         end
%     end
% 
%     % Keep signals in a reasonable range but preserve relative amplitudes
%     max_amplitude = max(abs(clean_signal));
%     if max_amplitude > 0
%         if max_amplitude > 5
%             clean_signal = clean_signal * (5 / max_amplitude);
%         end
%     end
% 
%     % Normalize clean signal
%     clean_signal = clean_signal / max(abs(clean_signal) + eps);  % Avoid division by zero
% 
%     % Add Noise
%     desired_snr = -10; % adjust SNR
%     noisy_signal = awgn(clean_signal, desired_snr, 'measured');
% 
%     % Save memory by using single precision
%     clean_signal = single(clean_signal);
%     noisy_signal = single(noisy_signal);
% 
%     clean_signals(i,:) = clean_signal;
%     noisy_signals(i,:) = noisy_signal;
% end
% 
% % Plot Type A, B, C, D samples
% figure;
% subplot(4,2,1);
% plot(t, clean_signals(1,:));
% title('Type A: 10mm (Clean)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,2);
% plot(t, noisy_signals(1,:));
% title('Type A: 10mm (Noisy)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,3);
% plot(t, clean_signals(2,:));
% title('Type B: 18mm(Clean)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,4);
% plot(t, noisy_signals(2,:));
% title('Type B: 18mm(Noisy)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,5);
% plot(t, clean_signals(3,:));
% title('Type C: 20mm(Clean)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,6);
% plot(t, noisy_signals(3,:));
% title('Type C: 20mm(Noisy)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,7);
% plot(t, clean_signals(4,:));
% title('Type D: 25mm (Clean)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,8);
% plot(t, noisy_signals(4,:));
% title('Type D: 25mm (Noisy)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% %% 2. Improved Data Preparation
% % Don't normalize to range, use standardization instead
% clean_mean = mean(clean_signals(:));
% clean_std = std(clean_signals(:));
% noisy_mean = mean(noisy_signals(:));
% noisy_std = std(noisy_signals(:));
% 
% clean_signals_norm = (clean_signals - clean_mean) / clean_std;
% noisy_signals_norm = (noisy_signals - noisy_mean) / noisy_std;
% 
% % Fixed data split with stratification (equal Type A/B/C/D in each)
% num_type_a = sum(1:4:num_samples <= num_samples);
% num_type_b = sum(2:4:num_samples <= num_samples);
% num_type_c = sum(3:4:num_samples <= num_samples);
% num_type_d = sum(4:4:num_samples <= num_samples);
% 
% % Type indices
% type_a_indices = 1:4:num_samples;
% type_b_indices = 2:4:num_samples;
% type_c_indices = 3:4:num_samples;
% type_d_indices = 4:4:num_samples;
% 
% type_a_train_size = floor(0.7 * num_type_a);
% type_a_val_size = floor(0.15 * num_type_a);
% type_b_train_size = floor(0.7 * num_type_b);
% type_b_val_size = floor(0.15 * num_type_b);
% type_c_train_size = floor(0.7 * num_type_c);
% type_c_val_size = floor(0.15 * num_type_c);
% type_d_train_size = floor(0.7 * num_type_d);
% type_d_val_size = floor(0.15 * num_type_d);
% 
% % Random permutation within types
% type_a_perm = type_a_indices(randperm(length(type_a_indices)));
% type_b_perm = type_b_indices(randperm(length(type_b_indices)));
% type_c_perm = type_c_indices(randperm(length(type_c_indices)));
% type_d_perm = type_d_indices(randperm(length(type_d_indices)));
% 
% % Split
% train_indices = [type_a_perm(1:type_a_train_size), type_b_perm(1:type_b_train_size), ...
%                  type_c_perm(1:type_c_train_size), type_d_perm(1:type_d_train_size)];
% val_indices = [type_a_perm(type_a_train_size+1:type_a_train_size+type_a_val_size), ...
%                type_b_perm(type_b_train_size+1:type_b_train_size+type_b_val_size), ...
%                type_c_perm(type_c_train_size+1:type_c_train_size+type_c_val_size), ...
%                type_d_perm(type_d_train_size+1:type_d_train_size+type_d_val_size)];
% test_indices = [type_a_perm(type_a_train_size+type_a_val_size+1:end), ...
%                 type_b_perm(type_b_train_size+type_b_val_size+1:end), ...
%                 type_c_perm(type_c_train_size+type_c_val_size+1:end), ...
%                 type_d_perm(type_d_train_size+type_d_val_size+1:end)];
% 
% % Reshape for network
% X_train = reshape(noisy_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% Y_train = reshape(clean_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% X_val = reshape(noisy_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% Y_val = reshape(clean_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% X_test = reshape(noisy_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% Y_test = reshape(clean_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% 
% %% 3. IMPROVED CNN Architecture
% layers = [
%     imageInputLayer([signal_length,1,1], 'Name', 'input')
% 
%     % Enhanced encoder with residual-like connections
%     convolution2dLayer([7,1], 64, 'Padding','same', 'Name','conv1')
%     batchNormalizationLayer('Name','bn1')
%     leakyReluLayer(0.1, 'Name','lrelu1')
% 
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','conv2')
%     batchNormalizationLayer('Name','bn2')
%     leakyReluLayer(0.1, 'Name','lrelu2')
% 
%     % Multi-scale feature extraction
%     convolution2dLayer([3,1], 128, 'Padding','same', 'Name','conv3a')
%     batchNormalizationLayer('Name','bn3a')
%     leakyReluLayer(0.1, 'Name','lrelu3a')
% 
%     % Dilated convolutions for larger receptive field
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',2, 'Name','dilated1')
%     batchNormalizationLayer('Name','bn_dilated1')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated1')
%     dropoutLayer(0.1, 'Name','dropout1')
% 
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',4, 'Name','dilated2')
%     batchNormalizationLayer('Name','bn_dilated2')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated2')
%     dropoutLayer(0.1, 'Name','dropout2')
% 
%     % Decoder path with improved architecture
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','decode1')
%     batchNormalizationLayer('Name','bn_decode1')
%     leakyReluLayer(0.1, 'Name','lrelu_decode1')
% 
%     convolution2dLayer([3,1], 32, 'Padding','same', 'Name','decode2')
%     batchNormalizationLayer('Name','bn_decode2')
%     leakyReluLayer(0.1, 'Name','lrelu_decode2')
% 
%     % Final output
%     convolution2dLayer([3,1], 1, 'Padding','same', 'Name','output')
%     regressionLayer()
% ];
% 
% 
% %% 4. Training Configuration 
% options = trainingOptions('adam', ...
%     'MaxEpochs', 300, ...                   
%     'MiniBatchSize', 24, ...                % Slightly reduced for better gradients
%     'InitialLearnRate', 3e-4, ...           % Optimized learning rate
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.6, ...         % More gradual reduction
%     'LearnRateDropPeriod', 25, ...          
%     'L2Regularization', 1.5e-4, ...         % Slightly increased regularization
%     'GradientThreshold', 0.8, ...           % Better gradient clipping
%     'ValidationData', {X_val, Y_val}, ...   
%     'ValidationFrequency', 80, ...          
%     'ValidationPatience', 35, ...           
%     'Shuffle', 'every-epoch', ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');
% 
% %% 5. Helper function for early stopping
% function stop = stopIfAccuracyNotImproving(info, N)
%     stop = false;
% 
%     % Check if validation loss hasn't improved for N iterations
%     if info.State == "iteration" && ~isempty(info.ValidationLoss)
%         % Get validation loss history
%         valLoss = info.ValidationLoss;
% 
%         % Check if we have enough history
%         if numel(valLoss) >= N
%             % Check if validation loss hasn't decreased in last N iterations
%             if all(diff(valLoss(end-N+1:end)) >= 0)
%                 disp('Validation loss not decreasing for multiple iterations. Stopping training.');
%                 stop = true;
%             end
%         end
%     end
% end
% 
% %% 6. Optimal wavelet selection function 
% function wname = selectOptimalWavelet(signal)
%     % Enhanced wavelet selection with better criteria
%     wavelets = {'sym4', 'sym6', 'sym8', 'db4', 'db6', 'db8', 'coif3', 'coif4', 'bior4.4', 'bior6.8'};
%     min_error = inf;
%     best_wavelet = 'sym6'; % Default
% 
%     % Analyze signal characteristics first
%     signal_energy = sum(signal.^2);
%     signal_sparsity = sum(abs(signal) > 0.1*max(abs(signal))) / length(signal);
% 
%     for w = 1:length(wavelets)
%         wname = wavelets{w};
%         try
%             % Adaptive level selection based on signal length and wavelet
%             max_level = wmaxlev(length(signal), wname);
%             level = min(6, max_level);
% 
%             [c, l] = wavedec(signal, level, wname);
% 
%             % Simple reconstruction test
%             rec = waverec(c, l, wname);
% 
%             % Multi-criteria error calculation
%             mse_error = mean((signal - rec).^2);
% 
%             % Energy preservation criterion
%             rec_energy = sum(rec.^2);
%             energy_error = abs(signal_energy - rec_energy) / max(signal_energy, 1e-10);
% 
%             % Peak preservation criterion (important for PD signals)
%             signal_peaks = abs(signal) > 0.5*max(abs(signal));
%             rec_peaks = abs(rec) > 0.5*max(abs(rec));
%             peak_error = sum(abs(signal_peaks - rec_peaks)) / length(signal);
% 
%             % Combined error with weights
%             total_error = mse_error + 0.1*energy_error + 0.05*peak_error;
% 
%             if total_error < min_error
%                 min_error = total_error;
%                 best_wavelet = wname;
%             end
%         catch
%             continue;
%         end
%     end
%     wname = best_wavelet;
% end
% 
% %% 7. Train Network 
% net = trainNetwork(X_train, Y_train, layers, options);
% 
% Y_pred_raw = predict(net, X_test);
% 
% % Convert back to original scale
% Y_pred = Y_pred_raw * clean_std + clean_mean;
% Y_test_orig = Y_test * clean_std + clean_mean; 
% 
% %% 8. Wavelet Post-Processing 
% Y_refined = zeros(size(Y_pred));
% 
% for i = 1:size(Y_pred, 4)
%     sig = squeeze(Y_pred(:,1,1,i));
% 
%     % Determine signal type from test indices
%     test_idx = test_indices(i);
%     signal_type = mod(test_idx-1, 4) + 1;
% 
%     % 1. Enhanced wavelet selection
%     wname = selectOptimalWavelet(sig);
% 
%     % 2. Adaptive decomposition level
%     max_level = wmaxlev(length(sig), wname);
%     if signal_type == 1 || signal_type == 2  % Sparse signals (Type A, B)
%         level = min(5, max_level);  % Conservative for sparse signals
%     else  % Dense signals (Type C, D)
%         level = min(6, max_level);  % Deeper for complex signals
%     end
% 
%     [c, l] = wavedec(sig, level, wname);
%     c_den = c;
% 
%     % 3. Signal-type specific processing
%     for j = 1:level
%         % Get detail coefficients at this level
%         d = detcoef(c, l, j);
% 
%         if length(d) < 3 || all(abs(d) < 1e-10)
%             continue;
%         end
% 
%         % Enhanced noise estimation
%         sigma = median(abs(d))/0.6745;
%         if sigma < 1e-10
%             sigma = std(d) / 8;
%         end
% 
%         % Signal-type adaptive thresholding
%         switch signal_type
%             case 1  % Type A: Very sparse, preserve sharp peaks
%                 % Conservative threshold to preserve sparse events
%                 thr = sigma * sqrt(2*log(length(d))) * 0.6;
% 
%             case 2  % Type B: Sparse, mixed characteristics
%                 % Balanced approach with BayesShrink
%                 var_signal = max(0, var(d) - sigma^2);
%                 if var_signal > 0
%                     thr = (sigma^2) / sqrt(var_signal) * 0.8;
%                 else
%                     thr = sigma * sqrt(2*log(length(d))) * 0.7;
%                 end
% 
%             case 3  % Type C: Moderate frequency, multi-component
%                 % Moderate threshold preserving multi-frequency content
%                 thr = sigma * sqrt(2*log(length(d))) * 0.75;
% 
%             case 4  % Type D: High frequency, complex
%                 % More aggressive for high-frequency noise
%                 thr = sigma * sqrt(2*log(length(d))) * 0.9;
% 
%             otherwise
%                 thr = sigma * sqrt(2*log(length(d))) * 0.8;
%         end
% 
%         % Level-dependent scaling (preserve low frequencies)
%         level_factor = 1.0 - 0.12*(j-1);
%         thr = thr * max(0.4, level_factor);
% 
%         % Apply soft thresholding
%         start_idx = sum(l(1:j)) + 1;
%         end_idx = sum(l(1:j+1));
%         idx = start_idx:end_idx;
% 
%         c_den(idx) = wthresh(c(idx), 's', thr);
% 
%         % For sparse signals, enhance significant coefficients
%         if signal_type <= 2
%             significant_mask = abs(c(idx)) > 1.5*sigma;
%             c_den(idx(significant_mask)) = c_den(idx(significant_mask)) * 1.05;
%         end
%     end
% 
%     % 4. Enhanced approximation coefficient processing
%     if l(1) > 0
%         approx_coef = c(1:l(1));
%         if any(abs(approx_coef) > 1e-10)
%             % Very conservative threshold for approximation
%             sigma_a = std(approx_coef) / 8;
%             c_den(1:l(1)) = wthresh(approx_coef, 's', sigma_a);
%         end
%     end
% 
%     % 5. Reconstruction
%     y_reconstructed = waverec(c_den, l, wname);
% 
%     % 6. Enhanced post-processing to remove artifacts
%     y_final = y_reconstructed;
% 
%     % Artifact detection and correction
%     diff_signal = abs(y_reconstructed - sig);
%     artifact_threshold = mean(diff_signal) + 2.5*std(diff_signal);
%     artifact_indices = find(diff_signal > artifact_threshold);
% 
%     % For each artifact, apply intelligent correction
%     for idx = artifact_indices'
%         if idx > 2 && idx < length(y_final)-1
%             % Check if it's an isolated spike
%             local_region = y_reconstructed(max(1,idx-2):min(length(y_final),idx+2));
%             if abs(y_reconstructed(idx)) > 2*std(local_region)
%                 % Replace with CNN prediction (more reliable for artifacts)
%                 y_final(idx) = sig(idx) * 0.8;
%             end
%         end
%     end
% 
%     % Signal-specific final smoothing
%     if signal_type >= 3  % For complex signals (Type C, D)
%         % Light smoothing to remove high-frequency artifacts
%         kernel = [0.25, 0.5, 0.25];  % Simple smoothing kernel
%         y_smooth = conv(y_final, kernel, 'same');
% 
%         % Apply smoothing only where necessary (high-frequency noise regions)
%         high_freq_mask = abs(diff([y_final; y_final(end)])) > std(diff(sig)) * 2;
%         y_final(high_freq_mask(1:end-1)) = y_smooth(high_freq_mask(1:end-1));
%     end
% 
%     Y_refined(:,1,1,i) = y_final;
% end
% 
% fprintf('Enhanced wavelet post-processing completed.\n');
% %% 9. Evaluation 
% X_test_orig = X_test * noisy_std + noisy_mean;
% 
% clean_power = mean(Y_test_orig(:).^2);
% noise_power = mean((Y_test_orig(:)-X_test_orig(:)).^2);
% snr_before = 10*log10(clean_power/noise_power);
% 
% % CNN output metrics
% residual_cnn = Y_test_orig(:) - Y_pred(:);
% mse_cnn = mean(residual_cnn.^2);
% snr_cnn = 10*log10(clean_power/mean(residual_cnn.^2));
% cc_cnn = corrcoef(Y_test_orig(:), Y_pred(:));
% cc_cnn = cc_cnn(1,2);
% 
% % Wavelet output metrics
% residual_cnn_wavelet = Y_test_orig(:) - Y_refined(:);
% mse_cnn_wavelet = mean(residual_cnn_wavelet.^2);
% snr_cnn_wavelet = 10*log10(clean_power/mean(residual_cnn_wavelet.^2));
% cc_cnn_wavelet = corrcoef(Y_test_orig(:), Y_refined(:));
% cc_cnn_wavelet = cc_cnn_wavelet(1,2);
% 
% % Display comprehensive results
% fprintf('=== Denoising Performance ===\n');
% fprintf('                |  Before  |   CNN   | CNN_Wavelet\n');
% fprintf('-----------------------------------------------------\n');
% fprintf('MSE             |    -     | %.4f  | %.4f\n', mse_cnn, mse_cnn_wavelet);
% fprintf('SNR (dB)        | %6.2f   | %6.2f  | %6.2f\n',snr_before, snr_cnn, snr_cnn_wavelet);
% fprintf('CC              |    -     | %.4f  | %.4f\n', cc_cnn, cc_cnn_wavelet);
% fprintf('Improvement (dB)| -        | %6.2f  | %6.2f\n', ...
%         snr_cnn - snr_before, snr_cnn_wavelet - snr_before);
% 
% %% 10. Evaluate on 10 Random Test Samples
% num_eval_samples = 10;
% random_indices = randperm(length(test_indices), num_eval_samples);
% avg_snr_before = 0;
% avg_mse_cnn = 0;
% avg_mse_cnn_wavelet = 0;
% avg_snr_cnn = 0;
% avg_snr_cnn_wavelet = 0;
% avg_cc_cnn = 0;
% avg_cc_cnn_wavelet = 0;
% 
% figure;
% for idx = 1:num_eval_samples
%     test_idx = random_indices(idx);
% 
%     x = X_test(:,1,1,test_idx);
%     y_true = Y_test(:,1,1,test_idx);
%     y_cnn = Y_pred(:,1,1,test_idx);
%     y_wave = Y_refined(:,1,1,test_idx);
% 
%     % Metrics
%     mse_cnn_i = mean((y_true - y_cnn).^2);
%     mse_cnn_wavelet_i = mean((y_true - y_wave).^2);
% 
%     snr_cnn_i = 10*log10(mean(y_true.^2) / mse_cnn_i);
%     snr_cnn_wavelet_i = 10*log10(mean(y_true.^2) / mean((y_true - y_wave).^2));
% 
%     cc_cnn_i = corrcoef(y_true, y_cnn); cc_cnn_i = cc_cnn_i(1,2);
%     cc_cnn_wavelet_i = corrcoef(y_true, y_wave); cc_cnn_wavelet_i = cc_cnn_wavelet_i(1,2);
% 
%     % Accumulate for average
%     avg_snr_before = avg_snr_before + snr_before;
%     avg_mse_cnn = avg_mse_cnn + mse_cnn_i;
%     avg_mse_cnn_wavelet = avg_mse_cnn_wavelet + mse_cnn_wavelet_i;
%     avg_snr_cnn = avg_snr_cnn + snr_cnn_i;
%     avg_snr_cnn_wavelet = avg_snr_cnn_wavelet + snr_cnn_wavelet_i;
%     avg_cc_cnn = avg_cc_cnn + cc_cnn_i;
%     avg_cc_cnn_wavelet = avg_cc_cnn_wavelet + cc_cnn_wavelet_i;
% 
%     % Plot
%     subplot(num_eval_samples, 4, (idx-1)*4 + 1);
%     plot(t, x); title(sprintf('Noisy #%d', idx)); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 4, (idx-1)*4 + 2);
%     plot(t, y_true); title('Clean'); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 4, (idx-1)*4 + 3);
%     plot(t, y_cnn); title('CNN'); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 4, (idx-1)*4 + 4);
%     plot(t, y_wave); title('CNN-Wavelet'); ylabel('Amplitude');
% end
% 
% % Take averages
% avg_snr_before = avg_snr_before / num_eval_samples;
% avg_mse_cnn = avg_mse_cnn / num_eval_samples;
% avg_mse_cnn_wavelet = avg_mse_cnn_wavelet / num_eval_samples;
% avg_snr_cnn = avg_snr_cnn / num_eval_samples;
% avg_snr_cnn_wavelet = avg_snr_cnn_wavelet / num_eval_samples;
% avg_cc_cnn = avg_cc_cnn / num_eval_samples;
% avg_cc_cnn_wavelet = avg_cc_cnn_wavelet / num_eval_samples;
% 
% % Display results
% fprintf('\n=== 10-Sample Random Evaluation ===\n');
% fprintf('MSE        | CNN: %.4f | CNN_Wavelet: %.4f\n', avg_mse_cnn, avg_mse_cnn_wavelet);
% fprintf('SNR (dB)   | CNN: %.2f   | CNN_Wavelet: %.2f\n', avg_snr_cnn, avg_snr_cnn_wavelet);
% fprintf('Corr Coef  | CNN: %.4f | CNN_Wavelet: %.4f\n', avg_cc_cnn, avg_cc_cnn_wavelet);
% 
% % Compare with overall results
% fprintf('\n=== Comparison with Overall Results ===\n');
% fprintf('                     | Overall Eval | 10-Sample Avg\n');
% fprintf('-----------------------------------------------------\n');
% fprintf('SNR Before (dB)      | %6.2f       | %6.2f\n', snr_before, avg_snr_before);
% fprintf('SNR CNN (dB)         | %6.2f       | %6.2f\n', snr_cnn, avg_snr_cnn);
% fprintf('SNR CNN_Wavelet (dB) | %6.2f       | %6.2f\n', snr_cnn_wavelet, avg_snr_cnn_wavelet);
% fprintf('CC CNN               | %6.4f       | %6.4f\n', cc_cnn, avg_cc_cnn);
% fprintf('CC CNN_Wavelet       | %6.4f       | %6.4f\n', cc_cnn_wavelet, avg_cc_cnn_wavelet);
% 
% %% 11. Visual comparison of denoised signals for Type A, B, C, and D
% % Find representative samples of each type from the test set
% % Type A: indices 1,5,9,... (mod 4 = 1)
% % Type B: indices 2,6,10,... (mod 4 = 2) 
% % Type C: indices 3,7,11,... (mod 4 = 3)
% % Type D: indices 4,8,12,... (mod 4 = 0)
% 
% type_a_indices = find(mod(test_indices-1, 4) == 0);  % Type A (signal_type = 1)
% type_b_indices = find(mod(test_indices-1, 4) == 1);  % Type B (signal_type = 2)
% type_c_indices = find(mod(test_indices-1, 4) == 2);  % Type C (signal_type = 3)
% type_d_indices = find(mod(test_indices-1, 4) == 3);  % Type D (signal_type = 4)
% 
% % Select the first one of each type (if available)
% sample_indices = [];
% sample_types = {};
% 
% if ~isempty(type_a_indices)
%     sample_indices(end+1) = type_a_indices(1);
%     sample_types{end+1} = 'Type A';
% end
% if ~isempty(type_b_indices)
%     sample_indices(end+1) = type_b_indices(1);
%     sample_types{end+1} = 'Type B';
% end
% if ~isempty(type_c_indices)
%     sample_indices(end+1) = type_c_indices(1);
%     sample_types{end+1} = 'Type C';
% end
% if ~isempty(type_d_indices)
%     sample_indices(end+1) = type_d_indices(1);
%     sample_types{end+1} = 'Type D';
% end
% 
% methods = {'CNN', 'CNN-Wavelet'};
% 
% % Create arrays to store the denoised signals for each method and sample
% denoised_signals = cell(length(methods), length(sample_indices));
% 
% for s = 1:length(sample_indices)
%     idx = sample_indices(s);
% 
%     % Store denoised signals from each method for this sample
%     denoised_signals{1, s} = squeeze(Y_pred(:,1,1,idx)) * clean_std + clean_mean;       % CNN
%     denoised_signals{2, s} = squeeze(Y_refined(:,1,1,idx)) * clean_std + clean_mean;    % CNN+Wavelet
% end
% 
% % Create the comparison figure
% figure('Position', [100, 100, 1500, 1000]);
% for s = 1:length(sample_indices)
%     idx = sample_indices(s);
%     noisy_signal = squeeze(X_test(:,1,1,idx)) * noisy_std + noisy_mean;
%     clean_signal = squeeze(Y_test(:,1,1,idx)) * clean_std + clean_mean;
% 
%     % Plot original signals
%     subplot(length(sample_indices), 3, (s-1)*3 + 1);
%     plot(t, noisy_signal, 'r'); hold on;
%     plot(t, clean_signal, 'g');
%     title([sample_types{s} ': Original Signals']);
%     legend('Noisy', 'Clean');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     % Plot all denoised signals
%     subplot(length(sample_indices), 3, (s-1)*3 + 2);
%     plot(t, clean_signal, 'g'); hold on;
%     for m = 1:length(methods)
%         plot(t, denoised_signals{m, s}, 'LineWidth', 1);
%     end
%     title([sample_types{s} ': Both Methods']);
%     legend('Clean', methods{:});
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     % Plot zoomed-in section for detail
%     subplot(length(sample_indices), 3, (s-1)*3 + 3);
%     % Define zoom ranges based on signal type
%     if contains(sample_types{s}, 'A')  % Type A - very sparse
%         zoom_range = round(0.15e-6 * fs):round(0.35e-6 * fs);
%     elseif contains(sample_types{s}, 'B')  % Type B - sparse  
%         zoom_range = round(0.55e-6 * fs):round(0.75e-6 * fs);
%     elseif contains(sample_types{s}, 'C')  % Type C - moderate-high
%         zoom_range = round(0.4e-6 * fs):round(0.6e-6 * fs);
%     else  % Type D - very high frequency
%         zoom_range = round(0.3e-6 * fs):round(0.5e-6 * fs);
%     end
% 
%     plot(t(zoom_range), clean_signal(zoom_range), 'g', 'LineWidth', 2); hold on;
%     for m = 1:length(methods)
%         plot(t(zoom_range), denoised_signals{m, s}(zoom_range), 'LineWidth', 1);
%     end
%     title([sample_types{s} ': Zoomed Detail']);
%     legend('Clean', methods{:});
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% end
% 
% %% 12. Calculate performance metrics for each signal type
% % Get test indices for each type
% type_a_test = test_indices(mod(test_indices-1, 4) == 0);
% type_b_test = test_indices(mod(test_indices-1, 4) == 1);
% type_c_test = test_indices(mod(test_indices-1, 4) == 2);
% type_d_test = test_indices(mod(test_indices-1, 4) == 3);
% 
% % Initialize arrays to store metrics
% num_types = 4;
% type_names = {'Type A', 'Type B', 'Type C', 'Type D'};
% type_test_indices = {type_a_test, type_b_test, type_c_test, type_d_test};
% 
% snr_before_types = zeros(num_types, 1);
% snr_improvements = zeros(num_types, 2); % [type, method]
% cc_values = zeros(num_types, 2); % [type, method]
% mse_values = zeros(num_types, 2); % [type, method]
% 
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
% 
%     if isempty(type_test)
%         fprintf('Warning: No test samples found for %s\n', type_names{type_idx});
%         continue;
%     end
% 
%     % Find indices in the test set
%     type_indices_in_test = find(ismember(test_indices, type_test));
% 
%     if isempty(type_indices_in_test)
%         continue;
%     end
% 
%     % Extract signals for this type
%     x_type = X_test(:,:,:,type_indices_in_test) * noisy_std + noisy_mean;
%     y_type = Y_test(:,:,:,type_indices_in_test) * clean_std + clean_mean;
%     y_pred_type = Y_pred(:,:,:,type_indices_in_test) * clean_std + clean_mean;
%     y_wave_type = Y_refined(:,:,:,type_indices_in_test) * clean_std + clean_mean;
% 
%     % Calculate SNR before denoising
%     clean_power_type = mean(y_type(:).^2);
%     noise_power_type = mean((y_type(:) - x_type(:)).^2);
%     snr_before_types(type_idx) = 10*log10(clean_power_type/noise_power_type);
% 
%     % CNN metrics
%     residual_cnn_type = y_type(:) - y_pred_type(:);
%     mse_cnn_type = mean(residual_cnn_type.^2);
%     snr_cnn_type = 10*log10(clean_power_type/mse_cnn_type);
%     cc_cnn_type = corrcoef(y_type(:), y_pred_type(:));
%     cc_cnn_type = cc_cnn_type(1,2);
% 
%     % CNN+Wavelet metrics
%     residual_wave_type = y_type(:) - y_wave_type(:);
%     mse_wave_type = mean(residual_wave_type.^2);
%     snr_wave_type = 10*log10(clean_power_type/mse_wave_type);
%     cc_wave_type = corrcoef(y_type(:), y_wave_type(:));
%     cc_wave_type = cc_wave_type(1,2);
% 
%     % Store improvements and correlation coefficients
%     snr_improvements(type_idx, 1) = snr_cnn_type - snr_before_types(type_idx);
%     snr_improvements(type_idx, 2) = snr_wave_type - snr_before_types(type_idx);
%     cc_values(type_idx, 1) = cc_cnn_type;
%     cc_values(type_idx, 2) = cc_wave_type;
%     mse_values(type_idx, 1) = mse_cnn_type;
%     mse_values(type_idx, 2) = mse_wave_type;
% end
% 
% % Display comprehensive results by signal type
% fprintf('\n=== Denoising Performance by Signal Type ===\n');
% fprintf('Signal Type | Initial SNR (dB) | Samples\n');
% fprintf('-------------------------------------------\n');
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
%     fprintf('%-11s | %8.2f        | %7d\n', type_names{type_idx}, ...
%         snr_before_types(type_idx), length(type_test));
% end
% 
% fprintf('\n=== SNR Improvement and Correlation by Signal Type ===\n');
% fprintf('Signal Type |                   CNN                |               CNN_Wavelet                |\n');
% fprintf('Evaluation  |     SNR      |     CC    |    MSE    |     SNR      |      CC     |     MSE     |\n');
% fprintf('----------------------------------------------------------------------------------------------\n');
% for type_idx = 1:num_types
%     fprintf('%-11s | %8.2f dB  | %8.4f  | %8.5f  | %8.2f dB  | %10.4f  | %11.5f |\n', ...
%         type_names{type_idx}, snr_improvements(type_idx, 1), cc_values(type_idx, 1),mse_values(type_idx, 1), ...
%         snr_improvements(type_idx, 2), cc_values(type_idx, 2), mse_values(type_idx, 2));
% end
% 
% % Calculate and display average improvements
% avg_snr_improvement_cnn = mean(snr_improvements(:, 1));
% avg_snr_improvement_wavelet = mean(snr_improvements(:, 2));
% avg_cc_cnn = mean(cc_values(:, 1));
% avg_cc_wavelet = mean(cc_values(:, 2));
% 
% fprintf('\n=== Average Performance Across All Types ===\n');
% fprintf('Method      |   Avg SNR    |   Avg CC\n');
% fprintf('------------------------------------\n');
% fprintf('CNN         | %8.2f dB  | %8.4f\n', avg_snr_improvement_cnn, avg_cc_cnn);
% fprintf('CNN+Wavelet | %8.2f dB  | %8.4f\n', avg_snr_improvement_wavelet, avg_cc_wavelet);
% 
% 
% %% 13. Save Results with Error Handling
% try
%     % First, verify that the network exists and is valid
%     if ~exist('net', 'var') || isempty(net)
%         error('Network variable is empty or not defined');
%     end
% 
%     % Save the comprehensive results file
%     save('cnn_wavelet_ABCD_result.mat', 'net', 'Y_pred', 'Y_refined', ...
%          'mse_cnn_wavelet', 'snr_before', 'snr_cnn_wavelet', 'cc_cnn_wavelet', ...
%          'snr_improvements', 'cc_values', 'mse_values', 'type_names', ...
%          'snr_before_types', 'avg_snr_improvement_cnn', 'avg_snr_improvement_wavelet');
%     fprintf('Comprehensive results saved successfully to cnn_wavelet_ABCD_result.mat\n');
% 
%     % Save just the network with verification
%     save('cnn_wavelet_ABCD.mat', 'net');
% 
%     % Verify the saved file
%     fileInfo = dir('cnn_wavelet_ABCD.mat');
%     if isempty(fileInfo) || fileInfo.bytes == 0
%         error('Failed to save network: File is empty');
%     else
%         fprintf('Network saved successfully to cnn_wavelet_4types.mat (%d bytes)\n', fileInfo.bytes);
% 
%         % Double-check by trying to load it
%         testLoad = load('cnn_wavelet_ABCD.mat');
%         if ~isfield(testLoad, 'net')
%             error('Saved file does not contain the network variable');
%         else
%             fprintf('Verified: Network was saved correctly and can be loaded\n');
%         end
%     end
% 
%     % Save detailed performance metrics
%     performance_summary = struct();
%     performance_summary.type_names = type_names;
%     performance_summary.snr_before = snr_before_types;
%     performance_summary.snr_improvements = snr_improvements;
%     performance_summary.correlation_coefficients = cc_values;
%     performance_summary.mse_values = mse_values;
%     performance_summary.avg_snr_improvement_cnn = avg_snr_improvement_cnn;
%     performance_summary.avg_snr_improvement_wavelet = avg_snr_improvement_wavelet;
%     performance_summary.avg_cc_cnn = avg_cc_cnn;
%     performance_summary.avg_cc_wavelet = avg_cc_wavelet;
% 
%     save('performance_summary_ABCD.mat', 'performance_summary');
%     fprintf('Performance summary saved to performance_summary_ABCD.mat\n');
% 
% catch ME
%     fprintf('ERROR saving results: %s\n', ME.message);
%     % Try an alternative save location
%     try
%         alternativePath = fullfile(pwd, 'cnn_wavelet_ABCD_backup.mat');
%         save(alternativePath, 'net');
%         fprintf('Network saved to alternative location: %s\n', alternativePath);
%     catch
%         fprintf('Failed to save to alternative location as well\n');
%     end
% end
% 
% %% 15. Final Summary
% fprintf('\n=== FINAL SUMMARY ===\n');
% fprintf('Dataset: %d samples with 4 signal types (A, B, C, D)\n', num_samples);
% fprintf('Signal length: %d samples (%.1f μs at %.0f MHz)\n', signal_length, t_total*1e6, fs/1e6);
% fprintf('Network: CNN with wavelet post-processing\n');
% fprintf('Training epochs: %d\n', options.MaxEpochs);
% fprintf('\nOverall Performance:\n');
% fprintf('• Average SNR improvement (CNN): %.2f dB\n', avg_snr_improvement_cnn);
% fprintf('• Average SNR improvement (CNN+Wavelet): %.2f dB\n', avg_snr_improvement_wavelet);
% fprintf('• Average correlation (CNN): %.4f\n', avg_cc_cnn);
% fprintf('• Average correlation (CNN+Wavelet): %.4f\n', avg_cc_wavelet);
% fprintf('• Wavelet processing provides additional %.2f dB improvement\n', ...
%         avg_snr_improvement_wavelet - avg_snr_improvement_cnn);
% 
% fprintf('\nBest performing signal types:\n');
% [~, best_cnn_idx] = max(snr_improvements(:, 1));
% [~, best_wavelet_idx] = max(snr_improvements(:, 2));
% fprintf('• CNN: %s (%.2f dB improvement)\n', type_names{best_cnn_idx}, snr_improvements(best_cnn_idx, 1));
% fprintf('• CNN+Wavelet: %s (%.2f dB improvement)\n', type_names{best_wavelet_idx}, snr_improvements(best_wavelet_idx, 2));
% 
% fprintf('\nFiles saved:\n');
% fprintf('• cnn_wavelet_ABCD.mat (trained network)\n');
% fprintf('• cnn_wavelet_result_ABCD.mat (complete results)\n');
% fprintf('• performance_summary_ABCD.mat (performance metrics)\n');
% fprintf('\n=== ANALYSIS COMPLETE ===\n');

%% --------------CNN_WAVELET (TYPE A B C D E F)
% % Type A: Sparse PD pulses (4 pulses with gaps)
% % Type B: Spike-dense signal (realistic sharp pulses with gaps)
% % Type C: 10mm (very sparse PD events)
% % Type D: 18mm (sparse PD events)
% % Type E: 20mm (moderate-high frequency PD events)
% % Type F: 25mm (high frequency, complex PD events)
% 
% %% 1. Enhanced Signal Simulation with Types A, B, C, D, E, F
% fs = 1000e6;                  % 1 GHz sampling
% t_total = 2e-6;               % 2 μs duration
% t = 0:1/fs:t_total;           % Time vector
% signal_length = length(t);
% num_samples = 1000;           % Increased for 6 types
% 
% clean_signals = zeros(num_samples, signal_length);
% noisy_signals = zeros(num_samples, signal_length);
% 
% for i = 1:num_samples
%     % Cycle through Type A, Type B, Type C, Type D, Type E, Type F
%     signal_type = mod(i-1, 6) + 1;  % Types 1,2,3,4,5,6 correspond to A,B,C,D,E,F
% 
%     if signal_type == 1  % Type A: Sparse PD pulses (4 pulses with gaps)
%         clean_signal = zeros(size(t));
%         start_times = [0.2e-6, 0.6e-6, 1.2e-6, 1.6e-6]; % Clear time gaps
% 
%         for k = 1:length(start_times)
%             A = 10 + rand()*10;
%             fc = 25e6 + rand()*10e6; % Higher freq helps sharpen
%             tau = 0.01e-6 + rand()*0.03e-6; % Very short pulse
%             pulse_t = t - start_times(k);
%             pulse_t = pulse_t(pulse_t >= 0);
% 
%             % Generate a short pulse with fewer points
%             pulse_duration = 0.05e-6; % 50 ns duration
%             pulse_t = pulse_t(pulse_t <= pulse_duration);
%             pulse = A * exp(-pulse_t/tau) .* sin(2*pi*fc*pulse_t);
% 
%             % Insert pulse at the correct position
%             start_idx = find(t >= start_times(k), 1);
%             pulse_len = length(pulse);
%             if start_idx + pulse_len - 1 <= length(clean_signal)
%                 clean_signal(start_idx:start_idx + pulse_len - 1) = pulse;
%             end
%         end
% 
%     elseif signal_type == 2  % Type B: Spike-dense signal (realistic sharp pulses with gaps)
%         clean_signal = zeros(size(t));
%         num_spikes = 20 + randi(10); % Fewer, more realistic pulses
%         spike_len = 2; % Length of each biphasic spike
% 
%         for s = 1:num_spikes
%             start_idx = randi([1, signal_length - spike_len]);
%             amp = 0.5 + 0.5*rand(); % Amplitude
%             direction = (-1)^randi([0 1]); % Flip polarity randomly
% 
%             % Biphasic spike: [positive, negative] or [negative, positive]
%             clean_signal(start_idx) = direction * amp;
%             clean_signal(start_idx + 1) = -direction * amp;
%         end
% 
%     elseif signal_type == 3  % Type C: 10mm
%         clean_signal = zeros(size(t));
% 
%         % Very sparse PD events with random locations
%         num_events = 5 + randi(5);  % 5-10 events (very sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Clear, distinct bipolar spikes
%                 amplitude = 2.5 + rand() * 2;  % 2.5-4.5 amplitude
%                 polarity = (-1)^randi([0 1]);  % Random polarity
% 
%                 % Sharp bipolar pulse
%                 spike_width = 3 + randi(3);  % 3-6 samples wide
% 
%                 if start_idx + spike_width - 1 <= length(clean_signal)
%                     % Main spike
%                     clean_signal(start_idx) = polarity * amplitude;
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.8;
%                     end
% 
%                     % Decay tail
%                     for j = 2:spike_width-1
%                         if start_idx + j <= length(clean_signal)
%                             clean_signal(start_idx + j) = polarity * amplitude * 0.3 * exp(-(j-1));
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 4  % Type D: 18mm
%         clean_signal = zeros(size(t));
% 
%         % Sparse PD events with random locations
%         num_events = 55 + randi(10);  % 55-65 events (sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Mixed event types with random characteristics
%                 event_type = rand();
%                 amplitude = 2 + rand() * 3;  % 2-5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 if event_type < 0.7  % 70% - Sharp bipolar spikes
%                     spike_width = 2 + randi(4);
% 
%                     if start_idx + spike_width - 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         if start_idx + 1 <= length(clean_signal)
%                             clean_signal(start_idx + 1) = -polarity * amplitude * 0.7;
%                         end
% 
%                         % Add some oscillatory tail
%                         for j = 2:spike_width-1
%                             if start_idx + j <= length(clean_signal)
%                                 clean_signal(start_idx + j) = polarity * amplitude * 0.2 * sin(j);
%                             end
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 5  % Type E: 20mm
%         clean_signal = zeros(size(t));
% 
%         % Moderate to high frequency PD events with random locations
%         num_events = 120 + randi(30);  % 120-150 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 2 + rand() * 4;  % 2-6 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.6  % 60% - Sharp spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 40% - Multi-frequency transients
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                 end
%             end
%         end
% 
%     else  % Type F: 25mm
%         clean_signal = zeros(size(t));
% 
%         % 25mm events with random locations throughout
%         num_events = 250 + randi(80);  % 250-330 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 3 + rand() * 4.2;  % 3-7.2 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.4  % 40% - Quick spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 60% - Complex multi-component events
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                 end
%             end
%         end
%     end
% 
%     % Keep signals in a reasonable range but preserve relative amplitudes
%     max_amplitude = max(abs(clean_signal));
%     if max_amplitude > 0
%         if max_amplitude > 5
%             clean_signal = clean_signal * (5 / max_amplitude);
%         end
%     end
% 
%     % Normalize clean signal
%     clean_signal = clean_signal / max(abs(clean_signal) + eps);  % Avoid division by zero
% 
%     % Add Noise (matching your original noise model)
%     white_noise = 0.08*randn(size(clean_signal));
%     powerline_noise = 0.025*sin(2*pi*50e6*t) + 0.015*sin(2*pi*150e6*t);
%     narrowband = 0.03*sin(2*pi*80e6*t + rand()*2*pi);
%     impulse_noise = zeros(size(clean_signal));
%     spike_pos = randperm(length(clean_signal), 15);
%     impulse_noise(spike_pos) = 0.4*(0.2 + 0.8*rand(1,15));
%     noise = white_noise + powerline_noise + narrowband + impulse_noise;
% 
%     % Adjust SNR
%     current_snr = 10*log10(var(clean_signal) / (var(noise) + eps));
%     desired_snr = -10 + rand()*8;
%     noise = noise * 10^((current_snr-desired_snr)/20);
%     noisy_signal = clean_signal + noise;
% 
%     % Save memory by using single precision
%     clean_signal = single(clean_signal);
%     noisy_signal = single(noisy_signal);
% 
%     clean_signals(i,:) = clean_signal;
%     noisy_signals(i,:) = noisy_signal;
% end
% 
% % Plot Type A-F samples
% figure('Position', [100, 100, 1600, 1200]);
% type_names = {'Type A: Sparse PD pulses', 'Type B: Spike-dense signal', ...
%               'Type C: 10mm', 'Type D: 18mm', 'Type E: 20mm', 'Type F: 25mm'};
% 
% for type_idx = 1:6
%     subplot(6,2,(type_idx-1)*2+1);
%     plot(t, clean_signals(type_idx,:));
%     title([type_names{type_idx} ' (Clean)']);
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     subplot(6,2,(type_idx-1)*2+2);
%     plot(t, noisy_signals(type_idx,:));
%     title([type_names{type_idx} ' (Noisy)']);
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% end
% 
% %% 2. Improved Data Preparation
% % Don't normalize to range, use standardization instead
% clean_mean = mean(clean_signals(:));
% clean_std = std(clean_signals(:));
% noisy_mean = mean(noisy_signals(:));
% noisy_std = std(noisy_signals(:));
% 
% clean_signals_norm = (clean_signals - clean_mean) / clean_std;
% noisy_signals_norm = (noisy_signals - noisy_mean) / noisy_std;
% 
% % Fixed data split with stratification (equal Type A/B/C/D/E/F in each)
% num_type_a = sum(1:6:num_samples <= num_samples);
% num_type_b = sum(2:6:num_samples <= num_samples);
% num_type_c = sum(3:6:num_samples <= num_samples);
% num_type_d = sum(4:6:num_samples <= num_samples);
% num_type_e = sum(5:6:num_samples <= num_samples);
% num_type_f = sum(6:6:num_samples <= num_samples);
% 
% % Type indices
% type_a_indices = 1:6:num_samples;
% type_b_indices = 2:6:num_samples;
% type_c_indices = 3:6:num_samples;
% type_d_indices = 4:6:num_samples;
% type_e_indices = 5:6:num_samples;
% type_f_indices = 6:6:num_samples;
% 
% type_a_train_size = floor(0.7 * num_type_a);
% type_a_val_size = floor(0.15 * num_type_a);
% type_b_train_size = floor(0.7 * num_type_b);
% type_b_val_size = floor(0.15 * num_type_b);
% type_c_train_size = floor(0.7 * num_type_c);
% type_c_val_size = floor(0.15 * num_type_c);
% type_d_train_size = floor(0.7 * num_type_d);
% type_d_val_size = floor(0.15 * num_type_d);
% type_e_train_size = floor(0.7 * num_type_e);
% type_e_val_size = floor(0.15 * num_type_e);
% type_f_train_size = floor(0.7 * num_type_f);
% type_f_val_size = floor(0.15 * num_type_f);
% 
% % Random permutation within types
% type_a_perm = type_a_indices(randperm(length(type_a_indices)));
% type_b_perm = type_b_indices(randperm(length(type_b_indices)));
% type_c_perm = type_c_indices(randperm(length(type_c_indices)));
% type_d_perm = type_d_indices(randperm(length(type_d_indices)));
% type_e_perm = type_e_indices(randperm(length(type_e_indices)));
% type_f_perm = type_f_indices(randperm(length(type_f_indices)));
% 
% % Split
% train_indices = [type_a_perm(1:type_a_train_size), type_b_perm(1:type_b_train_size), ...
%                  type_c_perm(1:type_c_train_size), type_d_perm(1:type_d_train_size), ...
%                  type_e_perm(1:type_e_train_size), type_f_perm(1:type_f_train_size)];
% val_indices = [type_a_perm(type_a_train_size+1:type_a_train_size+type_a_val_size), ...
%                type_b_perm(type_b_train_size+1:type_b_train_size+type_b_val_size), ...
%                type_c_perm(type_c_train_size+1:type_c_train_size+type_c_val_size), ...
%                type_d_perm(type_d_train_size+1:type_d_train_size+type_d_val_size), ...
%                type_e_perm(type_e_train_size+1:type_e_train_size+type_e_val_size), ...
%                type_f_perm(type_f_train_size+1:type_f_train_size+type_f_val_size)];
% test_indices = [type_a_perm(type_a_train_size+type_a_val_size+1:end), ...
%                 type_b_perm(type_b_train_size+type_b_val_size+1:end), ...
%                 type_c_perm(type_c_train_size+type_c_val_size+1:end), ...
%                 type_d_perm(type_d_train_size+type_d_val_size+1:end), ...
%                 type_e_perm(type_e_train_size+type_e_val_size+1:end), ...
%                 type_f_perm(type_f_train_size+type_f_val_size+1:end)];
% 
% % Reshape for network
% X_train = reshape(noisy_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% Y_train = reshape(clean_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% X_val = reshape(noisy_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% Y_val = reshape(clean_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% X_test = reshape(noisy_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% Y_test = reshape(clean_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% 
% %% 3. IMPROVED CNN Architecture
% layers = [
%     imageInputLayer([signal_length,1,1], 'Name', 'input')
% 
%     % Enhanced encoder with residual-like connections
%     convolution2dLayer([7,1], 64, 'Padding','same', 'Name','conv1')
%     batchNormalizationLayer('Name','bn1')
%     leakyReluLayer(0.1, 'Name','lrelu1')
% 
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','conv2')
%     batchNormalizationLayer('Name','bn2')
%     leakyReluLayer(0.1, 'Name','lrelu2')
% 
%     % Multi-scale feature extraction
%     convolution2dLayer([3,1], 128, 'Padding','same', 'Name','conv3a')
%     batchNormalizationLayer('Name','bn3a')
%     leakyReluLayer(0.1, 'Name','lrelu3a')
% 
%     % Dilated convolutions for larger receptive field
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',2, 'Name','dilated1')
%     batchNormalizationLayer('Name','bn_dilated1')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated1')
%     dropoutLayer(0.1, 'Name','dropout1')
% 
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',4, 'Name','dilated2')
%     batchNormalizationLayer('Name','bn_dilated2')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated2')
%     dropoutLayer(0.1, 'Name','dropout2')
% 
%     % Decoder path with improved architecture
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','decode1')
%     batchNormalizationLayer('Name','bn_decode1')
%     leakyReluLayer(0.1, 'Name','lrelu_decode1')
% 
%     convolution2dLayer([3,1], 32, 'Padding','same', 'Name','decode2')
%     batchNormalizationLayer('Name','bn_decode2')
%     leakyReluLayer(0.1, 'Name','lrelu_decode2')
% 
%     % Final output
%     convolution2dLayer([3,1], 1, 'Padding','same', 'Name','output')
%     regressionLayer()
% ];
% 
% %% 4. Training Configuration 
% options = trainingOptions('adam', ...
%     'MaxEpochs', 1, ...                   
%     'MiniBatchSize', 24, ...                % Slightly reduced for better gradients
%     'InitialLearnRate', 3e-4, ...           % Optimized learning rate
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.6, ...         % More gradual reduction
%     'LearnRateDropPeriod', 25, ...          
%     'L2Regularization', 1.5e-4, ...         % Slightly increased regularization
%     'GradientThreshold', 0.8, ...           % Better gradient clipping
%     'ValidationData', {X_val, Y_val}, ...   
%     'ValidationFrequency', 80, ...          
%     'ValidationPatience', 35, ...           
%     'Shuffle', 'every-epoch', ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');
% 
% %% 5. Optimal wavelet selection function 
% function wname = selectOptimalWavelet(signal)
%     % Enhanced wavelet selection with better criteria
%     wavelets = {'sym4', 'sym6', 'sym8', 'db4', 'db6', 'db8', 'coif3', 'coif4', 'bior4.4', 'bior6.8'};
%     min_error = inf;
%     best_wavelet = 'sym6'; % Default
% 
%     % Analyze signal characteristics first
%     signal_energy = sum(signal.^2);
%     signal_sparsity = sum(abs(signal) > 0.1*max(abs(signal))) / length(signal);
% 
%     for w = 1:length(wavelets)
%         wname = wavelets{w};
%         try
%             % Adaptive level selection based on signal length and wavelet
%             max_level = wmaxlev(length(signal), wname);
%             level = min(6, max_level);
% 
%             [c, l] = wavedec(signal, level, wname);
% 
%             % Simple reconstruction test
%             rec = waverec(c, l, wname);
% 
%             % Multi-criteria error calculation
%             mse_error = mean((signal - rec).^2);
% 
%             % Energy preservation criterion
%             rec_energy = sum(rec.^2);
%             energy_error = abs(signal_energy - rec_energy) / max(signal_energy, 1e-10);
% 
%             % Peak preservation criterion (important for PD signals)
%             signal_peaks = abs(signal) > 0.5*max(abs(signal));
%             rec_peaks = abs(rec) > 0.5*max(abs(rec));
%             peak_error = sum(abs(signal_peaks - rec_peaks)) / length(signal);
% 
%             % Combined error with weights
%             total_error = mse_error + 0.1*energy_error + 0.05*peak_error;
% 
%             if total_error < min_error
%                 min_error = total_error;
%                 best_wavelet = wname;
%             end
%         catch
%             continue;
%         end
%     end
%     wname = best_wavelet;
% end
% 
% %% 6. Train Network 
% net = trainNetwork(X_train, Y_train, layers, options);
% 
% Y_pred_raw = predict(net, X_test);
% 
% % Convert back to original scale
% Y_pred = Y_pred_raw * clean_std + clean_mean;
% Y_test_orig = Y_test * clean_std + clean_mean; 
% 
% %% 6. Plot Results - Random Samples of Type A-F (FIXED)
% % Find Type A-F samples in test set
% type_a_test_indices = find(mod(test_indices-1, 6) == 0);  % Type A samples
% type_b_test_indices = find(mod(test_indices-1, 6) == 1);  % Type B samples
% type_c_test_indices = find(mod(test_indices-1, 6) == 2);  % Type C samples
% type_d_test_indices = find(mod(test_indices-1, 6) == 3);  % Type D samples
% type_e_test_indices = find(mod(test_indices-1, 6) == 4);  % Type E samples
% type_f_test_indices = find(mod(test_indices-1, 6) == 5);  % Type F samples
% 
% % Store all test indices for each type
% all_type_test_indices = {type_a_test_indices, type_b_test_indices, type_c_test_indices, ...
%                          type_d_test_indices, type_e_test_indices, type_f_test_indices};
% type_labels = {'A', 'B', 'C', 'D', 'E', 'F'};
% 
% % Plot one random sample from each type
% for type_idx = 1:6
%     type_test_indices_current = all_type_test_indices{type_idx};
% 
%     if ~isempty(type_test_indices_current)
%         % Select a random position from this type's test indices
%         random_position = type_test_indices_current(randi(length(type_test_indices_current)));
% 
%         % Extract signals (convert back to original scale)
%         clean_signal = squeeze(Y_test(:,1,1,random_position)) * clean_std + clean_mean;
%         noisy_signal = squeeze(X_test(:,1,1,random_position)) * noisy_std + noisy_mean;
%         denoised_signal = squeeze(Y_pred(:,1,1,random_position));
% 
%         % Ensure vectors are the same orientation as t
%         clean_signal = clean_signal(:)';    % Convert to row vector
%         noisy_signal = noisy_signal(:)';    % Convert to row vector
%         denoised_signal = denoised_signal(:)'; % Convert to row vector
% 
%         % Create plot for this type
%         figure;
%         subplot(3,1,1); plot(t, clean_signal); 
%         title(['Type ' type_labels{type_idx} ' - Clean (' type_names{type_idx} ')']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
% 
%         subplot(3,1,2); plot(t, noisy_signal); 
%         title(['Type ' type_labels{type_idx} ' - Noisy']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
% 
%         subplot(3,1,3); plot(t, denoised_signal); 
%         title(['Type ' type_labels{type_idx} ' - Denoised (CNN)']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
%     end
% end
% 
% 
% %% 7. Wavelet Post-Processing 
% Y_refined = zeros(size(Y_pred));
% 
% for i = 1:size(Y_pred, 4)
%     sig = squeeze(Y_pred(:,1,1,i));
% 
%     % Determine signal type from test indices
%     test_idx = test_indices(i);
%     signal_type = mod(test_idx-1, 6) + 1;  % Updated for 6 types
% 
%     % 1. Enhanced wavelet selection
%     wname = selectOptimalWavelet(sig);
% 
%     % 2. Adaptive decomposition level
%     max_level = wmaxlev(length(sig), wname);
%     if signal_type <= 2  % Sparse signals (Type A, B)
%         level = min(5, max_level);  % Conservative for sparse signals
%     elseif signal_type <= 4  % Moderate signals (Type C, D)
%         level = min(6, max_level);  % Moderate for intermediate signals
%     else  % Dense signals (Type E, F)
%         level = min(7, max_level);  % Deeper for complex signals
%     end
% 
%     [c, l] = wavedec(sig, level, wname);
%     c_den = c;
% 
%     % 3. Signal-type specific processing
%     for j = 1:level
%         % Get detail coefficients at this level
%         d = detcoef(c, l, j);
% 
%         if length(d) < 3 || all(abs(d) < 1e-10)
%             continue;
%         end
% 
%         % Enhanced noise estimation
%         sigma = median(abs(d))/0.6745;
%         if sigma < 1e-10
%             sigma = std(d) / 8;
%         end
% 
%         % Signal-type adaptive thresholding
%         switch signal_type
%             case 1  % Type A: Sparse PD pulses
%                 % Conservative threshold to preserve sparse events
%                 thr = sigma * sqrt(2*log(length(d))) * 0.6;
% 
%             case 2  % Type B: Spike-dense signal
%                 % Balanced approach with BayesShrink
%                 var_signal = max(0, var(d) - sigma^2);
%                 if var_signal > 0
%                     thr = (sigma^2) / sqrt(var_signal) * 0.8;
%                 else
%                     thr = sigma * sqrt(2*log(length(d))) * 0.7;
%                 end
% 
%             case 3  % Type C: 10mm
%                 % Conservative threshold to preserve sparse events
%                 thr = sigma * sqrt(2*log(length(d))) * 0.65;
% 
%             case 4  % Type D: 18mm
%                 % Moderate threshold preserving mixed characteristics
%                 thr = sigma * sqrt(2*log(length(d))) * 0.75;
% 
%             case 5  % Type E: 20mm
%                 % Moderate threshold preserving multi-frequency content
%                 thr = sigma * sqrt(2*log(length(d))) * 0.8;
% 
%             case 6  % Type F: 25mm
%                 % More aggressive for high-frequency noise
%                 thr = sigma * sqrt(2*log(length(d))) * 0.9;
% 
%             otherwise
%                 thr = sigma * sqrt(2*log(length(d))) * 0.8;
%         end
% 
%         % Level-dependent scaling (preserve low frequencies)
%         level_factor = 1.0 - 0.12*(j-1);
%         thr = thr * max(0.4, level_factor);
% 
%         % Apply soft thresholding
%         start_idx = sum(l(1:j)) + 1;
%         end_idx = sum(l(1:j+1));
%         idx = start_idx:end_idx;
% 
%         c_den(idx) = wthresh(c(idx), 's', thr);
% 
%         % For sparse signals, enhance significant coefficients
%         if signal_type <= 2
%             significant_mask = abs(c(idx)) > 1.5*sigma;
%             c_den(idx(significant_mask)) = c_den(idx(significant_mask)) * 1.05;
%         end
%     end
% 
%     % 4. Enhanced approximation coefficient processing
%     if l(1) > 0
%         approx_coef = c(1:l(1));
%         if any(abs(approx_coef) > 1e-10)
%             % Very conservative threshold for approximation
%             sigma_a = std(approx_coef) / 8;
%             c_den(1:l(1)) = wthresh(approx_coef, 's', sigma_a);
%         end
%     end
% 
%     % 5. Reconstruction
%     y_reconstructed = waverec(c_den, l, wname);
% 
%     % 6. Enhanced post-processing to remove artifacts
%     y_final = y_reconstructed;
% 
%     % Artifact detection and correction
%     diff_signal = abs(y_reconstructed - sig);
%     artifact_threshold = mean(diff_signal) + 2.5*std(diff_signal);
%     artifact_indices = find(diff_signal > artifact_threshold);
% 
%     % For each artifact, apply intelligent correction
%     for idx = artifact_indices'
%         if idx > 2 && idx < length(y_final)-1
%             % Check if it's an isolated spike
%             local_region = y_reconstructed(max(1,idx-2):min(length(y_final),idx+2));
%             if abs(y_reconstructed(idx)) > 2*std(local_region)
%                 % Replace with CNN prediction (more reliable for artifacts)
%                 y_final(idx) = sig(idx) * 0.8;
%             end
%         end
%     end
% 
%     % Signal-specific final smoothing
%     if signal_type >= 5  % For complex signals (Type E, F)
%         % Light smoothing to remove high-frequency artifacts
%         kernel = [0.25, 0.5, 0.25];  % Simple smoothing kernel
%         y_smooth = conv(y_final, kernel, 'same');
% 
%         % Apply smoothing only where necessary (high-frequency noise regions)
%         high_freq_mask = abs(diff([y_final; y_final(end)])) > std(diff(sig)) * 2;
%         y_final(high_freq_mask(1:end-1)) = y_smooth(high_freq_mask(1:end-1));
%     end
% 
%     Y_refined(:,1,1,i) = y_final;
% end
% 
% fprintf('Enhanced wavelet post-processing completed for 6 signal types.\n');
% 
% %% 8. Evaluation 
% X_test_orig = X_test * noisy_std + noisy_mean;
% 
% clean_power = mean(Y_test_orig(:).^2);
% noise_power = mean((Y_test_orig(:)-X_test_orig(:)).^2);
% snr_before = 10*log10(clean_power/noise_power);
% 
% % CNN output metrics
% residual_cnn = Y_test_orig(:) - Y_pred(:);
% mse_cnn = mean(residual_cnn.^2);
% snr_cnn = 10*log10(clean_power/mean(residual_cnn.^2));
% cc_cnn = corrcoef(Y_test_orig(:), Y_pred(:));
% cc_cnn = cc_cnn(1,2);
% 
% % Wavelet output metrics
% residual_cnn_wavelet = Y_test_orig(:) - Y_refined(:);
% mse_cnn_wavelet = mean(residual_cnn_wavelet.^2);
% snr_cnn_wavelet = 10*log10(clean_power/mean(residual_cnn_wavelet.^2));
% cc_cnn_wavelet = corrcoef(Y_test_orig(:), Y_refined(:));
% cc_cnn_wavelet = cc_cnn_wavelet(1,2);
% 
% % Display comprehensive results
% fprintf('=== Denoising Performance ===\n');
% fprintf('                |  Before  |   CNN   | CNN_Wavelet\n');
% fprintf('-----------------------------------------------------\n');
% fprintf('MSE             |    -     | %.4f  | %.4f\n', mse_cnn, mse_cnn_wavelet);
% fprintf('SNR (dB)        | %6.2f   | %6.2f  | %6.2f\n',snr_before, snr_cnn, snr_cnn_wavelet);
% fprintf('CC              |    -     | %.4f  | %.4f\n', cc_cnn, cc_cnn_wavelet);
% fprintf('Improvement (dB)| -        | %6.2f  | %6.2f\n', ...
%         snr_cnn - snr_before, snr_cnn_wavelet - snr_before);
% 
% %% 9. Evaluate on 10 Random Test Samples
% num_eval_samples = 10;
% random_indices = randperm(length(test_indices), num_eval_samples);
% avg_snr_before = 0;
% avg_mse_cnn = 0;
% avg_mse_cnn_wavelet = 0;
% avg_snr_cnn = 0;
% avg_snr_cnn_wavelet = 0;
% avg_cc_cnn = 0;
% avg_cc_cnn_wavelet = 0;
% 
% figure;
% for idx = 1:num_eval_samples
%     test_idx = random_indices(idx);
% 
%     x = X_test(:,1,1,test_idx);
%     y_true = Y_test(:,1,1,test_idx);
%     y_cnn = Y_pred(:,1,1,test_idx);
%     y_wave = Y_refined(:,1,1,test_idx);
% 
%     % Metrics
%     mse_cnn_i = mean((y_true - y_cnn).^2);
%     mse_cnn_wavelet_i = mean((y_true - y_wave).^2);
% 
%     snr_cnn_i = 10*log10(mean(y_true.^2) / mse_cnn_i);
%     snr_cnn_wavelet_i = 10*log10(mean(y_true.^2) / mean((y_true - y_wave).^2));
% 
%     cc_cnn_i = corrcoef(y_true, y_cnn); cc_cnn_i = cc_cnn_i(1,2);
%     cc_cnn_wavelet_i = corrcoef(y_true, y_wave); cc_cnn_wavelet_i = cc_cnn_wavelet_i(1,2);
% 
%     % Accumulate for average
%     avg_snr_before = avg_snr_before + snr_before;
%     avg_mse_cnn = avg_mse_cnn + mse_cnn_i;
%     avg_mse_cnn_wavelet = avg_mse_cnn_wavelet + mse_cnn_wavelet_i;
%     avg_snr_cnn = avg_snr_cnn + snr_cnn_i;
%     avg_snr_cnn_wavelet = avg_snr_cnn_wavelet + snr_cnn_wavelet_i;
%     avg_cc_cnn = avg_cc_cnn + cc_cnn_i;
%     avg_cc_cnn_wavelet = avg_cc_cnn_wavelet + cc_cnn_wavelet_i;
% 
%     % Plot
%     subplot(num_eval_samples, 4, (idx-1)*4 + 1);
%     plot(t, x); title(sprintf('Noisy #%d', idx)); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 4, (idx-1)*4 + 2);
%     plot(t, y_true); title('Clean'); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 4, (idx-1)*4 + 3);
%     plot(t, y_cnn); title('CNN'); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 4, (idx-1)*4 + 4);
%     plot(t, y_wave); title('CNN-Wavelet'); ylabel('Amplitude');
% end
% 
% % Take averages
% avg_snr_before = avg_snr_before / num_eval_samples;
% avg_mse_cnn = avg_mse_cnn / num_eval_samples;
% avg_mse_cnn_wavelet = avg_mse_cnn_wavelet / num_eval_samples;
% avg_snr_cnn = avg_snr_cnn / num_eval_samples;
% avg_snr_cnn_wavelet = avg_snr_cnn_wavelet / num_eval_samples;
% avg_cc_cnn = avg_cc_cnn / num_eval_samples;
% avg_cc_cnn_wavelet = avg_cc_cnn_wavelet / num_eval_samples;
% 
% % Display results
% fprintf('\n=== 10-Sample Random Evaluation ===\n');
% fprintf('MSE        | CNN: %.4f | CNN_Wavelet: %.4f\n', avg_mse_cnn, avg_mse_cnn_wavelet);
% fprintf('SNR (dB)   | CNN: %.2f   | CNN_Wavelet: %.2f\n', avg_snr_cnn, avg_snr_cnn_wavelet);
% fprintf('Corr Coef  | CNN: %.4f | CNN_Wavelet: %.4f\n', avg_cc_cnn, avg_cc_cnn_wavelet);

% 
% %% 10. Calculate performance metrics for each signal type (A-F)
% % Get test indices for each type
% type_a_test = test_indices(mod(test_indices-1, 6) == 0);
% type_b_test = test_indices(mod(test_indices-1, 6) == 1);
% type_c_test = test_indices(mod(test_indices-1, 6) == 2);
% type_d_test = test_indices(mod(test_indices-1, 6) == 3);
% type_e_test = test_indices(mod(test_indices-1, 6) == 4);
% type_f_test = test_indices(mod(test_indices-1, 6) == 5);
% 
% % Initialize arrays to store metrics
% num_types = 6;
% type_names_full = {'Type A: Sparse PD pulses', 'Type B: Spike-dense signal', ...
%                    'Type C: 10mm', 'Type D: 18mm', 'Type E: 20mm', 'Type F: 25mm'};
% type_test_indices = {type_a_test, type_b_test, type_c_test, type_d_test, type_e_test, type_f_test};
% 
% snr_before_types = zeros(num_types, 1);
% snr_improvements = zeros(num_types, 2); % [type, method]
% cc_values = zeros(num_types, 2); % [type, method]
% mse_values = zeros(num_types, 2); % [type, method]
% 
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
% 
%     if isempty(type_test)
%         fprintf('Warning: No test samples found for %s\n', type_names_full{type_idx});
%         continue;
%     end
% 
%     % Find indices in the test set
%     type_indices_in_test = find(ismember(test_indices, type_test));
% 
%     if isempty(type_indices_in_test)
%         continue;
%     end
% 
%     % Extract signals for this type
%     x_type = X_test(:,:,:,type_indices_in_test) * noisy_std + noisy_mean;
%     y_type = Y_test(:,:,:,type_indices_in_test) * clean_std + clean_mean;
%     y_pred_type = Y_pred(:,:,:,type_indices_in_test) * clean_std + clean_mean;
%     y_wave_type = Y_refined(:,:,:,type_indices_in_test) * clean_std + clean_mean;
% 
%     % Calculate SNR before denoising
%     clean_power_type = mean(y_type(:).^2);
%     noise_power_type = mean((y_type(:) - x_type(:)).^2);
%     snr_before_types(type_idx) = 10*log10(clean_power_type/noise_power_type);
% 
%     % CNN metrics
%     residual_cnn_type = y_type(:) - y_pred_type(:);
%     mse_cnn_type = mean(residual_cnn_type.^2);
%     snr_cnn_type = 10*log10(clean_power_type/mse_cnn_type);
%     cc_cnn_type = corrcoef(y_type(:), y_pred_type(:));
%     cc_cnn_type = cc_cnn_type(1,2);
% 
%     % CNN+Wavelet metrics
%     residual_wave_type = y_type(:) - y_wave_type(:);
%     mse_wave_type = mean(residual_wave_type.^2);
%     snr_wave_type = 10*log10(clean_power_type/mse_wave_type);
%     cc_wave_type = corrcoef(y_type(:), y_wave_type(:));
%     cc_wave_type = cc_wave_type(1,2);
% 
%     % Store improvements and correlation coefficients
%     snr_improvements(type_idx, 1) = snr_cnn_type - snr_before_types(type_idx);
%     snr_improvements(type_idx, 2) = snr_wave_type - snr_before_types(type_idx);
%     cc_values(type_idx, 1) = cc_cnn_type;
%     cc_values(type_idx, 2) = cc_wave_type;
%     mse_values(type_idx, 1) = mse_cnn_type;
%     mse_values(type_idx, 2) = mse_wave_type;
% end
% 
% % Display comprehensive results by signal type
% fprintf('\n=== Denoising Performance by Signal Type ===\n');
% fprintf('Signal Type                | Initial SNR (dB) | Samples\n');
% fprintf('----------------------------------------------------------\n');
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
%     fprintf('%-26s | %8.2f        | %7d\n', type_names_full{type_idx}, ...
%         snr_before_types(type_idx), length(type_test));
% end
% 
% fprintf('\n=== SNR Improvement and Correlation by Signal Type ===\n');
% fprintf('Signal Type                |                   CNN                |               CNN_Wavelet                |\n');
% fprintf('                           |     SNR      |     CC    |    MSE    |     SNR      |      CC     |     MSE     |\n');
% fprintf('-------------------------------------------------------------------------------------------------------------\n');
% for type_idx = 1:num_types
%     fprintf('%-26s | %8.2f dB  | %8.4f  | %8.5f  | %8.2f dB  | %10.4f  | %11.5f |\n', ...
%         type_names_full{type_idx}, snr_improvements(type_idx, 1), cc_values(type_idx, 1),mse_values(type_idx, 1), ...
%         snr_improvements(type_idx, 2), cc_values(type_idx, 2), mse_values(type_idx, 2));
% end
% 
% % Calculate and display average improvements
% avg_snr_improvement_cnn = mean(snr_improvements(:, 1));
% avg_snr_improvement_wavelet = mean(snr_improvements(:, 2));
% avg_cc_cnn_all = mean(cc_values(:, 1));
% avg_cc_wavelet_all = mean(cc_values(:, 2));
% 
% fprintf('\n=== Average Performance Across All Types ===\n');
% fprintf('Method      |   Avg SNR    |   Avg CC\n');
% fprintf('------------------------------------\n');
% fprintf('CNN         | %8.2f dB  | %8.4f\n', avg_snr_improvement_cnn, avg_cc_cnn_all);
% fprintf('CNN+Wavelet | %8.2f dB  | %8.4f\n', avg_snr_improvement_wavelet, avg_cc_wavelet_all);
% 
% % Find representative samples of each type from the test set
% type_a_indices = find(mod(test_indices-1, 6) == 0);  % Type A
% type_b_indices = find(mod(test_indices-1, 6) == 1);  % Type B
% type_c_indices = find(mod(test_indices-1, 6) == 2);  % Type C
% type_d_indices = find(mod(test_indices-1, 6) == 3);  % Type D
% type_e_indices = find(mod(test_indices-1, 6) == 4);  % Type E
% type_f_indices = find(mod(test_indices-1, 6) == 5);  % Type F
% 
% sample_indices = [];
% sample_types = {};
% all_type_indices = {type_a_indices, type_b_indices, type_c_indices, ...
%                     type_d_indices, type_e_indices, type_f_indices};
% 
% for type_idx = 1:6
%     if ~isempty(all_type_indices{type_idx})
%         sample_indices(end+1) = all_type_indices{type_idx}(1);
%         sample_types{end+1} = ['Type ' type_labels{type_idx}];
%     end
% end
% 
% % Split into 2 figures: A-C and D-F
% types_per_figure = 3;
% 
% for fig_num = 1:2
%     figure('Position', [100 + (fig_num-1)*50, 100 + (fig_num-1)*50, 1200, 800]);
% 
%     start_type = (fig_num-1) * types_per_figure + 1;
%     end_type = min(fig_num * types_per_figure, length(sample_indices));
% 
%     for local_s = 1:(end_type - start_type + 1)
%         s = start_type + local_s - 1;
%         idx = sample_indices(s);
% 
%         % Extract and denormalize signals
%         noisy_signal = squeeze(X_test(:,1,1,idx)) * noisy_std + noisy_mean;
%         clean_signal = squeeze(Y_test(:,1,1,idx)) * clean_std + clean_mean;
%         cnn_wavelet_signal = squeeze(Y_refined(:,1,1,idx)) * clean_std + clean_mean;
% 
%         % Convert to row vectors
%         noisy_signal = noisy_signal(:)';
%         clean_signal = clean_signal(:)';
%         cnn_wavelet_signal = cnn_wavelet_signal(:)';
% 
%         % 1. Noisy vs Clean
%         subplot(types_per_figure, 3, (local_s-1)*3 + 1);
%         plot(t, noisy_signal, 'r'); hold on;
%         plot(t, clean_signal, 'g');
%         title([sample_types{s} ': Original Signals']);
%         legend('Noisy', 'Clean');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
% 
%         % 2. CNN+Wavelet vs Clean
%         subplot(types_per_figure, 3, (local_s-1)*3 + 2);
%         plot(t, clean_signal, 'g'); hold on;
%         plot(t, cnn_wavelet_signal, 'm', 'LineWidth', 1);
%         title([sample_types{s} ': CNN+Wavelet Denoising']);
%         legend('Clean', 'CNN+Wavelet');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
% 
%         % 3. Zoomed-in comparison
%         subplot(types_per_figure, 3, (local_s-1)*3 + 3);
% 
%         % Define zoom range
%         if contains(sample_types{s}, 'A')
%             zoom_range = round(0.15e-6 * fs):round(0.35e-6 * fs);
%         elseif contains(sample_types{s}, 'B')
%             zoom_range = round(0.55e-6 * fs):round(0.75e-6 * fs);
%         elseif contains(sample_types{s}, 'C')
%             zoom_range = round(0.2e-6 * fs):round(0.4e-6 * fs);
%         elseif contains(sample_types{s}, 'D')
%             zoom_range = round(0.4e-6 * fs):round(0.6e-6 * fs);
%         elseif contains(sample_types{s}, 'E')
%             zoom_range = round(0.3e-6 * fs):round(0.5e-6 * fs);
%         else  % Type F
%             zoom_range = round(0.2e-6 * fs):round(0.4e-6 * fs);
%         end
% 
%         % Ensure zoom range is within bounds
%         zoom_range = zoom_range(zoom_range >= 1 & zoom_range <= length(t));
% 
%         plot(t(zoom_range), clean_signal(zoom_range), 'g', 'LineWidth', 2); hold on;
%         plot(t(zoom_range), cnn_wavelet_signal(zoom_range), 'm', 'LineWidth', 1);
%         title([sample_types{s} ': Zoomed Detail']);
%         legend('Clean', 'CNN-Wavelet');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
%     end
% 
%     % Figure-wide title
%     if fig_num == 1
%         sgtitle('Signal Comparison - Types A, B, C', 'FontSize', 16, 'FontWeight', 'bold');
%     else
%         sgtitle('Signal Comparison - Types D, E, F', 'FontSize', 16, 'FontWeight', 'bold');
%     end
% end
% 
% %% 11. Visual comparison of denoised signals for Type A-F - Split into 2 figures
% % Find representative samples of each type from the test set
% type_sample_indices = [];
% type_sample_names = {};
% 
% for type_idx = 1:6
%     type_test = type_test_indices{type_idx};
%     if ~isempty(type_test)
%         % Find first sample of this type in test set
%         type_indices_in_test = find(ismember(test_indices, type_test));
%         if ~isempty(type_indices_in_test)
%             type_sample_indices(end+1) = type_indices_in_test(1);
%             type_sample_names{end+1} = type_names_full{type_idx};
%         end
%     end
% end
% 
% methods = {'CNN', 'CNN-Wavelet'};
% 
% % Create arrays to store the denoised signals for each method and sample
% denoised_signals_6types = cell(length(methods), length(type_sample_indices));
% 
% for s = 1:length(type_sample_indices)
%     idx = type_sample_indices(s);
% 
%     % Store denoised signals from each method for this sample
%     denoised_signals_6types{1, s} = squeeze(Y_pred(:,1,1,idx)) * clean_std + clean_mean;       % CNN
%     denoised_signals_6types{2, s} = squeeze(Y_refined(:,1,1,idx)) * clean_std + clean_mean;    % CNN+Wavelet
% end
% 
% % Split into 2 figures: 3 types each
% types_per_figure = 3;
% 
% for fig_num = 1:2
%     figure('Position', [150 + (fig_num-1)*50, 150 + (fig_num-1)*50, 1800, 900]);
% 
%     start_type = (fig_num-1) * types_per_figure + 1;
%     end_type = min(fig_num * types_per_figure, length(type_sample_indices));
% 
%     for local_type = 1:(end_type - start_type + 1)
%         s = start_type + local_type - 1;
%         idx = type_sample_indices(s);
%         noisy_signal = squeeze(X_test(:,1,1,idx)) * noisy_std + noisy_mean;
%         clean_signal = squeeze(Y_test(:,1,1,idx)) * clean_std + clean_mean;
% 
%         % Plot original signals
%         subplot(types_per_figure, 3, (local_type-1)*3 + 1);
%         plot(t, noisy_signal, 'r'); hold on;
%         plot(t, clean_signal, 'g');
%         title([type_sample_names{s} ': Original Signals']);
%         legend('Noisy', 'Clean');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
% 
%         % Plot all denoised signals
%         subplot(types_per_figure, 3, (local_type-1)*3 + 2);
%         plot(t, clean_signal, 'g'); hold on;
%         for m = 1:length(methods)
%             plot(t, denoised_signals_6types{m, s}, 'LineWidth', 1);
%         end
%         title([type_sample_names{s} ': Both Methods']);
%         legend('Clean', methods{:});
%         xlabel('Time (s)');
%         ylabel('Amplitude');
% 
%         % Plot zoomed-in section for detail
%         subplot(types_per_figure, 3, (local_type-1)*3 + 3);
%         % Define zoom ranges based on signal type
%         if contains(type_sample_names{s}, 'Type A')  % Sparse PD pulses
%             zoom_range = round(0.15e-6 * fs):round(0.35e-6 * fs);
%         elseif contains(type_sample_names{s}, 'Type B')  % Spike-dense
%             zoom_range = round(0.4e-6 * fs):round(0.6e-6 * fs);
%         elseif contains(type_sample_names{s}, 'Type C')  % 10mm - very sparse
%             zoom_range = round(0.2e-6 * fs):round(0.4e-6 * fs);
%         elseif contains(type_sample_names{s}, 'Type D')  % 18mm - sparse  
%             zoom_range = round(0.55e-6 * fs):round(0.75e-6 * fs);
%         elseif contains(type_sample_names{s}, 'Type E')  % 20mm - moderate-high
%             zoom_range = round(0.4e-6 * fs):round(0.6e-6 * fs);
%         else  % Type F - 25mm - very high frequency
%             zoom_range = round(0.3e-6 * fs):round(0.5e-6 * fs);
%         end
% 
%         plot(t(zoom_range), clean_signal(zoom_range), 'g', 'LineWidth', 2); hold on;
%         for m = 1:length(methods)
%             plot(t(zoom_range), denoised_signals_6types{m, s}(zoom_range), 'LineWidth', 1);
%         end
%         title([type_sample_names{s} ': Zoomed Detail']);
%         legend('Clean', methods{:});
%         xlabel('Time (s)');
%         ylabel('Amplitude');
%     end
% 
%     % Add figure title
%     if fig_num == 1
%         sgtitle('Signal Type Comparison - Types A, B, C', 'FontSize', 16, 'FontWeight', 'bold');
%     else
%         sgtitle('Signal Type Comparison - Types D, E, F', 'FontSize', 16, 'FontWeight', 'bold');
%     end
% end
% 
% %% 12. Save Results 
% try
%     % First, verify that the network exists and is valid
%     if ~exist('net', 'var') || isempty(net)
%         error('Network variable is empty or not defined');
%     end
% 
%     % Save the comprehensive results file
%     save('cnn_wavelet_ABCDEF_v3_result.mat', 'net', 'Y_pred', 'Y_refined', ...
%          'mse_cnn_wavelet', 'snr_before', 'snr_cnn_wavelet', 'cc_cnn_wavelet', ...
%          'snr_improvements', 'cc_values', 'mse_values', 'type_names_full', ...
%          'snr_before_types', 'avg_snr_improvement_cnn', 'avg_snr_improvement_wavelet');
%     fprintf('Comprehensive results saved successfully to cnn_wavelet_ABCDEF_v3_result.mat\n');
% 
%     % Save just the network with verification
%     save('cnn_wavelet_ABCDEF_v3.mat', 'net');
% 
%     % Verify the saved file
%     fileInfo = dir('cnn_wavelet_ABCDEF_v3.mat');
%     if isempty(fileInfo) || fileInfo.bytes == 0
%         error('Failed to save network: File is empty');
%     else
%         fprintf('Network saved successfully to cnn_wavelet_ABCDEF_v3.mat (%d bytes)\n', fileInfo.bytes);
% 
%         % Double-check by trying to load it
%         testLoad = load('cnn_wavelet_ABCDEF_v3.mat');
%         if ~isfield(testLoad, 'net')
%             error('Saved file does not contain the network variable');
%         else
%             fprintf('Verified: Network was saved correctly and can be loaded\n');
%         end
%     end
% 
%     % Save detailed performance metrics
%     performance_summary = struct();
%     performance_summary.type_names = type_names_full;
%     performance_summary.snr_before = snr_before_types;
%     performance_summary.snr_improvements = snr_improvements;
%     performance_summary.correlation_coefficients = cc_values;
%     performance_summary.mse_values = mse_values;
%     performance_summary.avg_snr_improvement_cnn = avg_snr_improvement_cnn;
%     performance_summary.avg_snr_improvement_wavelet = avg_snr_improvement_wavelet;
%     performance_summary.avg_cc_cnn = avg_cc_cnn_all;
%     performance_summary.avg_cc_wavelet = avg_cc_wavelet_all;
% 
%     save('performance_summary_ABCDEF.mat', 'performance_summary');
%     fprintf('Performance summary saved to performance_summary_ABCDEF.mat\n');
% 
% catch ME
%     fprintf('ERROR saving results: %s\n', ME.message);
%     % Try an alternative save location
%     try
%         alternativePath = fullfile(pwd, 'cnn_wavelet_ABCDEF_v3_backup.mat');
%         save(alternativePath, 'net');
%         fprintf('Network saved to alternative location: %s\n', alternativePath);
%     catch
%         fprintf('Failed to save to alternative location as well\n');
%     end
% end
% 
% %% 13. Final Summary
% fprintf('\n=== FINAL SUMMARY ===\n');
% fprintf('Dataset: %d samples with 6 signal types (A, B, C, D, E, F)\n', num_samples);
% fprintf('Signal length: %d samples (%.1f μs at %.0f MHz)\n', signal_length, t_total*1e6, fs/1e6);
% fprintf('Network: CNN with wavelet post-processing\n');
% fprintf('Training epochs: %d\n', options.MaxEpochs);
% fprintf('\nSignal Types:\n');
% for i = 1:6
%     fprintf('• %s\n', type_names_full{i});
% end
% fprintf('\nOverall Performance:\n');
% fprintf('• Average SNR improvement (CNN): %.2f dB\n', avg_snr_improvement_cnn);
% fprintf('• Average SNR improvement (CNN+Wavelet): %.2f dB\n', avg_snr_improvement_wavelet);
% fprintf('• Average correlation (CNN): %.4f\n', avg_cc_cnn_all);
% fprintf('• Average correlation (CNN+Wavelet): %.4f\n', avg_cc_wavelet_all);
% fprintf('• Wavelet processing provides additional %.2f dB improvement\n', ...
%         avg_snr_improvement_wavelet - avg_snr_improvement_cnn);
% 
% fprintf('\nBest performing signal types:\n');
% [~, best_cnn_idx] = max(snr_improvements(:, 1));
% [~, best_wavelet_idx] = max(snr_improvements(:, 2));
% fprintf('• CNN: %s (%.2f dB improvement)\n', type_names_full{best_cnn_idx}, snr_improvements(best_cnn_idx, 1));
% fprintf('• CNN+Wavelet: %s (%.2f dB improvement)\n', type_names_full{best_wavelet_idx}, snr_improvements(best_wavelet_idx, 2));
% 
% fprintf('\nFiles saved:\n');
% fprintf('• cnn_wavelet_ABCDEF_v3.mat (trained network)\n');
% fprintf('• cnn_wavelet_ABCDEF_v3_result.mat (complete results)\n');
% fprintf('• performance_summary_ABCDEF.mat (performance metrics)\n');
% fprintf('\n=== ANALYSIS COMPLETE ===\n');

%% Only CNN

%% --------------CNN_ONLY (TYPE A B）------------------

% % 1. Enhanced Signal Simulation with Types A and B Only
% fs = 1000e6;                  % 1 GHz sampling
% t_total = 2e-6;               % 2 μs duration
% t = 0:1/fs:t_total;           % Time vector
% signal_length = length(t);
% num_samples = 1000;
% 
% clean_signals = zeros(num_samples, signal_length);
% noisy_signals = zeros(num_samples, signal_length);
% 
% for i = 1:num_samples
%     % Alternate between Type A and Type B samples
%     if mod(i, 2) == 1  % Type A: Sparse PD pulses (4 pulses with gaps)
%         clean_signal = zeros(size(t));
%         start_times = [0.2e-6, 0.6e-6, 1.2e-6, 1.6e-6];  % Clear time gaps
% 
%         for k = 1:length(start_times)
%             A = 10 + rand()*10;
%             fc = 25e6 + rand()*10e6;                    % Higher freq helps sharpen
%             tau = 0.01e-6 + rand()*0.03e-6;             % Very short pulse
%             pulse_t = t - start_times(k);
%             pulse_t = pulse_t(pulse_t >= 0);
% 
%             % Generate a short pulse with fewer points
%             pulse_duration = 0.05e-6;  % 50 ns duration
%             pulse_t = pulse_t(pulse_t <= pulse_duration);
%             pulse = A * exp(-pulse_t/tau) .* sin(2*pi*fc*pulse_t);
% 
%             % Insert pulse at the correct position
%             start_idx = find(t >= start_times(k), 1);
%             pulse_len = length(pulse);
%             if start_idx + pulse_len - 1 <= length(clean_signal)
%                 clean_signal(start_idx:start_idx + pulse_len - 1) = pulse;
%             end
%         end
% 
%     else  % Type B: Spike-dense signal (realistic sharp pulses with gaps)
%         clean_signal = zeros(size(t));
%         num_spikes = 20 + randi(10);  % Fewer, more realistic pulses
%         spike_len = 2;  % Length of each biphasic spike
% 
%         for s = 1:num_spikes
%             start_idx = randi([1, signal_length - spike_len]);
%             amp = 0.5 + 0.5*rand();  % Amplitude
%             direction = (-1)^randi([0 1]);  % Flip polarity randomly
% 
%             % Biphasic spike: [positive, negative] or [negative, positive]
%             clean_signal(start_idx) = direction * amp;
%             clean_signal(start_idx + 1) = -direction * amp;
%         end
%     end
% 
%     % Normalize clean signal
%     clean_signal = clean_signal / max(abs(clean_signal) + eps);  % Avoid division by zero
% 
%     % Noise: White + powerline + narrowband + impulse (with reduced levels)
%     white_noise = 0.08*randn(size(clean_signal));
%     powerline_noise = 0.025*sin(2*pi*50e6*t) + 0.015*sin(2*pi*150e6*t);
%     narrowband = 0.03*sin(2*pi*80e6*t + rand()*2*pi);
%     impulse_noise = zeros(size(clean_signal));
%     spike_pos = randperm(length(clean_signal), 15);
%     impulse_noise(spike_pos) = 0.4*(0.2 + 0.8*rand(1,15));
% 
%     noise = white_noise + powerline_noise + narrowband + impulse_noise;
% 
%     % Adjust SNR (Improved control)
%     current_snr = 10*log10(var(clean_signal) / (var(noise) + eps));
%     desired_snr = -10 + rand()*8;  % 
%     noise = noise * 10^((current_snr-desired_snr)/20);
% 
%     noisy_signal = clean_signal + noise;
% 
%     clean_signals(i,:) = clean_signal;
%     noisy_signals(i,:) = noisy_signal;
% end
% 
% % Plot Type A and B samples
% figure;
% subplot(2,1,1);
% plot(t, clean_signals(1,:));
% title('Type A: Clean Signal (4 Pulses with Gaps)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(2,1,2);
% plot(t, clean_signals(2,:));
% title('Type B: Clean Signal (Simulated Real)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% figure;
% subplot(2,1,1);
% plot(t, noisy_signals(1,:));
% title('Type A : Noisy Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(2,1,2);
% plot(t, noisy_signals(2,:));
% title('Type B : Noisy Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% %% 2. Improved Data Preparation
% % Don't normalize to range, use standardization instead
% clean_mean = mean(clean_signals(:));
% clean_std = std(clean_signals(:));
% noisy_mean = mean(noisy_signals(:));
% noisy_std = std(noisy_signals(:));
% 
% clean_signals_norm = (clean_signals - clean_mean) / clean_std;
% noisy_signals_norm = (noisy_signals - noisy_mean) / noisy_std;
% 
% % Fixed data split with stratification (equal Type A/B in each)
% num_type_a = sum(1:2:num_samples <= num_samples);
% num_type_b = sum(2:2:num_samples <= num_samples);
% 
% % Type indices
% type_a_indices = 1:2:num_samples;
% type_b_indices = 2:2:num_samples;
% 
% type_a_train_size = floor(0.7 * num_type_a);
% type_a_val_size = floor(0.15 * num_type_a);
% type_b_train_size = floor(0.7 * num_type_b);
% type_b_val_size = floor(0.15 * num_type_b);
% 
% % Random permutation within types
% type_a_perm = type_a_indices(randperm(length(type_a_indices)));
% type_b_perm = type_b_indices(randperm(length(type_b_indices)));
% 
% % Split
% train_indices = [type_a_perm(1:type_a_train_size), type_b_perm(1:type_b_train_size)];
% val_indices = [type_a_perm(type_a_train_size+1:type_a_train_size+type_a_val_size), ...
%                type_b_perm(type_b_train_size+1:type_b_train_size+type_b_val_size)];
% test_indices = [type_a_perm(type_a_train_size+type_a_val_size+1:end), ...
%                 type_b_perm(type_b_train_size+type_b_val_size+1:end)];
% 
% % Reshape for network
% X_train = reshape(noisy_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% Y_train = reshape(clean_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% X_val = reshape(noisy_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% Y_val = reshape(clean_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% X_test = reshape(noisy_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% Y_test = reshape(clean_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% 
% %% 3. CNN Architecture
% layers = [
%     imageInputLayer([signal_length,1,1], 'Name', 'input')
% 
%     % Enhanced encoder with residual-like connections
%     convolution2dLayer([7,1], 64, 'Padding','same', 'Name','conv1')
%     batchNormalizationLayer('Name','bn1')
%     leakyReluLayer(0.1, 'Name','lrelu1')
% 
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','conv2')
%     batchNormalizationLayer('Name','bn2')
%     leakyReluLayer(0.1, 'Name','lrelu2')
% 
%     % Multi-scale feature extraction
%     convolution2dLayer([3,1], 128, 'Padding','same', 'Name','conv3a')
%     batchNormalizationLayer('Name','bn3a')
%     leakyReluLayer(0.1, 'Name','lrelu3a')
% 
%     % Dilated convolutions for larger receptive field
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',2, 'Name','dilated1')
%     batchNormalizationLayer('Name','bn_dilated1')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated1')
%     dropoutLayer(0.1, 'Name','dropout1')
% 
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',4, 'Name','dilated2')
%     batchNormalizationLayer('Name','bn_dilated2')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated2')
%     dropoutLayer(0.1, 'Name','dropout2')
% 
%     % Decoder path with improved architecture
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','decode1')
%     batchNormalizationLayer('Name','bn_decode1')
%     leakyReluLayer(0.1, 'Name','lrelu_decode1')
% 
%     convolution2dLayer([3,1], 32, 'Padding','same', 'Name','decode2')
%     batchNormalizationLayer('Name','bn_decode2')
%     leakyReluLayer(0.1, 'Name','lrelu_decode2')
% 
%     % Final output
%     convolution2dLayer([3,1], 1, 'Padding','same', 'Name','output')
%     regressionLayer()
% ];
% 
% %% 4. Training Configuration 
% options = trainingOptions('adam', ...
%     'MaxEpochs', 1, ...                   % Significantly increased
%     'MiniBatchSize', 24, ...                
%     'InitialLearnRate', 2e-4, ...           % Slightly lower initial rate
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.75, ...        % More gradual decay
%     'LearnRateDropPeriod', 50, ...          % Less frequent drops
%     'L2Regularization', 1.5e-4, ...         
%     'GradientThreshold', 0.8, ...           
%     'ValidationData', {X_val, Y_val}, ...   
%     'ValidationFrequency', 120, ...         % Check validation less frequently
%     'ValidationPatience', 60, ...           % Much more patient
%     'Shuffle', 'every-epoch', ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');
% 
% %% 5. Train Network 
% net = trainNetwork(X_train, Y_train, layers, options);
% 
% Y_pred_raw = predict(net, X_test);
% 
% % Convert back to original scale
% Y_pred = Y_pred_raw * clean_std + clean_mean;
% Y_test_orig = Y_test * clean_std + clean_mean; 
% 
% %% 6. Plot Results - Random Samples of Type A and B (FIXED)
% % Find Type A and Type B samples in test set
% type_a_test_indices = find(mod(test_indices-1, 2) == 0);  % Type A samples
% type_b_test_indices = find(mod(test_indices-1, 2) == 1);  % Type B samples
% 
% % Select random samples from each type (CORRECTED LOGIC)
% if ~isempty(type_a_test_indices)
%     % Directly select a random position from Type A test indices
%     type_a_test_position = type_a_test_indices(randi(length(type_a_test_indices)));
% 
%     % Extract signals (convert back to original scale)
%     clean_signal_A = squeeze(Y_test(:,1,1,type_a_test_position)) * clean_std + clean_mean;
%     noisy_signal_A = squeeze(X_test(:,1,1,type_a_test_position)) * noisy_std + noisy_mean;
%     denoised_signal_A = squeeze(Y_pred(:,1,1,type_a_test_position));
% 
%     % Ensure vectors are the same orientation as t
%     clean_signal_A = clean_signal_A(:)';    % Convert to row vector
%     noisy_signal_A = noisy_signal_A(:)';    % Convert to row vector  
%     denoised_signal_A = denoised_signal_A(:)'; % Convert to row vector
% 
%     % Type A Plot
%     figure;
%     subplot(3,1,1); plot(t, clean_signal_A); title('Type A - Clean'); xlabel('Time (s)'); ylabel('Amplitude');
%     subplot(3,1,2); plot(t, noisy_signal_A); title('Type A - Noisy'); xlabel('Time (s)'); ylabel('Amplitude');
%     subplot(3,1,3); plot(t, denoised_signal_A); title('Type A - Denoised'); xlabel('Time (s)'); ylabel('Amplitude');
% end
% 
% if ~isempty(type_b_test_indices)
%     % Directly select a random position from Type B test indices
%     type_b_test_position = type_b_test_indices(randi(length(type_b_test_indices)));
% 
%     % Extract signals (convert back to original scale)
%     clean_signal_B = squeeze(Y_test(:,1,1,type_b_test_position)) * clean_std + clean_mean;
%     noisy_signal_B = squeeze(X_test(:,1,1,type_b_test_position)) * noisy_std + noisy_mean;
%     denoised_signal_B = squeeze(Y_pred(:,1,1,type_b_test_position));
% 
%     % Ensure vectors are the same orientation as t
%     clean_signal_B = clean_signal_B(:)';    % Convert to row vector
%     noisy_signal_B = noisy_signal_B(:)';    % Convert to row vector
%     denoised_signal_B = denoised_signal_B(:)'; % Convert to row vector
% 
%     % Type B Plot
%     figure;
%     subplot(3,1,1); plot(t, clean_signal_B); title('Type B - Clean'); xlabel('Time (s)'); ylabel('Amplitude');
%     subplot(3,1,2); plot(t, noisy_signal_B); title('Type B - Noisy'); xlabel('Time (s)'); ylabel('Amplitude');
%     subplot(3,1,3); plot(t, denoised_signal_B); title('Type B - Denoised'); xlabel('Time (s)'); ylabel('Amplitude');
% end

% %% 7. Evaluation 
% X_test_orig = X_test * noisy_std + noisy_mean;
% 
% clean_power = mean(Y_test_orig(:).^2);
% noise_power = mean((Y_test_orig(:)-X_test_orig(:)).^2);
% snr_before = 10*log10(clean_power/noise_power);
% 
% % CNN output metrics
% residual_cnn = Y_test_orig(:) - Y_pred(:);
% mse_cnn = mean(residual_cnn.^2);
% snr_cnn = 10*log10(clean_power/mean(residual_cnn.^2));
% cc_cnn = corrcoef(Y_test_orig(:), Y_pred(:));
% cc_cnn = cc_cnn(1,2);
% 
% % Display comprehensive results
% fprintf('=== CNN Denoising Performance ===\n');
% fprintf('                |  Before  |   CNN   \n');
% fprintf('------------------------------------\n');
% fprintf('MSE             |    -     | %.4f  \n', mse_cnn);
% fprintf('SNR (dB)        | %6.2f   | %6.2f  \n',snr_before, snr_cnn);
% fprintf('CC              |    -     | %.4f  \n', cc_cnn);
% fprintf('Improvement (dB)| -        | %6.2f  \n', snr_cnn - snr_before);
% 
% %% 8. Evaluate on 10 Random Test Samples
% num_eval_samples = 10;
% random_indices = randperm(length(test_indices), num_eval_samples);
% avg_snr_before = 0;
% avg_mse_cnn = 0;
% avg_snr_cnn = 0;
% avg_cc_cnn = 0;
% 
% figure;
% for idx = 1:num_eval_samples
%     test_idx = random_indices(idx);
% 
%     x = X_test(:,1,1,test_idx);
%     y_true = Y_test(:,1,1,test_idx);
%     y_cnn = Y_pred(:,1,1,test_idx);
% 
%     % Metrics
%     mse_cnn_i = mean((y_true - y_cnn).^2);
%     snr_cnn_i = 10*log10(mean(y_true.^2) / mse_cnn_i);
%     cc_cnn_i = corrcoef(y_true, y_cnn); cc_cnn_i = cc_cnn_i(1,2);
% 
%     % Accumulate for average
%     avg_snr_before = avg_snr_before + snr_before;
%     avg_mse_cnn = avg_mse_cnn + mse_cnn_i;
%     avg_snr_cnn = avg_snr_cnn + snr_cnn_i;
%     avg_cc_cnn = avg_cc_cnn + cc_cnn_i;
% 
%     % Plot
%     subplot(num_eval_samples, 3, (idx-1)*3 + 1);
%     plot(t, x); title(sprintf('Noisy #%d', idx)); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 3, (idx-1)*3 + 2);
%     plot(t, y_true); title('Clean'); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 3, (idx-1)*3 + 3);
%     plot(t, y_cnn); title('CNN'); ylabel('Amplitude');
% end
% 
% % Take averages
% avg_snr_before = avg_snr_before / num_eval_samples;
% avg_mse_cnn = avg_mse_cnn / num_eval_samples;
% avg_snr_cnn = avg_snr_cnn / num_eval_samples;
% avg_cc_cnn = avg_cc_cnn / num_eval_samples;
% 
% % Display results
% fprintf('\n=== 10-Sample Random Evaluation ===\n');
% fprintf('MSE        | CNN: %.4f\n', avg_mse_cnn);
% fprintf('SNR (dB)   | CNN: %.2f\n', avg_snr_cnn);
% fprintf('Corr Coef  | CNN: %.4f\n', avg_cc_cnn);
% 
% % Compare with overall results
% fprintf('\n=== Comparison with Overall Results ===\n');
% fprintf('                     | Overall Eval | 10-Sample Avg\n');
% fprintf('-----------------------------------------------------\n');
% fprintf('SNR Before (dB)      | %6.2f       | %6.2f\n', snr_before, avg_snr_before);
% fprintf('SNR CNN (dB)         | %6.2f       | %6.2f\n', snr_cnn, avg_snr_cnn);
% fprintf('CC CNN               | %6.4f       | %6.4f\n', cc_cnn, avg_cc_cnn);
% 
% %% 9. Visual comparison of denoised signals for Type A and B (FIXED)
% % Find representative samples of each type from the test set
% type_a_indices = find(mod(test_indices-1, 2) == 0);  % Type A (signal_type = 1)
% type_b_indices = find(mod(test_indices-1, 2) == 1);  % Type B (signal_type = 2)
% 
% % Select the first one of each type (if available) - CORRECTED
% sample_indices = [];
% sample_types = {};
% 
% if ~isempty(type_a_indices)
%     sample_indices(end+1) = type_a_indices(1);  % Use the position directly
%     sample_types{end+1} = 'Type A';
% end
% if ~isempty(type_b_indices)
%     sample_indices(end+1) = type_b_indices(1);  % Use the position directly
%     sample_types{end+1} = 'Type B';
% end
% 
% % Create the comparison figure
% figure('Position', [100, 100, 1200, 600]);
% for s = 1:length(sample_indices)
%     idx = sample_indices(s);
% 
%     % Extract and ensure proper orientation
%     noisy_signal = squeeze(X_test(:,1,1,idx)) * noisy_std + noisy_mean;
%     clean_signal = squeeze(Y_test(:,1,1,idx)) * clean_std + clean_mean;
%     cnn_signal = squeeze(Y_pred(:,1,1,idx));
% 
%     % Convert to row vectors to match t
%     noisy_signal = noisy_signal(:)';
%     clean_signal = clean_signal(:)';
%     cnn_signal = cnn_signal(:)';
% 
%     % Plot original signals
%     subplot(length(sample_indices), 3, (s-1)*3 + 1);
%     plot(t, noisy_signal, 'r'); hold on;
%     plot(t, clean_signal, 'g');
%     title([sample_types{s} ': Original Signals']);
%     legend('Noisy', 'Clean');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     % Plot CNN denoised signal
%     subplot(length(sample_indices), 3, (s-1)*3 + 2);
%     plot(t, clean_signal, 'g'); hold on;
%     plot(t, cnn_signal, 'b', 'LineWidth', 1);
%     title([sample_types{s} ': CNN Denoising']);
%     legend('Clean', 'CNN');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     % Plot zoomed-in section for detail
%     subplot(length(sample_indices), 3, (s-1)*3 + 3);
%     % Define zoom ranges based on signal type
%     if contains(sample_types{s}, 'A')  % Type A - very sparse
%         zoom_range = round(0.15e-6 * fs):round(0.35e-6 * fs);
%     else  % Type B - sparse  
%         zoom_range = round(0.55e-6 * fs):round(0.75e-6 * fs);
%     end
% 
%     % Ensure zoom_range is within bounds
%     zoom_range = zoom_range(zoom_range >= 1 & zoom_range <= length(t));
% 
%     plot(t(zoom_range), clean_signal(zoom_range), 'g', 'LineWidth', 2); hold on;
%     plot(t(zoom_range), cnn_signal(zoom_range), 'b', 'LineWidth', 1);
%     title([sample_types{s} ': Zoomed Detail']);
%     legend('Clean', 'CNN');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% end
% %% 10. Calculate performance metrics for each signal type
% % Get test indices for each type
% type_a_test = test_indices(mod(test_indices-1, 2) == 0);
% type_b_test = test_indices(mod(test_indices-1, 2) == 1);
% 
% % Initialize arrays to store metrics
% num_types = 2;
% type_names = {'Type A', 'Type B'};
% type_test_indices = {type_a_test, type_b_test};
% 
% snr_before_types = zeros(num_types, 1);
% snr_improvements = zeros(num_types, 1);
% cc_values = zeros(num_types, 1);
% mse_values = zeros(num_types, 1);
% 
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
% 
%     if isempty(type_test)
%         fprintf('Warning: No test samples found for %s\n', type_names{type_idx});
%         continue;
%     end
% 
%     % Find indices in the test set
%     type_indices_in_test = find(ismember(test_indices, type_test));
% 
%     if isempty(type_indices_in_test)
%         continue;
%     end
% 
%     % Extract signals for this type
%     x_type = X_test(:,:,:,type_indices_in_test) * noisy_std + noisy_mean;
%     y_type = Y_test(:,:,:,type_indices_in_test) * clean_std + clean_mean;
%     y_pred_type = Y_pred(:,:,:,type_indices_in_test) * clean_std + clean_mean;
% 
%     % Calculate SNR before denoising
%     clean_power_type = mean(y_type(:).^2);
%     noise_power_type = mean((y_type(:) - x_type(:)).^2);
%     snr_before_types(type_idx) = 10*log10(clean_power_type/noise_power_type);
% 
%     % CNN metrics
%     residual_cnn_type = y_type(:) - y_pred_type(:);
%     mse_cnn_type = mean(residual_cnn_type.^2);
%     snr_cnn_type = 10*log10(clean_power_type/mse_cnn_type);
%     cc_cnn_type = corrcoef(y_type(:), y_pred_type(:));
%     cc_cnn_type = cc_cnn_type(1,2);
% 
%     % Store improvements and correlation coefficients
%     snr_improvements(type_idx) = snr_cnn_type - snr_before_types(type_idx);
%     cc_values(type_idx) = cc_cnn_type;
%     mse_values(type_idx) = mse_cnn_type;
% end
% 
% % Display comprehensive results by signal type
% fprintf('\n=== CNN Denoising Performance by Signal Type ===\n');
% fprintf('Signal Type | Initial SNR (dB) | Samples\n');
% fprintf('-------------------------------------------\n');
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
%     fprintf('%-11s | %8.2f        | %7d\n', type_names{type_idx}, ...
%         snr_before_types(type_idx), length(type_test));
% end
% 
% fprintf('\n=== SNR Improvement and Correlation by Signal Type ===\n');
% fprintf('Signal Type |               CNN               |\n');
% fprintf('Evaluation  |     SNR      |     CC    |    MSE    |\n');
% fprintf('----------------------------------------------------\n');
% for type_idx = 1:num_types
%     fprintf('%-11s | %8.2f dB  | %8.4f  | %8.5f  |\n', ...
%         type_names{type_idx}, snr_improvements(type_idx), cc_values(type_idx), mse_values(type_idx));
% end
% 
% % Calculate and display average improvements
% avg_snr_improvement_cnn = mean(snr_improvements);
% avg_cc_cnn_overall = mean(cc_values);
% 
% fprintf('\n=== Average Performance Across All Types ===\n');
% fprintf('Method |   Avg SNR    |   Avg CC\n');
% fprintf('-------------------------------\n');
% fprintf('CNN    | %8.2f dB  | %8.4f\n', avg_snr_improvement_cnn, avg_cc_cnn_overall);
% 
% %% 11. Save Results with Error Handling
% try
%     % First, verify that the network exists and is valid
%     if ~exist('net', 'var') || isempty(net)
%         error('Network variable is empty or not defined');
%     end
% 
%     % Save the comprehensive results file
%     save('cnn_AB_result.mat', 'net', 'Y_pred', ...
%          'mse_cnn', 'snr_before', 'snr_cnn', 'cc_cnn', ...
%          'snr_improvements', 'cc_values', 'mse_values', 'type_names', ...
%          'snr_before_types', 'avg_snr_improvement_cnn');
%     fprintf('Comprehensive results saved successfully to cnn_AB_result.mat\n');
% 
%     % Save just the network with verification
%     save('cnn_AB.mat', 'net');
% 
%     % Verify the saved file
%     fileInfo = dir('cnn_AB.mat');
%     if isempty(fileInfo) || fileInfo.bytes == 0
%         error('Failed to save network: File is empty');
%     else
%         fprintf('Network saved successfully to cnn_AB.mat (%d bytes)\n', fileInfo.bytes);
% 
%         % Double-check by trying to load it
%         testLoad = load('cnn_AB.mat');
%         if ~isfield(testLoad, 'net')
%             error('Saved file does not contain the network variable');
%         else
%             fprintf('Verified: Network was saved correctly and can be loaded\n');
%         end
%     end
% 
%     % Save detailed performance metrics
%     performance_summary = struct();
%     performance_summary.type_names = type_names;
%     performance_summary.snr_before = snr_before_types;
%     performance_summary.snr_improvements = snr_improvements;
%     performance_summary.correlation_coefficients = cc_values;
%     performance_summary.mse_values = mse_values;
%     performance_summary.avg_snr_improvement_cnn = avg_snr_improvement_cnn;
%     performance_summary.avg_cc_cnn = avg_cc_cnn_overall;
% 
%     save('performance_summary_AB.mat', 'performance_summary');
%     fprintf('Performance summary saved to performance_summary_AB.mat\n');
% 
% catch ME
%     fprintf('ERROR saving results: %s\n', ME.message);
%     % Try an alternative save location
%     try
%         alternativePath = fullfile(pwd, 'cnn_AB_backup.mat');
%         save(alternativePath, 'net');
%         fprintf('Network saved to alternative location: %s\n', alternativePath);
%     catch
%         fprintf('Failed to save to alternative location as well\n');
%     end
% end
% 
% %% 12. Final Summary
% fprintf('\n=== FINAL SUMMARY ===\n');
% fprintf('Dataset: %d samples with 2 signal types (A, B)\n', num_samples);
% fprintf('Signal length: %d samples (%.1f μs at %.0f MHz)\n', signal_length, t_total*1e6, fs/1e6);
% fprintf('Network: CNN only\n');
% fprintf('Training epochs: %d\n', options.MaxEpochs);
% fprintf('\nOverall Performance:\n');
% fprintf('• Average SNR improvement: %.2f dB\n', avg_snr_improvement_cnn);
% fprintf('• Average correlation: %.4f\n', avg_cc_cnn_overall);
% 
% fprintf('\nBest performing signal type:\n');
% [~, best_cnn_idx] = max(snr_improvements);
% fprintf('• CNN: %s (%.2f dB improvement)\n', type_names{best_cnn_idx}, snr_improvements(best_cnn_idx));
% 
% fprintf('\nFiles saved:\n');
% fprintf('• cnn_AB.mat (trained network)\n');
% fprintf('• cnn_AB_result.mat (complete results)\n');
% fprintf('• performance_summary_AB.mat (performance metrics)\n');
% fprintf('\n=== ANALYSIS COMPLETE ===\n');
%% --------------CNN_ONLY (TYPE A B C D : 10 18 20 25MM) 
% % 1. Enhanced Signal Simulation with Types A, B, C, D
% fs = 1000e6;                  % 1 GHz sampling
% t_total = 2e-6;               % 2 μs duration
% t = 0:1/fs:t_total;           % Time vector
% signal_length = length(t);
% num_samples = 2000;
% 
% clean_signals = zeros(num_samples, signal_length);
% noisy_signals = zeros(num_samples, signal_length);
% 
% for i = 1:num_samples
%     % Cycle through Type A, Type B, Type C, and Type D
%     signal_type = mod(i-1, 4) + 1;  % Types 1,2,3,4 correspond to A,B,C,D
% 
%     if signal_type == 1  % Type A: 10mm
%         clean_signal = zeros(size(t));
% 
%         % Very sparse PD events with random locations
%         num_events = 5 + randi(5);  % 3-7 events (very sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Clear, distinct bipolar spikes
%                 amplitude = 2.5 + rand() * 2;  % 2.5-4.5 amplitude
%                 polarity = (-1)^randi([0 1]);  % Random polarity
% 
%                 % Sharp bipolar pulse
%                 spike_width = 3 + randi(3);  % 3-6 samples wide
% 
%                 if start_idx + spike_width - 1 <= length(clean_signal)
%                     % Main spike
%                     clean_signal(start_idx) = polarity * amplitude;
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.8;
%                     end
% 
%                     % Decay tail
%                     for j = 2:spike_width-1
%                         if start_idx + j <= length(clean_signal)
%                             clean_signal(start_idx + j) = polarity * amplitude * 0.3 * exp(-(j-1));
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 2  % Type B: 18mm
%         clean_signal = zeros(size(t));
% 
%         % Sparse PD events with random locations
%         num_events = 55 + randi(10);  % 10-25 events (sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Mixed event types with random characteristics
%                 event_type = rand();
%                 amplitude = 2 + rand() * 3;  % 1.5-4.5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 if event_type < 0.7  % 70% - Sharp bipolar spikes
%                     spike_width = 2 + randi(4);
% 
%                     if start_idx + spike_width - 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         if start_idx + 1 <= length(clean_signal)
%                             clean_signal(start_idx + 1) = -polarity * amplitude * 0.7;
%                         end
% 
%                         % Add some oscillatory tail
%                         for j = 2:spike_width-1
%                             if start_idx + j <= length(clean_signal)
%                                 clean_signal(start_idx + j) = polarity * amplitude * 0.2 * sin(j);
%                             end
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 3  % Type C: 20mm
%         clean_signal = zeros(size(t));
% 
%         % Moderate to high frequency PD events with random locations
%         num_events = 120 + randi(30);  % 40-75 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 2 + rand() * 4;  % 1-5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.8  % 40% - Sharp spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 30% - Multi-frequency transients
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                   end
%                end
%             end
% 
%        else  % Type D: 25mm
%         clean_signal = zeros(size(t));
% 
%         % 25mm events with random locations throughout
%         num_events = 250 + randi(80);  % 100-180 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 3 + rand() * 4.2;  % 0.8-5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.6  % 30% - Quick spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 40% - Complex multi-component events
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                 end
%             end
%         end
%     end
% 
%     % Keep signals in a reasonable range but preserve relative amplitudes
%     max_amplitude = max(abs(clean_signal));
%     if max_amplitude > 0
%         if max_amplitude > 5
%             clean_signal = clean_signal * (5 / max_amplitude);
%         end
%     end
% 
%     % Normalize clean signal
%     clean_signal = clean_signal / max(abs(clean_signal) + eps);  % Avoid division by zero
% 
%     % Add Noise
%     desired_snr = -10; % adjust SNR
%     noisy_signal = awgn(clean_signal, desired_snr, 'measured');
% 
%     % Save memory by using single precision
%     clean_signal = single(clean_signal);
%     noisy_signal = single(noisy_signal);
% 
%     clean_signals(i,:) = clean_signal;
%     noisy_signals(i,:) = noisy_signal;
% end
% 
% % Plot Type A, B, C, D samples
% figure;
% subplot(4,2,1);
% plot(t, clean_signals(1,:));
% title('Type A: 10mm (Clean)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,2);
% plot(t, noisy_signals(1,:));
% title('Type A: 10mm (Noisy)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,3);
% plot(t, clean_signals(2,:));
% title('Type B: 18mm(Clean)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,4);
% plot(t, noisy_signals(2,:));
% title('Type B: 18mm(Noisy)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,5);
% plot(t, clean_signals(3,:));
% title('Type C: 20mm(Clean)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,6);
% plot(t, noisy_signals(3,:));
% title('Type C: 20mm(Noisy)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,7);
% plot(t, clean_signals(4,:));
% title('Type D: 25mm (Clean)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(4,2,8);
% plot(t, noisy_signals(4,:));
% title('Type D: 25mm (Noisy)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% %% 2. Improved Data Preparation
% % Don't normalize to range, use standardization instead
% clean_mean = mean(clean_signals(:));
% clean_std = std(clean_signals(:));
% noisy_mean = mean(noisy_signals(:));
% noisy_std = std(noisy_signals(:));
% 
% clean_signals_norm = (clean_signals - clean_mean) / clean_std;
% noisy_signals_norm = (noisy_signals - noisy_mean) / noisy_std;
% 
% % Fixed data split with stratification (equal Type A/B/C/D in each)
% num_type_a = sum(1:4:num_samples <= num_samples);
% num_type_b = sum(2:4:num_samples <= num_samples);
% num_type_c = sum(3:4:num_samples <= num_samples);
% num_type_d = sum(4:4:num_samples <= num_samples);
% 
% % Type indices
% type_a_indices = 1:4:num_samples;
% type_b_indices = 2:4:num_samples;
% type_c_indices = 3:4:num_samples;
% type_d_indices = 4:4:num_samples;
% 
% type_a_train_size = floor(0.7 * num_type_a);
% type_a_val_size = floor(0.15 * num_type_a);
% type_b_train_size = floor(0.7 * num_type_b);
% type_b_val_size = floor(0.15 * num_type_b);
% type_c_train_size = floor(0.7 * num_type_c);
% type_c_val_size = floor(0.15 * num_type_c);
% type_d_train_size = floor(0.7 * num_type_d);
% type_d_val_size = floor(0.15 * num_type_d);
% 
% % Random permutation within types
% type_a_perm = type_a_indices(randperm(length(type_a_indices)));
% type_b_perm = type_b_indices(randperm(length(type_b_indices)));
% type_c_perm = type_c_indices(randperm(length(type_c_indices)));
% type_d_perm = type_d_indices(randperm(length(type_d_indices)));
% 
% % Split
% train_indices = [type_a_perm(1:type_a_train_size), type_b_perm(1:type_b_train_size), ...
%                  type_c_perm(1:type_c_train_size), type_d_perm(1:type_d_train_size)];
% val_indices = [type_a_perm(type_a_train_size+1:type_a_train_size+type_a_val_size), ...
%                type_b_perm(type_b_train_size+1:type_b_train_size+type_b_val_size), ...
%                type_c_perm(type_c_train_size+1:type_c_train_size+type_c_val_size), ...
%                type_d_perm(type_d_train_size+1:type_d_train_size+type_d_val_size)];
% test_indices = [type_a_perm(type_a_train_size+type_a_val_size+1:end), ...
%                 type_b_perm(type_b_train_size+type_b_val_size+1:end), ...
%                 type_c_perm(type_c_train_size+type_c_val_size+1:end), ...
%                 type_d_perm(type_d_train_size+type_d_val_size+1:end)];
% 
% % Reshape for network
% X_train = reshape(noisy_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% Y_train = reshape(clean_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% X_val = reshape(noisy_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% Y_val = reshape(clean_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% X_test = reshape(noisy_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% Y_test = reshape(clean_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% 
% %% 3. CNN Architecture
% layers = [
%     imageInputLayer([signal_length,1,1], 'Name', 'input')
% 
%     % Enhanced encoder with residual-like connections
%     convolution2dLayer([7,1], 64, 'Padding','same', 'Name','conv1')
%     batchNormalizationLayer('Name','bn1')
%     leakyReluLayer(0.1, 'Name','lrelu1')
% 
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','conv2')
%     batchNormalizationLayer('Name','bn2')
%     leakyReluLayer(0.1, 'Name','lrelu2')
% 
%     % Multi-scale feature extraction
%     convolution2dLayer([3,1], 128, 'Padding','same', 'Name','conv3a')
%     batchNormalizationLayer('Name','bn3a')
%     leakyReluLayer(0.1, 'Name','lrelu3a')
% 
%     % Dilated convolutions for larger receptive field
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',2, 'Name','dilated1')
%     batchNormalizationLayer('Name','bn_dilated1')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated1')
%     dropoutLayer(0.1, 'Name','dropout1')
% 
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',4, 'Name','dilated2')
%     batchNormalizationLayer('Name','bn_dilated2')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated2')
%     dropoutLayer(0.1, 'Name','dropout2')
% 
%     % Decoder path with improved architecture
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','decode1')
%     batchNormalizationLayer('Name','bn_decode1')
%     leakyReluLayer(0.1, 'Name','lrelu_decode1')
% 
%     convolution2dLayer([3,1], 32, 'Padding','same', 'Name','decode2')
%     batchNormalizationLayer('Name','bn_decode2')
%     leakyReluLayer(0.1, 'Name','lrelu_decode2')
% 
%     % Final output
%     convolution2dLayer([3,1], 1, 'Padding','same', 'Name','output')
%     regressionLayer()
% ];
% 
% %% 4. Training Configuration 
% options = trainingOptions('adam', ...
%     'MaxEpochs', 200, ...                   % Significantly increased
%     'MiniBatchSize', 24, ...                
%     'InitialLearnRate', 2e-4, ...           % Slightly lower initial rate
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.75, ...        % More gradual decay
%     'LearnRateDropPeriod', 50, ...          % Less frequent drops
%     'L2Regularization', 1.5e-4, ...         
%     'GradientThreshold', 0.8, ...           
%     'ValidationData', {X_val, Y_val}, ...   
%     'ValidationFrequency', 120, ...         % Check validation less frequently
%     'ValidationPatience', 60, ...           % Much more patient
%     'Shuffle', 'every-epoch', ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');
% 
% %% 5. Train Network 
% net = trainNetwork(X_train, Y_train, layers, options);
% 
% Y_pred_raw = predict(net, X_test);
% 
% % Convert back to original scale
% Y_pred = Y_pred_raw * clean_std + clean_mean;
% Y_test_orig = Y_test * clean_std + clean_mean; 
% 
% %% 6. Evaluation 
% X_test_orig = X_test * noisy_std + noisy_mean;
% 
% clean_power = mean(Y_test_orig(:).^2);
% noise_power = mean((Y_test_orig(:)-X_test_orig(:)).^2);
% snr_before = 10*log10(clean_power/noise_power);
% 
% % CNN output metrics
% residual_cnn = Y_test_orig(:) - Y_pred(:);
% mse_cnn = mean(residual_cnn.^2);
% snr_cnn = 10*log10(clean_power/mean(residual_cnn.^2));
% cc_cnn = corrcoef(Y_test_orig(:), Y_pred(:));
% cc_cnn = cc_cnn(1,2);
% 
% % Display comprehensive results
% fprintf('=== CNN Denoising Performance ===\n');
% fprintf('                |  Before  |   CNN   \n');
% fprintf('------------------------------------\n');
% fprintf('MSE             |    -     | %.4f  \n', mse_cnn);
% fprintf('SNR (dB)        | %6.2f   | %6.2f  \n',snr_before, snr_cnn);
% fprintf('CC              |    -     | %.4f  \n', cc_cnn);
% fprintf('Improvement (dB)| -        | %6.2f  \n', snr_cnn - snr_before);
% 
% %% 7. Evaluate on 10 Random Test Samples
% num_eval_samples = 10;
% random_indices = randperm(length(test_indices), num_eval_samples);
% avg_snr_before = 0;
% avg_mse_cnn = 0;
% avg_snr_cnn = 0;
% avg_cc_cnn = 0;
% 
% figure;
% for idx = 1:num_eval_samples
%     test_idx = random_indices(idx);
% 
%     x = X_test(:,1,1,test_idx);
%     y_true = Y_test(:,1,1,test_idx);
%     y_cnn = Y_pred(:,1,1,test_idx);
% 
%     % Metrics
%     mse_cnn_i = mean((y_true - y_cnn).^2);
%     snr_cnn_i = 10*log10(mean(y_true.^2) / mse_cnn_i);
%     cc_cnn_i = corrcoef(y_true, y_cnn); cc_cnn_i = cc_cnn_i(1,2);
% 
%     % Accumulate for average
%     avg_snr_before = avg_snr_before + snr_before;
%     avg_mse_cnn = avg_mse_cnn + mse_cnn_i;
%     avg_snr_cnn = avg_snr_cnn + snr_cnn_i;
%     avg_cc_cnn = avg_cc_cnn + cc_cnn_i;
% 
%     % Plot
%     subplot(num_eval_samples, 3, (idx-1)*3 + 1);
%     plot(t, x); title(sprintf('Noisy #%d', idx)); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 3, (idx-1)*3 + 2);
%     plot(t, y_true); title('Clean'); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 3, (idx-1)*3 + 3);
%     plot(t, y_cnn); title('CNN'); ylabel('Amplitude');
% end
% 
% % Take averages
% avg_snr_before = avg_snr_before / num_eval_samples;
% avg_mse_cnn = avg_mse_cnn / num_eval_samples;
% avg_snr_cnn = avg_snr_cnn / num_eval_samples;
% avg_cc_cnn = avg_cc_cnn / num_eval_samples;
% 
% % Display results
% fprintf('\n=== 10-Sample Random Evaluation ===\n');
% fprintf('MSE        | CNN: %.4f\n', avg_mse_cnn);
% fprintf('SNR (dB)   | CNN: %.2f\n', avg_snr_cnn);
% fprintf('Corr Coef  | CNN: %.4f\n', avg_cc_cnn);
% 
% % Compare with overall results
% fprintf('\n=== Comparison with Overall Results ===\n');
% fprintf('                     | Overall Eval | 10-Sample Avg\n');
% fprintf('-----------------------------------------------------\n');
% fprintf('SNR Before (dB)      | %6.2f       | %6.2f\n', snr_before, avg_snr_before);
% fprintf('SNR CNN (dB)         | %6.2f       | %6.2f\n', snr_cnn, avg_snr_cnn);
% fprintf('CC CNN               | %6.4f       | %6.4f\n', cc_cnn, avg_cc_cnn);
% 
% %% 8. Visual comparison of denoised signals for Type A, B, C, and D
% % Find representative samples of each type from the test set
% type_a_indices = find(mod(test_indices-1, 4) == 0);  % Type A (signal_type = 1)
% type_b_indices = find(mod(test_indices-1, 4) == 1);  % Type B (signal_type = 2)
% type_c_indices = find(mod(test_indices-1, 4) == 2);  % Type C (signal_type = 3)
% type_d_indices = find(mod(test_indices-1, 4) == 3);  % Type D (signal_type = 4)
% 
% % Select the first one of each type (if available)
% sample_indices = [];
% sample_types = {};
% 
% if ~isempty(type_a_indices)
%     sample_indices(end+1) = type_a_indices(1);
%     sample_types{end+1} = 'Type A';
% end
% if ~isempty(type_b_indices)
%     sample_indices(end+1) = type_b_indices(1);
%     sample_types{end+1} = 'Type B';
% end
% if ~isempty(type_c_indices)
%     sample_indices(end+1) = type_c_indices(1);
%     sample_types{end+1} = 'Type C';
% end
% if ~isempty(type_d_indices)
%     sample_indices(end+1) = type_d_indices(1);
%     sample_types{end+1} = 'Type D';
% end
% 
% % Create the comparison figure
% figure('Position', [100, 100, 1200, 1000]);
% for s = 1:length(sample_indices)
%     idx = sample_indices(s);
%     noisy_signal = squeeze(X_test(:,1,1,idx)) * noisy_std + noisy_mean;
%     clean_signal = squeeze(Y_test(:,1,1,idx)) * clean_std + clean_mean;
%     cnn_signal = squeeze(Y_pred(:,1,1,idx)) * clean_std + clean_mean;
% 
%     % Plot original signals
%     subplot(length(sample_indices), 3, (s-1)*3 + 1);
%     plot(t, noisy_signal, 'r'); hold on;
%     plot(t, clean_signal, 'g');
%     title([sample_types{s} ': Original Signals']);
%     legend('Noisy', 'Clean');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     % Plot CNN denoised signal
%     subplot(length(sample_indices), 3, (s-1)*3 + 2);
%     plot(t, clean_signal, 'g'); hold on;
%     plot(t, cnn_signal, 'b', 'LineWidth', 1);
%     title([sample_types{s} ': CNN Denoising']);
%     legend('Clean', 'CNN');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     % Plot zoomed-in section for detail
%     subplot(length(sample_indices), 3, (s-1)*3 + 3);
%     % Define zoom ranges based on signal type
%     if contains(sample_types{s}, 'A')  % Type A - very sparse
%         zoom_range = round(0.15e-6 * fs):round(0.35e-6 * fs);
%     elseif contains(sample_types{s}, 'B')  % Type B - sparse  
%         zoom_range = round(0.55e-6 * fs):round(0.75e-6 * fs);
%     elseif contains(sample_types{s}, 'C')  % Type C - moderate-high
%         zoom_range = round(0.4e-6 * fs):round(0.6e-6 * fs);
%     else  % Type D - very high frequency
%         zoom_range = round(0.3e-6 * fs):round(0.5e-6 * fs);
%     end
% 
%     plot(t(zoom_range), clean_signal(zoom_range), 'g', 'LineWidth', 2); hold on;
%     plot(t(zoom_range), cnn_signal(zoom_range), 'b', 'LineWidth', 1);
%     title([sample_types{s} ': Zoomed Detail']);
%     legend('Clean', 'CNN');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% end
% 
% %% 9. Calculate performance metrics for each signal type
% % Get test indices for each type
% type_a_test = test_indices(mod(test_indices-1, 4) == 0);
% type_b_test = test_indices(mod(test_indices-1, 4) == 1);
% type_c_test = test_indices(mod(test_indices-1, 4) == 2);
% type_d_test = test_indices(mod(test_indices-1, 4) == 3);
% 
% % Initialize arrays to store metrics
% num_types = 4;
% type_names = {'Type A', 'Type B', 'Type C', 'Type D'};
% type_test_indices = {type_a_test, type_b_test, type_c_test, type_d_test};
% 
% snr_before_types = zeros(num_types, 1);
% snr_improvements = zeros(num_types, 1);
% cc_values = zeros(num_types, 1);
% mse_values = zeros(num_types, 1);
% 
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
% 
%     if isempty(type_test)
%         fprintf('Warning: No test samples found for %s\n', type_names{type_idx});
%         continue;
%     end
% 
%     % Find indices in the test set
%     type_indices_in_test = find(ismember(test_indices, type_test));
% 
%     if isempty(type_indices_in_test)
%         continue;
%     end
% 
%     % Extract signals for this type
%     x_type = X_test(:,:,:,type_indices_in_test) * noisy_std + noisy_mean;
%     y_type = Y_test(:,:,:,type_indices_in_test) * clean_std + clean_mean;
%     y_pred_type = Y_pred(:,:,:,type_indices_in_test) * clean_std + clean_mean;
% 
%     % Calculate SNR before denoising
%     clean_power_type = mean(y_type(:).^2);
%     noise_power_type = mean((y_type(:) - x_type(:)).^2);
%     snr_before_types(type_idx) = 10*log10(clean_power_type/noise_power_type);
% 
%     % CNN metrics
%     residual_cnn_type = y_type(:) - y_pred_type(:);
%     mse_cnn_type = mean(residual_cnn_type.^2);
%     snr_cnn_type = 10*log10(clean_power_type/mse_cnn_type);
%     cc_cnn_type = corrcoef(y_type(:), y_pred_type(:));
%     cc_cnn_type = cc_cnn_type(1,2);
% 
%     % Store improvements and correlation coefficients
%     snr_improvements(type_idx) = snr_cnn_type - snr_before_types(type_idx);
%     cc_values(type_idx) = cc_cnn_type;
%     mse_values(type_idx) = mse_cnn_type;
% end
% 
% % Display comprehensive results by signal type
% fprintf('\n=== CNN Denoising Performance by Signal Type ===\n');
% fprintf('Signal Type | Initial SNR (dB) | Samples\n');
% fprintf('-------------------------------------------\n');
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
%     fprintf('%-11s | %8.2f        | %7d\n', type_names{type_idx}, ...
%         snr_before_types(type_idx), length(type_test));
% end
% 
% fprintf('\n=== SNR Improvement and Correlation by Signal Type ===\n');
% fprintf('Signal Type |               CNN               |\n');
% fprintf('Evaluation  |     SNR      |     CC    |    MSE    |\n');
% fprintf('----------------------------------------------------\n');
% for type_idx = 1:num_types
%     fprintf('%-11s | %8.2f dB  | %8.4f  | %8.5f  |\n', ...
%         type_names{type_idx}, snr_improvements(type_idx), cc_values(type_idx), mse_values(type_idx));
% end
% 
% % Calculate and display average improvements
% avg_snr_improvement_cnn = mean(snr_improvements);
% avg_cc_cnn_overall = mean(cc_values);
% 
% fprintf('\n=== Average Performance Across All Types ===\n');
% fprintf('Method |   Avg SNR    |   Avg CC\n');
% fprintf('-------------------------------\n');
% fprintf('CNN    | %8.2f dB  | %8.4f\n', avg_snr_improvement_cnn, avg_cc_cnn_overall);
% 
% %% 10. Save Results with Error Handling
% try
%     % First, verify that the network exists and is valid
%     if ~exist('net', 'var') || isempty(net)
%         error('Network variable is empty or not defined');
%     end
% 
%     % Save the comprehensive results file
%     save('cnn_ABCD_result.mat', 'net', 'Y_pred', ...
%          'mse_cnn', 'snr_before', 'snr_cnn', 'cc_cnn', ...
%          'snr_improvements', 'cc_values', 'mse_values', 'type_names', ...
%          'snr_before_types', 'avg_snr_improvement_cnn');
%     fprintf('Comprehensive results saved successfully to cnn_ABCD_result.mat\n');
% 
%     % Save just the network with verification
%     save('cnn_ABCD.mat', 'net');
% 
%     % Verify the saved file
%     fileInfo = dir('cnn_ABCD.mat');
%     if isempty(fileInfo) || fileInfo.bytes == 0
%         error('Failed to save network: File is empty');
%     else
%         fprintf('Network saved successfully to cnn_ABCD.mat (%d bytes)\n', fileInfo.bytes);
% 
%         % Double-check by trying to load it
%         testLoad = load('cnn_ABCD.mat');
%         if ~isfield(testLoad, 'net')
%             error('Saved file does not contain the network variable');
%         else
%             fprintf('Verified: Network was saved correctly and can be loaded\n');
%         end
%     end
% 
%     % Save detailed performance metrics
%     performance_summary = struct();
%     performance_summary.type_names = type_names;
%     performance_summary.snr_before = snr_before_types;
%     performance_summary.snr_improvements = snr_improvements;
%     performance_summary.correlation_coefficients = cc_values;
%     performance_summary.mse_values = mse_values;
%     performance_summary.avg_snr_improvement_cnn = avg_snr_improvement_cnn;
%     performance_summary.avg_cc_cnn = avg_cc_cnn_overall;
% 
%     save('performance_summary_ABCD.mat', 'performance_summary');
%     fprintf('Performance summary saved to performance_summary_ABCD.mat\n');
% 
% catch ME
%     fprintf('ERROR saving results: %s\n', ME.message);
%     % Try an alternative save location
%     try
%         alternativePath = fullfile(pwd, 'cnn_ABCD_backup.mat');
%         save(alternativePath, 'net');
%         fprintf('Network saved to alternative location: %s\n', alternativePath);
%     catch
%         fprintf('Failed to save to alternative location as well\n');
%     end
% end
% 
% %% 11. Final Summary
% fprintf('\n=== FINAL SUMMARY ===\n');
% fprintf('Dataset: %d samples with 4 signal types (A, B, C, D)\n', num_samples);
% fprintf('Signal length: %d samples (%.1f μs at %.0f MHz)\n', signal_length, t_total*1e6, fs/1e6);
% fprintf('Network: CNN only\n');
% fprintf('Training epochs: %d\n', options.MaxEpochs);
% fprintf('\nOverall Performance:\n');
% fprintf('• Average SNR improvement: %.2f dB\n', avg_snr_improvement_cnn);
% fprintf('• Average correlation: %.4f\n', avg_cc_cnn_overall);
% 
% fprintf('\nBest performing signal type:\n');
% [~, best_cnn_idx] = max(snr_improvements);
% fprintf('• CNN: %s (%.2f dB improvement)\n', type_names{best_cnn_idx}, snr_improvements(best_cnn_idx));
% 
% fprintf('\nFiles saved:\n');
% fprintf('• cnn_ABCD.mat (trained network)\n');
% fprintf('• cnn_ABCD_result.mat (complete results)\n');
% fprintf('• performance_summary_ABCD.mat (performance metrics)\n');
% fprintf('\n=== ANALYSIS COMPLETE ===\n');

%% ------------ONLY CNN (ABCDEF)---------
% % Enhanced Signal Simulation with Types A, B, C, D, E, F (CNN Only)
% % Type A: Sparse PD pulses (4 pulses with gaps)
% % Type B: Spike-dense signal (realistic sharp pulses with gaps)
% % Type C: 10mm (very sparse PD events)
% % Type D: 18mm (sparse PD events)
% % Type E: 20mm (moderate-high frequency PD events)
% % Type F: 25mm (high frequency, complex PD events)
% 
% fs = 1000e6;                  % 1 GHz sampling
% t_total = 2e-6;               % 2 μs duration
% t = 0:1/fs:t_total;           % Time vector
% signal_length = length(t);
% num_samples = 3000;           % Total samples for 6 types
% 
% clean_signals = zeros(num_samples, signal_length);
% noisy_signals = zeros(num_samples, signal_length);
% 
% for i = 1:num_samples
%     % Cycle through Type A, Type B, Type C, Type D, Type E, Type F
%     signal_type = mod(i-1, 6) + 1;  % Types 1,2,3,4,5,6 correspond to A,B,C,D,E,F
% 
%     if signal_type == 1  % Type A: Sparse PD pulses (4 pulses with gaps)
%         clean_signal = zeros(size(t));
%         start_times = [0.2e-6, 0.6e-6, 1.2e-6, 1.6e-6]; % Clear time gaps
% 
%         for k = 1:length(start_times)
%             A = 10 + rand()*10;
%             fc = 25e6 + rand()*10e6; % Higher freq helps sharpen
%             tau = 0.01e-6 + rand()*0.03e-6; % Very short pulse
%             pulse_t = t - start_times(k);
%             pulse_t = pulse_t(pulse_t >= 0);
% 
%             % Generate a short pulse with fewer points
%             pulse_duration = 0.05e-6; % 50 ns duration
%             pulse_t = pulse_t(pulse_t <= pulse_duration);
%             pulse = A * exp(-pulse_t/tau) .* sin(2*pi*fc*pulse_t);
% 
%             % Insert pulse at the correct position
%             start_idx = find(t >= start_times(k), 1);
%             pulse_len = length(pulse);
%             if start_idx + pulse_len - 1 <= length(clean_signal)
%                 clean_signal(start_idx:start_idx + pulse_len - 1) = pulse;
%             end
%         end
% 
%     elseif signal_type == 2  % Type B: Spike-dense signal (realistic sharp pulses with gaps)
%         clean_signal = zeros(size(t));
%         num_spikes = 20 + randi(10); % Fewer, more realistic pulses
%         spike_len = 2; % Length of each biphasic spike
% 
%         for s = 1:num_spikes
%             start_idx = randi([1, signal_length - spike_len]);
%             amp = 0.5 + 0.5*rand(); % Amplitude
%             direction = (-1)^randi([0 1]); % Flip polarity randomly
% 
%             % Biphasic spike: [positive, negative] or [negative, positive]
%             clean_signal(start_idx) = direction * amp;
%             clean_signal(start_idx + 1) = -direction * amp;
%         end
% 
%     elseif signal_type == 3  % Type C: 10mm
%         clean_signal = zeros(size(t));
% 
%         % Very sparse PD events with random locations
%         num_events = 5 + randi(5);  % 5-10 events (very sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Clear, distinct bipolar spikes
%                 amplitude = 2.5 + rand() * 2;  % 2.5-4.5 amplitude
%                 polarity = (-1)^randi([0 1]);  % Random polarity
% 
%                 % Sharp bipolar pulse
%                 spike_width = 3 + randi(3);  % 3-6 samples wide
% 
%                 if start_idx + spike_width - 1 <= length(clean_signal)
%                     % Main spike
%                     clean_signal(start_idx) = polarity * amplitude;
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.8;
%                     end
% 
%                     % Decay tail
%                     for j = 2:spike_width-1
%                         if start_idx + j <= length(clean_signal)
%                             clean_signal(start_idx + j) = polarity * amplitude * 0.3 * exp(-(j-1));
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 4  % Type D: 18mm
%         clean_signal = zeros(size(t));
% 
%         % Sparse PD events with random locations
%         num_events = 55 + randi(10);  % 55-65 events (sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Mixed event types with random characteristics
%                 event_type = rand();
%                 amplitude = 2 + rand() * 3;  % 2-5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 if event_type < 0.7  % 70% - Sharp bipolar spikes
%                     spike_width = 2 + randi(4);
% 
%                     if start_idx + spike_width - 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         if start_idx + 1 <= length(clean_signal)
%                             clean_signal(start_idx + 1) = -polarity * amplitude * 0.7;
%                         end
% 
%                         % Add some oscillatory tail
%                         for j = 2:spike_width-1
%                             if start_idx + j <= length(clean_signal)
%                                 clean_signal(start_idx + j) = polarity * amplitude * 0.2 * sin(j);
%                             end
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 5  % Type E: 20mm
%         clean_signal = zeros(size(t));
% 
%         % Moderate to high frequency PD events with random locations
%         num_events = 120 + randi(30);  % 120-150 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 2 + rand() * 4;  % 2-6 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.6  % 60% - Sharp spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 40% - Multi-frequency transients
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                 end
%             end
%         end
% 
%     else  % Type F: 25mm
%         clean_signal = zeros(size(t));
% 
%         % 25mm events with random locations throughout
%         num_events = 250 + randi(80);  % 250-330 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 3 + rand() * 4.2;  % 3-7.2 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.4  % 40% - Quick spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
%                     end
% 
%                 else  % 60% - Complex multi-component events
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                 end
%             end
%         end
%     end
% 
%     % Keep signals in a reasonable range but preserve relative amplitudes
%     max_amplitude = max(abs(clean_signal));
%     if max_amplitude > 0
%         if max_amplitude > 5
%             clean_signal = clean_signal * (5 / max_amplitude);
%         end
%     end
% 
%     % Normalize clean signal
%     clean_signal = clean_signal / max(abs(clean_signal) + eps);  % Avoid division by zero
% 
%     % Noise: White + powerline + narrowband + impulse (with reduced levels)
%     white_noise = 0.08*randn(size(clean_signal));
%     powerline_noise = 0.025*sin(2*pi*50e6*t) + 0.015*sin(2*pi*150e6*t);
%     narrowband = 0.03*sin(2*pi*80e6*t + rand()*2*pi);
%     impulse_noise = zeros(size(clean_signal));
%     spike_pos = randperm(length(clean_signal), 15);
%     impulse_noise(spike_pos) = 0.4*(0.2 + 0.8*rand(1,15));
% 
%     noise = white_noise + powerline_noise + narrowband + impulse_noise;
% 
%     % Adjust SNR (Improved control)
%     current_snr = 10*log10(var(clean_signal) / (var(noise) + eps));
%     desired_snr = -10 + rand()*8;  % 
%     noise = noise * 10^((current_snr-desired_snr)/20);
% 
%     noisy_signal = clean_signal + noise;
% 
%     clean_signals(i,:) = clean_signal;
%     noisy_signals(i,:) = noisy_signal;
% end
% 
% % Plot Type A-F samples
% figure('Position', [100, 100, 1600, 1200]);
% type_names = {'Type A: Sparse PD pulses', 'Type B: Spike-dense signal', ...
%               'Type C: 10mm', 'Type D: 18mm', 'Type E: 20mm', 'Type F: 25mm'};
% 
% for type_idx = 1:6
%     subplot(6,2,(type_idx-1)*2+1);
%     plot(t, clean_signals(type_idx,:));
%     title([type_names{type_idx} ' (Clean)']);
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% 
%     subplot(6,2,(type_idx-1)*2+2);
%     plot(t, noisy_signals(type_idx,:));
%     title([type_names{type_idx} ' (Noisy)']);
%     xlabel('Time (s)');
%     ylabel('Amplitude');
% end
% 
% %% 2. Improved Data Preparation
% % Don't normalize to range, use standardization instead
% clean_mean = mean(clean_signals(:));
% clean_std = std(clean_signals(:));
% noisy_mean = mean(noisy_signals(:));
% noisy_std = std(noisy_signals(:));
% 
% clean_signals_norm = (clean_signals - clean_mean) / clean_std;
% noisy_signals_norm = (noisy_signals - noisy_mean) / noisy_std;
% 
% % Fixed data split with stratification (equal Type A/B/C/D/E/F in each)
% num_type_a = sum(1:6:num_samples <= num_samples);
% num_type_b = sum(2:6:num_samples <= num_samples);
% num_type_c = sum(3:6:num_samples <= num_samples);
% num_type_d = sum(4:6:num_samples <= num_samples);
% num_type_e = sum(5:6:num_samples <= num_samples);
% num_type_f = sum(6:6:num_samples <= num_samples);
% 
% % Type indices
% type_a_indices = 1:6:num_samples;
% type_b_indices = 2:6:num_samples;
% type_c_indices = 3:6:num_samples;
% type_d_indices = 4:6:num_samples;
% type_e_indices = 5:6:num_samples;
% type_f_indices = 6:6:num_samples;
% 
% type_a_train_size = floor(0.7 * num_type_a);
% type_a_val_size = floor(0.15 * num_type_a);
% type_b_train_size = floor(0.7 * num_type_b);
% type_b_val_size = floor(0.15 * num_type_b);
% type_c_train_size = floor(0.7 * num_type_c);
% type_c_val_size = floor(0.15 * num_type_c);
% type_d_train_size = floor(0.7 * num_type_d);
% type_d_val_size = floor(0.15 * num_type_d);
% type_e_train_size = floor(0.7 * num_type_e);
% type_e_val_size = floor(0.15 * num_type_e);
% type_f_train_size = floor(0.7 * num_type_f);
% type_f_val_size = floor(0.15 * num_type_f);
% 
% % Random permutation within types
% type_a_perm = type_a_indices(randperm(length(type_a_indices)));
% type_b_perm = type_b_indices(randperm(length(type_b_indices)));
% type_c_perm = type_c_indices(randperm(length(type_c_indices)));
% type_d_perm = type_d_indices(randperm(length(type_d_indices)));
% type_e_perm = type_e_indices(randperm(length(type_e_indices)));
% type_f_perm = type_f_indices(randperm(length(type_f_indices)));
% 
% % Split
% train_indices = [type_a_perm(1:type_a_train_size), type_b_perm(1:type_b_train_size), ...
%                  type_c_perm(1:type_c_train_size), type_d_perm(1:type_d_train_size), ...
%                  type_e_perm(1:type_e_train_size), type_f_perm(1:type_f_train_size)];
% val_indices = [type_a_perm(type_a_train_size+1:type_a_train_size+type_a_val_size), ...
%                type_b_perm(type_b_train_size+1:type_b_train_size+type_b_val_size), ...
%                type_c_perm(type_c_train_size+1:type_c_train_size+type_c_val_size), ...
%                type_d_perm(type_d_train_size+1:type_d_train_size+type_d_val_size), ...
%                type_e_perm(type_e_train_size+1:type_e_train_size+type_e_val_size), ...
%                type_f_perm(type_f_train_size+1:type_f_train_size+type_f_val_size)];
% test_indices = [type_a_perm(type_a_train_size+type_a_val_size+1:end), ...
%                 type_b_perm(type_b_train_size+type_b_val_size+1:end), ...
%                 type_c_perm(type_c_train_size+type_c_val_size+1:end), ...
%                 type_d_perm(type_d_train_size+type_d_val_size+1:end), ...
%                 type_e_perm(type_e_train_size+type_e_val_size+1:end), ...
%                 type_f_perm(type_f_train_size+type_f_val_size+1:end)];
% 
% % Reshape for network
% X_train = reshape(noisy_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% Y_train = reshape(clean_signals_norm(train_indices,:)', [signal_length,1,1,length(train_indices)]);
% X_val = reshape(noisy_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% Y_val = reshape(clean_signals_norm(val_indices,:)', [signal_length,1,1,length(val_indices)]);
% X_test = reshape(noisy_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% Y_test = reshape(clean_signals_norm(test_indices,:)', [signal_length,1,1,length(test_indices)]);
% 
% %% 3. CNN Architecture 
% layers = [
%     imageInputLayer([signal_length,1,1], 'Name', 'input')
% 
%     % Enhanced encoder with residual-like connections
%     convolution2dLayer([7,1], 64, 'Padding','same', 'Name','conv1')
%     batchNormalizationLayer('Name','bn1')
%     leakyReluLayer(0.1, 'Name','lrelu1')
% 
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','conv2')
%     batchNormalizationLayer('Name','bn2')
%     leakyReluLayer(0.1, 'Name','lrelu2')
% 
%     % Multi-scale feature extraction
%     convolution2dLayer([3,1], 128, 'Padding','same', 'Name','conv3a')
%     batchNormalizationLayer('Name','bn3a')
%     leakyReluLayer(0.1, 'Name','lrelu3a')
% 
%     % Dilated convolutions for larger receptive field
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',2, 'Name','dilated1')
%     batchNormalizationLayer('Name','bn_dilated1')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated1')
%     dropoutLayer(0.1, 'Name','dropout1')
% 
%     convolution2dLayer([3,1], 128, 'Padding','same','DilationFactor',4, 'Name','dilated2')
%     batchNormalizationLayer('Name','bn_dilated2')
%     leakyReluLayer(0.1, 'Name','lrelu_dilated2')
%     dropoutLayer(0.1, 'Name','dropout2')
% 
%     % Decoder path with improved architecture
%     convolution2dLayer([5,1], 64, 'Padding','same', 'Name','decode1')
%     batchNormalizationLayer('Name','bn_decode1')
%     leakyReluLayer(0.1, 'Name','lrelu_decode1')
% 
%     convolution2dLayer([3,1], 32, 'Padding','same', 'Name','decode2')
%     batchNormalizationLayer('Name','bn_decode2')
%     leakyReluLayer(0.1, 'Name','lrelu_decode2')
% 
%     % Final output
%     convolution2dLayer([3,1], 1, 'Padding','same', 'Name','output')
%     regressionLayer()
% ];
% 
% %% 4. Training Configuration 
% options = trainingOptions('adam', ...
%     'MaxEpochs', 300, ...                   % Significantly increased
%     'MiniBatchSize', 24, ...                
%     'InitialLearnRate', 2e-4, ...           % Slightly lower initial rate
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.75, ...        % More gradual decay
%     'LearnRateDropPeriod', 50, ...          % Less frequent drops
%     'L2Regularization', 1.5e-4, ...         
%     'GradientThreshold', 0.8, ...           
%     'ValidationData', {X_val, Y_val}, ...   
%     'ValidationFrequency', 150, ...         % Check validation less frequently
%     'ValidationPatience', 80, ...           % Much more patient
%     'Shuffle', 'every-epoch', ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');
% 
% %% 5. Train Network 
% net = trainNetwork(X_train, Y_train, layers, options);
% 
% Y_pred_raw = predict(net, X_test);
% 
% % Convert back to original scale
% Y_pred = Y_pred_raw * clean_std + clean_mean;
% Y_test_orig = Y_test * clean_std + clean_mean; 
% 
% %% 6. Plot Results - Random Samples of Type A-F (FIXED)
% % Find Type A-F samples in test set
% type_a_test_indices = find(mod(test_indices-1, 6) == 0);  % Type A samples
% type_b_test_indices = find(mod(test_indices-1, 6) == 1);  % Type B samples
% type_c_test_indices = find(mod(test_indices-1, 6) == 2);  % Type C samples
% type_d_test_indices = find(mod(test_indices-1, 6) == 3);  % Type D samples
% type_e_test_indices = find(mod(test_indices-1, 6) == 4);  % Type E samples
% type_f_test_indices = find(mod(test_indices-1, 6) == 5);  % Type F samples
% 
% % Store all test indices for each type
% all_type_test_indices = {type_a_test_indices, type_b_test_indices, type_c_test_indices, ...
%                          type_d_test_indices, type_e_test_indices, type_f_test_indices};
% type_labels = {'A', 'B', 'C', 'D', 'E', 'F'};
% 
% % Plot one random sample from each type
% for type_idx = 1:6
%     type_test_indices_current = all_type_test_indices{type_idx};
% 
%     if ~isempty(type_test_indices_current)
%         % Select a random position from this type's test indices
%         random_position = type_test_indices_current(randi(length(type_test_indices_current)));
% 
%         % Extract signals (convert back to original scale)
%         clean_signal = squeeze(Y_test(:,1,1,random_position)) * clean_std + clean_mean;
%         noisy_signal = squeeze(X_test(:,1,1,random_position)) * noisy_std + noisy_mean;
%         denoised_signal = squeeze(Y_pred(:,1,1,random_position));
% 
%         % Ensure vectors are the same orientation as t
%         clean_signal = clean_signal(:)';    % Convert to row vector
%         noisy_signal = noisy_signal(:)';    % Convert to row vector
%         denoised_signal = denoised_signal(:)'; % Convert to row vector
% 
%         % Create plot for this type
%         figure;
%         subplot(3,1,1); plot(t, clean_signal); 
%         title(['Type ' type_labels{type_idx} ' - Clean (' type_names{type_idx} ')']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
% 
%         subplot(3,1,2); plot(t, noisy_signal); 
%         title(['Type ' type_labels{type_idx} ' - Noisy']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
% 
%         subplot(3,1,3); plot(t, denoised_signal); 
%         title(['Type ' type_labels{type_idx} ' - Denoised (CNN)']); 
%         xlabel('Time (s)'); ylabel('Amplitude');
%     end
% end
% 
% %% 7. Evaluation 
% X_test_orig = X_test * noisy_std + noisy_mean;
% 
% clean_power = mean(Y_test_orig(:).^2);
% noise_power = mean((Y_test_orig(:)-X_test_orig(:)).^2);
% snr_before = 10*log10(clean_power/noise_power);
% 
% % CNN output metrics
% residual_cnn = Y_test_orig(:) - Y_pred(:);
% mse_cnn = mean(residual_cnn.^2);
% snr_cnn = 10*log10(clean_power/mean(residual_cnn.^2));
% cc_cnn = corrcoef(Y_test_orig(:), Y_pred(:));
% cc_cnn = cc_cnn(1,2);
% 
% % Display comprehensive results
% fprintf('=== CNN Denoising Performance ===\n');
% fprintf('                |  Before  |   CNN   \n');
% fprintf('------------------------------------\n');
% fprintf('MSE             |    -     | %.4f  \n', mse_cnn);
% fprintf('SNR (dB)        | %6.2f   | %6.2f  \n',snr_before, snr_cnn);
% fprintf('CC              |    -     | %.4f  \n', cc_cnn);
% fprintf('Improvement (dB)| -        | %6.2f  \n', snr_cnn - snr_before);
% 
% %% 8. Evaluate on 10 Random Test Samples
% num_eval_samples = 10;
% random_indices = randperm(length(test_indices), num_eval_samples);
% avg_snr_before = 0;
% avg_mse_cnn = 0;
% avg_snr_cnn = 0;
% avg_cc_cnn = 0;
% 
% figure;
% for idx = 1:num_eval_samples
%     test_idx = random_indices(idx);
% 
%     x = X_test(:,1,1,test_idx);
%     y_true = Y_test(:,1,1,test_idx);
%     y_cnn = Y_pred(:,1,1,test_idx);
% 
%     % Metrics
%     mse_cnn_i = mean((y_true - y_cnn).^2);
%     snr_cnn_i = 10*log10(mean(y_true.^2) / mse_cnn_i);
%     cc_cnn_i = corrcoef(y_true, y_cnn); cc_cnn_i = cc_cnn_i(1,2);
% 
%     % Accumulate for average
%     avg_snr_before = avg_snr_before + snr_before;
%     avg_mse_cnn = avg_mse_cnn + mse_cnn_i;
%     avg_snr_cnn = avg_snr_cnn + snr_cnn_i;
%     avg_cc_cnn = avg_cc_cnn + cc_cnn_i;
% 
%     % Plot
%     subplot(num_eval_samples, 3, (idx-1)*3 + 1);
%     plot(t, x); title(sprintf('Noisy #%d', idx)); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 3, (idx-1)*3 + 2);
%     plot(t, y_true); title('Clean'); ylabel('Amplitude');
% 
%     subplot(num_eval_samples, 3, (idx-1)*3 + 3);
%     plot(t, y_cnn); title('CNN'); ylabel('Amplitude');
% end
% 
% % Take averages
% avg_snr_before = avg_snr_before / num_eval_samples;
% avg_mse_cnn = avg_mse_cnn / num_eval_samples;
% avg_snr_cnn = avg_snr_cnn / num_eval_samples;
% avg_cc_cnn = avg_cc_cnn / num_eval_samples;
% 
% % Display results
% fprintf('\n=== 10-Sample Random Evaluation ===\n');
% fprintf('MSE        | CNN: %.4f\n', avg_mse_cnn);
% fprintf('SNR (dB)   | CNN: %.2f\n', avg_snr_cnn);
% fprintf('Corr Coef  | CNN: %.4f\n', avg_cc_cnn);
% 
% % Compare with overall results
% fprintf('\n=== Comparison with Overall Results ===\n');
% fprintf('                     | Overall Eval | 10-Sample Avg\n');
% fprintf('-----------------------------------------------------\n');
% fprintf('SNR Before (dB)      | %6.2f       | %6.2f\n', snr_before, avg_snr_before);
% fprintf('SNR CNN (dB)         | %6.2f       | %6.2f\n', snr_cnn, avg_snr_cnn);
% fprintf('CC CNN               | %6.4f       | %6.4f\n', cc_cnn, avg_cc_cnn);
% 
% %% 9. Visual comparison of denoised signals for Type A-F (FIXED)
% % Find representative samples of each type from the test set
% type_a_indices = find(mod(test_indices-1, 6) == 0);  % Type A
% type_b_indices = find(mod(test_indices-1, 6) == 1);  % Type B
% type_c_indices = find(mod(test_indices-1, 6) == 2);  % Type C
% type_d_indices = find(mod(test_indices-1, 6) == 3);  % Type D
% type_e_indices = find(mod(test_indices-1, 6) == 4);  % Type E
% type_f_indices = find(mod(test_indices-1, 6) == 5);  % Type F
% 
% % Select the first one of each type (if available)
% sample_indices = [];
% sample_types = {};
% all_type_indices = {type_a_indices, type_b_indices, type_c_indices, ...
%                     type_d_indices, type_e_indices, type_f_indices};
% 
% for type_idx = 1:6
%     if ~isempty(all_type_indices{type_idx})
%         sample_indices(end+1) = all_type_indices{type_idx}(1);
%         sample_types{end+1} = ['Type ' type_labels{type_idx}];
%     end
% end
% 
% % Create the comparison figure - Split into 2 figures for better visibility
% types_per_figure = 3;
% 
% for fig_num = 1:2
%     figure('Position', [100 + (fig_num-1)*50, 100 + (fig_num-1)*50, 1200, 800]);
% 
%     start_type = (fig_num-1) * types_per_figure + 1;
%     end_type = min(fig_num * types_per_figure, length(sample_indices));
% 
%     for local_s = 1:(end_type - start_type + 1)
%         s = start_type + local_s - 1;
%         idx = sample_indices(s);
% 
%         % Extract and ensure proper orientation
%         noisy_signal = squeeze(X_test(:,1,1,idx)) * noisy_std + noisy_mean;
%         clean_signal = squeeze(Y_test(:,1,1,idx)) * clean_std + clean_mean;
%         cnn_signal = squeeze(Y_pred(:,1,1,idx));
% 
%         % Convert to row vectors to match t
%         noisy_signal = noisy_signal(:)';
%         clean_signal = clean_signal(:)';
%         cnn_signal = cnn_signal(:)';
% 
%         % Plot original signals
%         subplot(types_per_figure, 3, (local_s-1)*3 + 1);
%         plot(t, noisy_signal, 'r'); hold on;
%         plot(t, clean_signal, 'g');
%         title([sample_types{s} ': Original Signals']);
%         legend('Noisy', 'Clean');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
% 
%         % Plot CNN denoised signal
%         subplot(types_per_figure, 3, (local_s-1)*3 + 2);
%         plot(t, clean_signal, 'g'); hold on;
%         plot(t, cnn_signal, 'b', 'LineWidth', 1);
%         title([sample_types{s} ': CNN Denoising']);
%         legend('Clean', 'CNN');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
% 
%         % Plot zoomed-in section for detail
%         subplot(types_per_figure, 3, (local_s-1)*3 + 3);
% 
%         % Define zoom ranges based on signal type
%         if contains(sample_types{s}, 'A')  % Type A - sparse
%             zoom_range = round(0.15e-6 * fs):round(0.35e-6 * fs);
%         elseif contains(sample_types{s}, 'B')  % Type B - spike-dense
%             zoom_range = round(0.55e-6 * fs):round(0.75e-6 * fs);
%         elseif contains(sample_types{s}, 'C')  % Type C - 10mm
%             zoom_range = round(0.2e-6 * fs):round(0.4e-6 * fs);
%         elseif contains(sample_types{s}, 'D')  % Type D - 18mm
%             zoom_range = round(0.4e-6 * fs):round(0.6e-6 * fs);
%         elseif contains(sample_types{s}, 'E')  % Type E - 20mm
%             zoom_range = round(0.3e-6 * fs):round(0.5e-6 * fs);
%         else  % Type F - 25mm
%             zoom_range = round(0.2e-6 * fs):round(0.4e-6 * fs);
%         end
% 
%         % Ensure zoom_range is within bounds
%         zoom_range = zoom_range(zoom_range >= 1 & zoom_range <= length(t));
% 
%         plot(t(zoom_range), clean_signal(zoom_range), 'g', 'LineWidth', 2); hold on;
%         plot(t(zoom_range), cnn_signal(zoom_range), 'b', 'LineWidth', 1);
%         title([sample_types{s} ': Zoomed Detail']);
%         legend('Clean', 'CNN');
%         xlabel('Time (s)');
%         ylabel('Amplitude');
%     end
% 
%     % Add figure title
%     if fig_num == 1
%         sgtitle('Signal Type Comparison - Types A, B, C', 'FontSize', 16, 'FontWeight', 'bold');
%     else
%         sgtitle('Signal Type Comparison - Types D, E, F', 'FontSize', 16, 'FontWeight', 'bold');
%     end
% end
% 
% %% 10. Calculate performance metrics for each signal type
% % Get test indices for each type
% type_a_test = test_indices(mod(test_indices-1, 6) == 0);
% type_b_test = test_indices(mod(test_indices-1, 6) == 1);
% type_c_test = test_indices(mod(test_indices-1, 6) == 2);
% type_d_test = test_indices(mod(test_indices-1, 6) == 3);
% type_e_test = test_indices(mod(test_indices-1, 6) == 4);
% type_f_test = test_indices(mod(test_indices-1, 6) == 5);
% 
% % Initialize arrays to store metrics
% num_types = 6;
% type_names_eval = {'Type A: Sparse PD pulses', 'Type B: Spike-dense signal', ...
%                    'Type C: 10mm', 'Type D: 18mm', 'Type E: 20mm', 'Type F: 25mm'};
% type_test_indices = {type_a_test, type_b_test, type_c_test, type_d_test, type_e_test, type_f_test};
% 
% snr_before_types = zeros(num_types, 1);
% snr_improvements = zeros(num_types, 1);
% cc_values = zeros(num_types, 1);
% mse_values = zeros(num_types, 1);
% 
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
% 
%     if isempty(type_test)
%         fprintf('Warning: No test samples found for %s\n', type_names_eval{type_idx});
%         continue;
%     end
% 
%     % Find indices in the test set
%     type_indices_in_test = find(ismember(test_indices, type_test));
% 
%     if isempty(type_indices_in_test)
%         continue;
%     end
% 
%     % Extract signals for this type
%     x_type = X_test(:,:,:,type_indices_in_test) * noisy_std + noisy_mean;
%     y_type = Y_test(:,:,:,type_indices_in_test) * clean_std + clean_mean;
%     y_pred_type = Y_pred(:,:,:,type_indices_in_test) * clean_std + clean_mean;
% 
%     % Calculate SNR before denoising
%     clean_power_type = mean(y_type(:).^2);
%     noise_power_type = mean((y_type(:) - x_type(:)).^2);
%     snr_before_types(type_idx) = 10*log10(clean_power_type/noise_power_type);
% 
%     % CNN metrics
%     residual_cnn_type = y_type(:) - y_pred_type(:);
%     mse_cnn_type = mean(residual_cnn_type.^2);
%     snr_cnn_type = 10*log10(clean_power_type/mse_cnn_type);
%     cc_cnn_type = corrcoef(y_type(:), y_pred_type(:));
%     cc_cnn_type = cc_cnn_type(1,2);
% 
%     % Store improvements and correlation coefficients
%     snr_improvements(type_idx) = snr_cnn_type - snr_before_types(type_idx);
%     cc_values(type_idx) = cc_cnn_type;
%     mse_values(type_idx) = mse_cnn_type;
% end
% 
% % Display comprehensive results by signal type
% fprintf('\n=== CNN Denoising Performance by Signal Type ===\n');
% fprintf('Signal Type                | Initial SNR (dB) | Samples\n');
% fprintf('----------------------------------------------------------\n');
% for type_idx = 1:num_types
%     type_test = type_test_indices{type_idx};
%     fprintf('%-26s | %8.2f        | %7d\n', type_names_eval{type_idx}, ...
%         snr_before_types(type_idx), length(type_test));
% end
% 
% fprintf('\n=== SNR Improvement and Correlation by Signal Type ===\n');
% fprintf('Signal Type                |               CNN               |\n');
% fprintf('Evaluation                 |     SNR      |     CC    |    MSE    |\n');
% fprintf('--------------------------------------------------------------------\n');
% for type_idx = 1:num_types
%     fprintf('%-26s | %8.2f dB  | %8.4f  | %8.5f  |\n', ...
%         type_names_eval{type_idx}, snr_improvements(type_idx), cc_values(type_idx), mse_values(type_idx));
% end
% 
% % Calculate and display average improvements
% avg_snr_improvement_cnn = mean(snr_improvements);
% avg_cc_cnn_overall = mean(cc_values);
% 
% fprintf('\n=== Average Performance Across All Types ===\n');
% fprintf('Method |   Avg SNR    |   Avg CC\n');
% fprintf('-------------------------------\n');
% fprintf('CNN    | %8.2f dB  | %8.4f\n', avg_snr_improvement_cnn, avg_cc_cnn_overall);
% 
% %% 11. Save Results with Error Handling
% try
%     % First, verify that the network exists and is valid
%     if ~exist('net', 'var') || isempty(net)
%         error('Network variable is empty or not defined');
%     end
% 
%     % Save the comprehensive results file
%     save('cnn_ABCDEF_v2_result.mat', 'net', 'Y_pred', ...
%          'mse_cnn', 'snr_before', 'snr_cnn', 'cc_cnn', ...
%          'snr_improvements', 'cc_values', 'mse_values', 'type_names_eval', ...
%          'snr_before_types', 'avg_snr_improvement_cnn');
%     fprintf('Comprehensive results saved successfully to cnn_ABCDEF_v2_result.mat\n');
% 
%     % Save just the network with verification
%     save('cnn_ABCDEF_v2.mat', 'net');
% 
%     % Verify the saved file
%     fileInfo = dir('cnn_ABCDEF_v2.mat');
%     if isempty(fileInfo) || fileInfo.bytes == 0
%         error('Failed to save network: File is empty');
%     else
%         fprintf('Network saved successfully to cnn_ABCDEF_v2.mat (%d bytes)\n', fileInfo.bytes);
% 
%         % Double-check by trying to load it
%         testLoad = load('cnn_ABCDEF_v2.mat');
%         if ~isfield(testLoad, 'net')
%             error('Saved file does not contain the network variable');
%         else
%             fprintf('Verified: Network was saved correctly and can be loaded\n');
%         end
%     end
% 
%     % Save detailed performance metrics
%     performance_summary = struct();
%     performance_summary.type_names = type_names_eval;
%     performance_summary.snr_before = snr_before_types;
%     performance_summary.snr_improvements = snr_improvements;
%     performance_summary.correlation_coefficients = cc_values;
%     performance_summary.mse_values = mse_values;
%     performance_summary.avg_snr_improvement_cnn = avg_snr_improvement_cnn;
%     performance_summary.avg_cc_cnn = avg_cc_cnn_overall;
% 
%     save('performance_summary_ABCDEF_v2.mat', 'performance_summary');
%     fprintf('Performance summary saved to performance_summary_ABCDEF_v2.mat\n');
% 
% catch ME
%     fprintf('ERROR saving results: %s\n', ME.message);
%     % Try an alternative save location
%     try
%         alternativePath = fullfile(pwd, 'cnn_ABCDEF_v2_backup.mat');
%         save(alternativePath, 'net');
%         fprintf('Network saved to alternative location: %s\n', alternativePath);
%     catch
%         fprintf('Failed to save to alternative location as well\n');
%     end
% end
% 
% %% 12. Final Summary
% fprintf('\n=== FINAL SUMMARY ===\n');
% fprintf('Dataset: %d samples with 6 signal types (A, B, C, D, E, F)\n', num_samples);
% fprintf('Signal length: %d samples (%.1f μs at %.0f MHz)\n', signal_length, t_total*1e6, fs/1e6);
% fprintf('Network: CNN only\n');
% fprintf('Training epochs: %d\n', options.MaxEpochs);
% fprintf('\nSignal Types:\n');
% for i = 1:6
%     fprintf('• %s\n', type_names_eval{i});
% end
% fprintf('\nOverall Performance:\n');
% fprintf('• Average SNR improvement: %.2f dB\n', avg_snr_improvement_cnn);
% fprintf('• Average correlation: %.4f\n', avg_cc_cnn_overall);
% 
% fprintf('\nBest performing signal type:\n');
% [~, best_cnn_idx] = max(snr_improvements);
% fprintf('• CNN: %s (%.2f dB improvement)\n', type_names_eval{best_cnn_idx}, snr_improvements(best_cnn_idx));
% 
% fprintf('\nFiles saved:\n');
% fprintf('• cnn_ABCDEF_v2.mat (trained network)\n');
% fprintf('• cnn_ABCDEF_v2_result.mat (complete results)\n');
% fprintf('• performance_summary_ABCDEF.mat (performance metrics)\n');
% fprintf('\n=== ANALYSIS COMPLETE ===\n');
% fs = 1000e6;                  % 1 GHz sampling
% t_total = 2e-6;               % 2 μs duration
% t = 0:1/fs:t_total-1/fs;      % Time vector
% signal_length = length(t);
% num_samples = 100;            % Generate 25 samples per type for training dataset
% 
% clean_signals = zeros(num_samples, signal_length);
% noisy_signals = zeros(num_samples, signal_length);
% signal_labels = zeros(num_samples, 1);  % Labels for training: 1=A, 2=B, 3=C, 4=D
% 
% fprintf('Generating %d samples with %d points each for training dataset...\n', num_samples, signal_length);
% 
% for i = 1:num_samples
%     % Cycle through Type A, Type B, Type C, and Type D
%     signal_type = mod(i-1, 4) + 1;  % Types 1,2,3,4 correspond to A,B,C,D
%     signal_labels(i) = signal_type;
% 
%     if signal_type == 1  % Type A: 10mm
%         clean_signal = zeros(size(t));
% 
%         % Very sparse PD events with random locations
%         num_events = 5 + randi(5);  % 3-7 events (very sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Clear, distinct bipolar spikes
%                 amplitude = 2.5 + rand() * 2;  % 2.5-4.5 amplitude
%                 polarity = (-1)^randi([0 1]);  % Random polarity
% 
%                 % Sharp bipolar pulse
%                 spike_width = 3 + randi(3);  % 3-6 samples wide
% 
%                 if start_idx + spike_width - 1 <= length(clean_signal)
%                     % Main spike
%                     clean_signal(start_idx) = polarity * amplitude;
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.8;
%                     end
% 
%                     % Decay tail
%                     for j = 2:spike_width-1
%                         if start_idx + j <= length(clean_signal)
%                             clean_signal(start_idx + j) = polarity * amplitude * 0.3 * exp(-(j-1));
%                         end
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 2  % Type B: 18mm(slightly more than A)
%         clean_signal = zeros(size(t));
% 
%         % Sparse PD events with random locations
%         num_events = 15 + randi(10);  % 10-25 events (sparse)
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 % Mixed event types with random characteristics
%                 event_type = rand();
%                 amplitude = 1.5 + rand() * 3;  % 1.5-4.5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 if event_type < 0.7  % 70% - Sharp bipolar spikes
%                     spike_width = 2 + randi(4);
% 
%                     if start_idx + spike_width - 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         if start_idx + 1 <= length(clean_signal)
%                             clean_signal(start_idx + 1) = -polarity * amplitude * 0.7;
%                         end
% 
%                         % Add some oscillatory tail
%                         for j = 2:spike_width-1
%                             if start_idx + j <= length(clean_signal)
%                                 clean_signal(start_idx + j) = polarity * amplitude * 0.2 * sin(j);
%                             end
%                         end
%                     end
% 
%                 else  % 30% - Short oscillatory bursts
%                     fc = 30e6 + rand() * 50e6;  % 30-80 MHz
%                     burst_duration = 15e-9 + rand() * 25e-9;  % 15-40 ns
%                     burst_samples = round(burst_duration * fs);
% 
%                     if start_idx + burst_samples - 1 <= length(clean_signal)
%                         burst_time = (0:burst_samples-1) / fs;
%                         envelope = exp(-burst_time / (burst_duration * 0.3));
%                         oscillation = sin(2*pi*fc*burst_time);
%                         burst_signal = polarity * amplitude * envelope .* oscillation;
% 
%                         clean_signal(start_idx:start_idx + burst_samples - 1) = burst_signal;
%                     end
%                 end
%             end
%         end
% 
%     elseif signal_type == 3  % Type C: 20mm
%         clean_signal = zeros(size(t));
% 
%         % Moderate to high frequency PD events with random locations
%         num_events = 40 + randi(35);  % 40-75 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 1 + rand() * 4;  % 1-5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.4  % 40% - Sharp spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.6;
%                     end
% 
%                 elseif event_type < 0.7  % 30% - Oscillatory bursts
%                     fc = 25e6 + rand() * 60e6;
%                     burst_duration = 10e-9 + rand() * 30e-9;
%                     burst_samples = round(burst_duration * fs);
% 
%                     if start_idx + burst_samples - 1 <= length(clean_signal)
%                         burst_time = (0:burst_samples-1) / fs;
%                         envelope = exp(-burst_time / (burst_duration * 0.4));
%                         oscillation = sin(2*pi*fc*burst_time);
%                         burst_signal = polarity * amplitude * envelope .* oscillation;
% 
%                         clean_signal(start_idx:start_idx + burst_samples - 1) = burst_signal;
%                     end
% 
%                 else  % 30% - Multi-frequency transients
%                     fc1 = 20e6 + rand() * 40e6;
%                     fc2 = 50e6 + rand() * 50e6;
%                     event_duration = 8e-9 + rand() * 20e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples - 1 <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.3));
% 
%                         component1 = 0.6 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.4 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
% 
%                         transient_signal = polarity * (component1 + component2);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = transient_signal;
%                     end
%                 end
%             end
%         end
% 
%         % Add moderate background activity at random locations
%         background_level = 0.3;
%         background_density = 0.4;  % 40% of time points
%         background_points = rand(size(clean_signal)) < background_density;
%         background_signal = background_level * (rand(size(clean_signal)) - 0.5) * 2;
%         clean_signal = clean_signal + background_signal .* background_points;
% 
%     else  % Type D: 25mm
%         clean_signal = zeros(size(t));
% 
%         % 25mm events with random locations throughout
%         num_events = 100 + randi(80);  % 100-180 events
% 
%         % Generate completely random event times
%         event_times = sort(rand(1, num_events) * t_total);
% 
%         for e = 1:length(event_times)
%             event_time = event_times(e);
%             start_idx = round(event_time * fs) + 1;
% 
%             if start_idx <= length(clean_signal)
%                 amplitude = 0.8 + rand() * 4.2;  % 0.8-5 range
%                 polarity = (-1)^randi([0 1]);
% 
%                 event_type = rand();
% 
%                 if event_type < 0.3  % 30% - Quick spikes
%                     if start_idx + 1 <= length(clean_signal)
%                         clean_signal(start_idx) = polarity * amplitude;
%                         clean_signal(start_idx + 1) = -polarity * amplitude * 0.5;
%                     end
% 
%                 elseif event_type < 0.6  % 30% - Short bursts
%                     fc = 40e6 + rand() * 80e6;  % Higher frequency
%                     burst_duration = 8e-9 + rand() * 15e-9;  % Shorter bursts
%                     burst_samples = round(burst_duration * fs);
% 
%                     if start_idx + burst_samples - 1 <= length(clean_signal)
%                         burst_time = (0:burst_samples-1) / fs;
%                         envelope = exp(-burst_time / (burst_duration * 0.2));
%                         oscillation = sin(2*pi*fc*burst_time);
%                         burst_signal = polarity * amplitude * envelope .* oscillation;
% 
%                         clean_signal(start_idx:start_idx + burst_samples - 1) = burst_signal;
%                     end
% 
%                 else  % 40% - Complex multi-component events
%                     % Multiple frequency components
%                     fc1 = 30e6 + rand() * 50e6;
%                     fc2 = 60e6 + rand() * 60e6;
%                     fc3 = 100e6 + rand() * 50e6;  % Add third component
% 
%                     event_duration = 5e-9 + rand() * 15e-9;
%                     event_samples = round(event_duration * fs);
% 
%                     if start_idx + event_samples - 1 <= length(clean_signal)
%                         event_time_vec = (0:event_samples-1) / fs;
%                         envelope = exp(-event_time_vec / (event_duration * 0.2));
% 
%                         component1 = 0.4 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
%                         component2 = 0.3 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
%                         component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);
% 
%                         complex_signal = polarity * (component1 + component2 + component3);
%                         clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
%                     end
%                 end
%             end
%         end
% 
%         % Very dense background activity at random locations
%         background_level = 0.5;
%         background_density = 0.7;  % 70% of time points
%         background_points = rand(size(clean_signal)) < background_density;
%         background_signal = background_level * (rand(size(clean_signal)) - 0.5) * 2;
%         clean_signal = clean_signal + background_signal .* background_points;
% 
%         % Add continuous low-level oscillations at random phases
%         continuous_osc1 = 0.2 * sin(2*pi*35e6*t + rand()*2*pi);
%         continuous_osc2 = 0.15 * sin(2*pi*75e6*t + rand()*2*pi);
%         clean_signal = clean_signal + continuous_osc1 + continuous_osc2;
%     end
% 
%     % Keep signals in a reasonable range but preserve relative amplitudes
%     max_amplitude = max(abs(clean_signal));
%     if max_amplitude > 0
%         if max_amplitude > 5
%             clean_signal = clean_signal * (5 / max_amplitude);
%         end
%     end
% 
%     % Add realistic noise with random characteristics
%     white_noise = (0.1 + rand()*0.1) * randn(size(clean_signal));  % Random noise level
%     powerline_noise = (0.03 + rand()*0.04) * sin(2*pi*50e6*t + rand()*2*pi) + ...
%                      (0.02 + rand()*0.03) * sin(2*pi*150e6*t + rand()*2*pi);
%     narrowband = (0.04 + rand()*0.04) * sin(2*pi*(60e6 + rand()*40e6)*t + rand()*2*pi);
% 
%     % Random impulse noise
%     impulse_noise = zeros(size(clean_signal));
%     num_impulses = randi([5, 20]);  % Random number of impulses
%     if num_impulses <= length(clean_signal)
%         spike_pos = randperm(length(clean_signal), num_impulses);
%         impulse_noise(spike_pos) = (0.3 + rand(1,num_impulses)*0.4) .* (rand(1,num_impulses) > 0.5) .* ...
%                                   (2*randi([0,1], 1, num_impulses) - 1);  % Random polarity
%     end
% 
%     noise = white_noise + powerline_noise + narrowband + impulse_noise;
%     noisy_signal = clean_signal + noise;
% 
%     clean_signals(i,:) = clean_signal;
%     noisy_signals(i,:) = noisy_signal;
% 
%     if mod(i, 25) == 0
%         fprintf('Generated %d samples...\n', i);
%     end
% end
% 
% %% Visualization of All Four Signal Types (A, B, C, D)
% 
% figure('Position', [50, 50, 1600, 1000]);
% 
% % Get representative samples of each type
% typeA_idx = 1;   % First sample (Type A)
% typeB_idx = 2;   % Second sample (Type B) 
% typeC_idx = 3;   % Third sample (Type C)
% typeD_idx = 4;   % Fourth sample (Type D)
% 
% % Time vector in microseconds for x-axis
% t_us = t * 1e6;
% 
% % Check signal statistics first
% fprintf('\n=== Signal Statistics ===\n');
% fprintf('Type A: max=%.3f, active=%.1f%%\n', max(abs(clean_signals(typeA_idx,:))), ...
%     100*mean(abs(clean_signals(typeA_idx,:)) > 0.1));
% fprintf('Type B: max=%.3f, active=%.1f%%\n', max(abs(clean_signals(typeB_idx,:))), ...
%     100*mean(abs(clean_signals(typeB_idx,:)) > 0.1));
% fprintf('Type C: max=%.3f, active=%.1f%%\n', max(abs(clean_signals(typeC_idx,:))), ...
%     100*mean(abs(clean_signals(typeC_idx,:)) > 0.1));
% fprintf('Type D: max=%.3f, active=%.1f%%\n', max(abs(clean_signals(typeD_idx,:))), ...
%     100*mean(abs(clean_signals(typeD_idx,:)) > 0.1));
% 
% %% Plot Clean Signals
% subplot(2,4,1);
% plot(t_us, clean_signals(typeA_idx,:), 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
% title('Type A: 10mm (Clean)', 'FontSize', 12, 'FontWeight', 'bold');
% xlabel('Time (μs)');
% ylabel('Amplitude');
% grid on;
% xlim([0 2]);
% ylim([-5 5]);
% 
% subplot(2,4,2);
% plot(t_us, clean_signals(typeB_idx,:), 'LineWidth', 1.5, 'Color', [0.8500 0.3250 0.0980]);
% title('Type B: 18mm(Clean)', 'FontSize', 12, 'FontWeight', 'bold');
% xlabel('Time (μs)');
% ylabel('Amplitude');
% grid on;
% xlim([0 2]);
% ylim([-5 5]);
% 
% subplot(2,4,3);
% plot(t_us, clean_signals(typeC_idx,:), 'LineWidth', 1.5, 'Color', [0.9290 0.6940 0.1250]);
% title('Type C: 20mm(Clean)', 'FontSize', 12, 'FontWeight', 'bold');
% xlabel('Time (μs)');
% ylabel('Amplitude');
% grid on;
% xlim([0 2]);
% ylim([-5 5]);
% 
% subplot(2,4,4);
% plot(t_us, clean_signals(typeD_idx,:), 'LineWidth', 1.5, 'Color', [0.4940 0.1840 0.5560]);
% title('Type D: 25mm (Clean)', 'FontSize', 12, 'FontWeight', 'bold');
% xlabel('Time (μs)');
% ylabel('Amplitude');
% grid on;
% xlim([0 2]);
% ylim([-5 5]);
% 
% %% Plot Noisy Signals
% subplot(2,4,5);
% plot(t_us, noisy_signals(typeA_idx,:), 'LineWidth', 1, 'Color', [0 0.4470 0.7410]);
% title('Type A: 10mm (Noisy)', 'FontSize', 12);
% xlabel('Time (μs)');
% ylabel('Amplitude');
% grid on;
% xlim([0 2]);
% ylim([-6 6]);
% 
% subplot(2,4,6);
% plot(t_us, noisy_signals(typeB_idx,:), 'LineWidth', 1, 'Color', [0.8500 0.3250 0.0980]);
% title('Type B: 18mm(Noisy)', 'FontSize', 12);
% xlabel('Time (μs)');
% ylabel('Amplitude');
% grid on;
% xlim([0 2]);
% ylim([-6 6]);
% 
% subplot(2,4,7);
% plot(t_us, noisy_signals(typeC_idx,:), 'LineWidth', 1, 'Color', [0.9290 0.6940 0.1250]);
% title('Type C: 20mm(Noisy)', 'FontSize', 12);
% xlabel('Time (μs)');
% ylabel('Amplitude');
% grid on;
% xlim([0 2]);
% ylim([-6 6]);
% 
% subplot(2,4,8);
% plot(t_us, noisy_signals(typeD_idx,:), 'LineWidth', 1, 'Color', [0.4940 0.1840 0.5560]);
% title('Type D: 25mm (Noisy)', 'FontSize', 12);
% xlabel('Time (μs)');
% ylabel('Amplitude');
% grid on;
% xlim([0 2]);
% ylim([-6 6]);
% 
% sgtitle('PD Signal Training Dataset - Types A, B, C, D (Random PD Locations)', 'FontSize', 16, 'FontWeight', 'bold');
% 
% %% Dataset Statistics Summary
% fprintf('\n=== Training Dataset Statistics ===\n');
% 
% % Separate samples by type
% typeA_samples = clean_signals(signal_labels == 1, :);
% typeB_samples = clean_signals(signal_labels == 2, :);
% typeC_samples = clean_signals(signal_labels == 3, :);
% typeD_samples = clean_signals(signal_labels == 4, :);
% 
% activity_threshold = 0.1;
% typeA_activity = mean(abs(typeA_samples(:)) > activity_threshold) * 100;
% typeB_activity = mean(abs(typeB_samples(:)) > activity_threshold) * 100;
% typeC_activity = mean(abs(typeC_samples(:)) > activity_threshold) * 100;
% typeD_activity = mean(abs(typeD_samples(:)) > activity_threshold) * 100;
% 
% fprintf('\nType A (10mm):\n');
% fprintf('  Samples: %d\n', sum(signal_labels == 1));
% fprintf('  Activity level: %.1f%%\n', typeA_activity);
% fprintf('  Max amplitude: %.2f\n', max(abs(typeA_samples(:))));
% fprintf('  RMS value: %.3f\n', rms(typeA_samples(:)));
% 
% fprintf('\nType B (Sparse PD Activity):\n');
% fprintf('  Samples: %d\n', sum(signal_labels == 2));
% fprintf('  Activity level: %.1f%%\n', typeB_activity);
% fprintf('  Max amplitude: %.2f\n', max(abs(typeB_samples(:))));
% fprintf('  RMS value: %.3f\n', rms(typeB_samples(:)));
% 
% fprintf('\nType C (Moderate-High PD Activity):\n');
% fprintf('  Samples: %d\n', sum(signal_labels == 3));
% fprintf('  Activity level: %.1f%%\n', typeC_activity);
% fprintf('  Max amplitude: %.2f\n', max(abs(typeC_samples(:))));
% fprintf('  RMS value: %.3f\n', rms(typeC_samples(:)));
% 
% fprintf('\nType D (25mm):\n');
% fprintf('  Samples: %d\n', sum(signal_labels == 4));
% fprintf('  Activity level: %.1f%%\n', typeD_activity);
% fprintf('  Max amplitude: %.2f\n', max(abs(typeD_samples(:))));
% fprintf('  RMS value: %.3f\n', rms(typeD_samples(:)));
% 
% fprintf('\n=== Dataset Summary ===\n');
% fprintf('Total samples: %d (%d per type)\n', num_samples, num_samples/4);
% fprintf('Signal length: %d points\n', signal_length);
% fprintf('Sampling rate: %.0f MHz\n', fs/1e6);
% fprintf('Total duration: %.1f μs\n', t_total*1e6);
% fprintf('PD locations: Completely randomized for each sample\n');
% 
% %% Save training dataset
% save('pd_training_dataset.mat', 'clean_signals', 'noisy_signals', 'signal_labels', 'fs', 't_total', 't');
% fprintf('\nTraining dataset saved to pd_training_dataset.mat\n');
% fprintf('Variables saved: clean_signals, noisy_signals, signal_labels, fs, t_total, t\n');

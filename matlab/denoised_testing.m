clc; clear;

%% Deep Learning Approaches 
% %% Real Signal testing
% clc; clear;
% 
% % 1. Load Real PD Signals
% load('clean_data_18mm.mat');           % clean_data_18mm [samples x time]
% load('noisy_minus5dB_18mm.mat');       % noisy_data_minus5dB_18mm [samples x time]
% load('cnn_ABCDEF.mat');                % trained CNN model (variable: net)
% 
% % 2. Parameters
% fs = 1000e6;                            % 1 GHz sampling
% expected_length = 2001;                % CNN expects this input size
% 
% % 3. Select a sample
% idx = 1;  % sample index to test
% clean_signal = clean_data_18mm(idx, :);
% noisy_signal = noisy_data_minus5dB_18mm(idx, :);
% 
% % --- Pad or trim both signals to match expected CNN input length ---
% if length(noisy_signal) > expected_length
%     noisy_signal = noisy_signal(1:expected_length);
% else
%     noisy_signal = [noisy_signal, zeros(1, expected_length - length(noisy_signal))];
% end
% 
% if length(clean_signal) > expected_length
%     clean_signal = clean_signal(1:expected_length);
% else
%     clean_signal = [clean_signal, zeros(1, expected_length - length(clean_signal))];
% end
% 
% % --- CNN Prediction ---
% input_signal = reshape(noisy_signal, [expected_length, 1, 1]);
% denoised_signal = predict(net, input_signal);
% denoised_signal = reshape(denoised_signal, [1, expected_length]);
% 
% % 4. Generate time vector (must match signal length)
% t = linspace(0, (expected_length - 1) / fs, expected_length);
% 
% % 5. Evaluation Metrics
% SNR_before = 10 * log10(sum(clean_signal.^2) / sum((noisy_signal - clean_signal).^2));
% SNR_after  = 10 * log10(sum(clean_signal.^2) / sum((denoised_signal - clean_signal).^2));
% MSE        = mean((clean_signal - denoised_signal).^2);
% CC         = sum(clean_signal .* denoised_signal) / sqrt(sum(clean_signal.^2) * sum(denoised_signal.^2));
% 
% % 6. Display Results
% fprintf('=== Real Signal Denoising ===\n');
% fprintf('SNR before denoising: %.2f dB\n', SNR_before);
% fprintf('SNR after denoising : %.2f dB\n', SNR_after);
% fprintf('Mean Square Error (MSE): %.6f\n', MSE);
% fprintf('Correlation Coefficient (CC): %.4f\n', CC);
% 
% % 7. Plot Results
% figure;
% subplot(3,1,1); plot(t, clean_signal); title('Real PD - Clean'); xlabel('Time (s)'); ylabel('Amplitude');
% subplot(3,1,2); plot(t, noisy_signal); title('Real PD - Noisy'); xlabel('Time (s)'); ylabel('Amplitude');
% subplot(3,1,3); plot(t, denoised_signal); title('Real PD - CNN Denoised'); xlabel('Time (s)'); ylabel('Amplitude');

%% Conventional Methods
%% Simulated Signal 
% 1. Signal Simulation with Types A, B, C, D, E, F
fs = 1000e6;                  % 1 GHz sampling
t_total = 2e-6;               % 2 μs duration
t = 0:1/fs:t_total;           % Time vector
signal_length = length(t);
num_samples = 3000;           % 500 samples per type (6 types)

clean_signals = zeros(num_samples, signal_length);
noisy_signals = zeros(num_samples, signal_length);

for i = 1:num_samples
    % Cycle through Type A, Type B, Type C, Type D, Type E, Type F
    signal_type = mod(i-1, 6) + 1;  % Types 1,2,3,4,5,6 correspond to A,B,C,D,E,F

    if signal_type == 1  % Type A: Sparse PD pulses (4 pulses with gaps)
        clean_signal = zeros(size(t));
        start_times = [0.2e-6, 0.6e-6, 1.2e-6, 1.6e-6]; % Clear time gaps

        for k = 1:length(start_times)
            A = 10 + rand()*10;
            fc = 25e6 + rand()*10e6; % Higher freq helps sharpen
            tau = 0.01e-6 + rand()*0.03e-6; % Very short pulse
            pulse_t = t - start_times(k);
            pulse_t = pulse_t(pulse_t >= 0);

            % Generate a short pulse with fewer points
            pulse_duration = 0.05e-6; % 50 ns duration
            pulse_t = pulse_t(pulse_t <= pulse_duration);
            pulse = A * exp(-pulse_t/tau) .* sin(2*pi*fc*pulse_t);

            % Insert pulse at the correct position
            start_idx = find(t >= start_times(k), 1);
            pulse_len = length(pulse);
            if start_idx + pulse_len - 1 <= length(clean_signal)
                clean_signal(start_idx:start_idx + pulse_len - 1) = pulse;
            end
        end

    elseif signal_type == 2  % Type B: Spike-dense signal (realistic sharp pulses with gaps)
        clean_signal = zeros(size(t));
        num_spikes = 20 + randi(10); % Fewer, more realistic pulses
        spike_len = 2; % Length of each biphasic spike

        for s = 1:num_spikes
            start_idx = randi([1, signal_length - spike_len]);
            amp = 0.5 + 0.5*rand(); % Amplitude
            direction = (-1)^randi([0 1]); % Flip polarity randomly

            % Biphasic spike: [positive, negative] or [negative, positive]
            clean_signal(start_idx) = direction * amp;
            clean_signal(start_idx + 1) = -direction * amp;
        end

    elseif signal_type == 3  % Type C: 10mm
        clean_signal = zeros(size(t));

        % Very sparse PD events with random locations
        num_events = 5 + randi(5);  % 5-10 events (very sparse)

        % Generate completely random event times
        event_times = sort(rand(1, num_events) * t_total);

        for e = 1:length(event_times)
            event_time = event_times(e);
            start_idx = round(event_time * fs) + 1;

            if start_idx <= length(clean_signal)
                % Clear, distinct bipolar spikes
                amplitude = 2.5 + rand() * 2;  % 2.5-4.5 amplitude
                polarity = (-1)^randi([0 1]);  % Random polarity

                % Sharp bipolar pulse
                spike_width = 3 + randi(3);  % 3-6 samples wide

                if start_idx + spike_width - 1 <= length(clean_signal)
                    % Main spike
                    clean_signal(start_idx) = polarity * amplitude;
                    if start_idx + 1 <= length(clean_signal)
                        clean_signal(start_idx + 1) = -polarity * amplitude * 0.8;
                    end

                    % Decay tail
                    for j = 2:spike_width-1
                        if start_idx + j <= length(clean_signal)
                            clean_signal(start_idx + j) = polarity * amplitude * 0.3 * exp(-(j-1));
                        end
                    end
                end
            end
        end

    elseif signal_type == 4  % Type D: 18mm
        clean_signal = zeros(size(t));

        % Sparse PD events with random locations
        num_events = 55 + randi(10);  % 55-65 events (sparse)

        % Generate completely random event times
        event_times = sort(rand(1, num_events) * t_total);

        for e = 1:length(event_times)
            event_time = event_times(e);
            start_idx = round(event_time * fs) + 1;

            if start_idx <= length(clean_signal)
                % Mixed event types with random characteristics
                event_type = rand();
                amplitude = 2 + rand() * 3;  % 2-5 range
                polarity = (-1)^randi([0 1]);

                if event_type < 0.7  % 70% - Sharp bipolar spikes
                    spike_width = 2 + randi(4);

                    if start_idx + spike_width - 1 <= length(clean_signal)
                        clean_signal(start_idx) = polarity * amplitude;
                        if start_idx + 1 <= length(clean_signal)
                            clean_signal(start_idx + 1) = -polarity * amplitude * 0.7;
                        end

                        % Add some oscillatory tail
                        for j = 2:spike_width-1
                            if start_idx + j <= length(clean_signal)
                                clean_signal(start_idx + j) = polarity * amplitude * 0.2 * sin(j);
                            end
                        end
                    end
                end
            end
        end

    elseif signal_type == 5  % Type E: 20mm
        clean_signal = zeros(size(t));

        % Moderate to high frequency PD events with random locations
        num_events = 120 + randi(30);  % 120-150 events

        % Generate completely random event times
        event_times = sort(rand(1, num_events) * t_total);

        for e = 1:length(event_times)
            event_time = event_times(e);
            start_idx = round(event_time * fs) + 1;

            if start_idx <= length(clean_signal)
                amplitude = 2 + rand() * 4;  % 2-6 range
                polarity = (-1)^randi([0 1]);

                event_type = rand();

                if event_type < 0.6  % 60% - Sharp spikes
                    if start_idx + 1 <= length(clean_signal)
                        clean_signal(start_idx) = polarity * amplitude;
                        clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
                    end

                else  % 40% - Multi-frequency transients
                    % Multiple frequency components
                    fc1 = 30e6 + rand() * 50e6;
                    fc2 = 60e6 + rand() * 60e6;
                    fc3 = 100e6 + rand() * 50e6;  % Add third component

                    event_duration = 5e-9 + rand() * 15e-9;
                    event_samples = round(event_duration * fs);

                    if start_idx + event_samples <= length(clean_signal)
                        event_time_vec = (0:event_samples-1) / fs;
                        envelope = exp(-event_time_vec / (event_duration * 0.2));

                        component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
                        component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
                        component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);

                        complex_signal = polarity * (component1 + component2 + component3);
                        clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
                    end
                end
            end
        end

    else  % Type F: 25mm
        clean_signal = zeros(size(t));

        % 25mm events with random locations throughout
        num_events = 250 + randi(80);  % 250-330 events

        % Generate completely random event times
        event_times = sort(rand(1, num_events) * t_total);

        for e = 1:length(event_times)
            event_time = event_times(e);
            start_idx = round(event_time * fs) + 1;

            if start_idx <= length(clean_signal)
                amplitude = 3 + rand() * 4.2;  % 3-7.2 range
                polarity = (-1)^randi([0 1]);

                event_type = rand();

                if event_type < 0.4  % 40% - Quick spikes
                    if start_idx + 1 <= length(clean_signal)
                        clean_signal(start_idx) = polarity * amplitude;
                        clean_signal(start_idx + 1) = -polarity * amplitude * 0.2;
                    end

                else  % 60% - Complex multi-component events
                    % Multiple frequency components
                    fc1 = 30e6 + rand() * 50e6;
                    fc2 = 60e6 + rand() * 60e6;
                    fc3 = 100e6 + rand() * 50e6;  % Add third component

                    event_duration = 5e-9 + rand() * 15e-9;
                    event_samples = round(event_duration * fs);

                    if start_idx + event_samples <= length(clean_signal)
                        event_time_vec = (0:event_samples-1) / fs;
                        envelope = exp(-event_time_vec / (event_duration * 0.2));

                        component1 = 0.3 * amplitude * envelope .* sin(2*pi*fc1*event_time_vec);
                        component2 = 0.2 * amplitude * envelope .* sin(2*pi*fc2*event_time_vec);
                        component3 = 0.3 * amplitude * envelope .* sin(2*pi*fc3*event_time_vec);

                        complex_signal = polarity * (component1 + component2 + component3);
                        clean_signal(start_idx:start_idx + event_samples - 1) = complex_signal;
                    end
                end
            end
        end
    end

    % Keep signals in a reasonable range but preserve relative amplitudes
    max_amplitude = max(abs(clean_signal));
    if max_amplitude > 0
        if max_amplitude > 5
            clean_signal = clean_signal * (5 / max_amplitude);
        end
    end

    % Normalize clean signal
    clean_signal = clean_signal / (max(abs(clean_signal)) + eps);  % Avoid division by zero

    % Add noise manually (alternative to awgn function)
    desired_snr = -10;
    
    % Calculate signal power
    signal_power = mean(clean_signal.^2);
    
    % Calculate required noise power for desired SNR
    noise_power = signal_power / (10^(desired_snr/10));
    
    % Generate white Gaussian noise with calculated power
    noise = sqrt(noise_power) * randn(size(clean_signal));
    
    % Add noise to clean signal
    noisy_signal = clean_signal + noise;

    clean_signals(i,:) = clean_signal;
    noisy_signals(i,:) = noisy_signal;
end

%% 2. Select representative samples for each type
type_indices = [1, 2, 3, 4, 5, 6]; % One sample from each type

clean_signal_A = clean_signals(type_indices(1), :);
noisy_signal_A = noisy_signals(type_indices(1), :);
clean_signal_B = clean_signals(type_indices(2), :);
noisy_signal_B = noisy_signals(type_indices(2), :);
clean_signal_C = clean_signals(type_indices(3), :);
noisy_signal_C = noisy_signals(type_indices(3), :);
clean_signal_D = clean_signals(type_indices(4), :);
noisy_signal_D = noisy_signals(type_indices(4), :);
clean_signal_E = clean_signals(type_indices(5), :);
noisy_signal_E = noisy_signals(type_indices(5), :);
clean_signal_F = clean_signals(type_indices(6), :);
noisy_signal_F = noisy_signals(type_indices(6), :);

%% 3. Apply Denoising

% denoised_signal_A = wdenoise(noisy_signal_A, 'Wavelet', 'db2', ...
%     'DenoisingMethod', 'Minimax', 'ThresholdRule', 'Soft');
% denoised_signal_B = wdenoise(noisy_signal_B, 'Wavelet', 'db2', ...
%     'DenoisingMethod', 'Minimax', 'ThresholdRule', 'Soft');
% denoised_signal_C = wdenoise(noisy_signal_C, 'Wavelet', 'db2', ...
%     'DenoisingMethod', 'Minimax', 'ThresholdRule', 'Soft');
% denoised_signal_D = wdenoise(noisy_signal_D, 'Wavelet', 'db2', ...
%     'DenoisingMethod', 'Minimax', 'ThresholdRule', 'Soft');
% denoised_signal_E = wdenoise(noisy_signal_E, 'Wavelet', 'db2', ...
%     'DenoisingMethod', 'Minimax', 'ThresholdRule', 'Soft');
% denoised_signal_F = wdenoise(noisy_signal_F, 'Wavelet', 'db2', ...
%     'DenoisingMethod', 'Minimax', 'ThresholdRule', 'Soft');

% % Using wavelet packet denoising (assuming you have this function)
% denoised_signal_A = wavelet_packet_denoising(noisy_signal_A, 'db2', 4);
% denoised_signal_B = wavelet_packet_denoising(noisy_signal_B, 'db2', 4);
% denoised_signal_C = wavelet_packet_denoising(noisy_signal_C, 'db2', 4);
% denoised_signal_D = wavelet_packet_denoising(noisy_signal_D, 'db2', 4);
% denoised_signal_E = wavelet_packet_denoising(noisy_signal_E, 'db2', 4);
% denoised_signal_F = wavelet_packet_denoising(noisy_signal_F, 'db2', 4);

% denoised_signal_A = pod_denoising(noisy_signal_A, 0.2);
% denoised_signal_B = pod_denoising(noisy_signal_B, 0.2);
% denoised_signal_C = pod_denoising(noisy_signal_C, 0.2);
% denoised_signal_D = pod_denoising(noisy_signal_D, 0.2);
% denoised_signal_E = pod_denoising(noisy_signal_E, 0.2);
% denoised_signal_F = pod_denoising(noisy_signal_F, 0.2);

% denoised_signal_A = pca_wavelet_denoising(noisy_signal_A, 'db2', 4, 5);
% denoised_signal_B = pca_wavelet_denoising(noisy_signal_B, 'db2', 4, 5);
% denoised_signal_C = pca_wavelet_denoising(noisy_signal_C, 'db2', 4, 5);
% denoised_signal_D = pca_wavelet_denoising(noisy_signal_D, 'db2', 4, 5);
% denoised_signal_E = pca_wavelet_denoising(noisy_signal_E, 'db2', 4, 5);
% denoised_signal_F = pca_wavelet_denoising(noisy_signal_F, 'db2', 4, 5);

% Savitzky-Golay + Wavelet
denoised_signal_A = savgol_wavelet_denoising(noisy_signal_A);
denoised_signal_B = savgol_wavelet_denoising(noisy_signal_B);
denoised_signal_C = savgol_wavelet_denoising(noisy_signal_C);
denoised_signal_D = savgol_wavelet_denoising(noisy_signal_D);
denoised_signal_E = savgol_wavelet_denoising(noisy_signal_E);
denoised_signal_F = savgol_wavelet_denoising(noisy_signal_F);


%% 4. Ensure Lengths Match
L = length(clean_signal_A);
denoised_signal_A = denoised_signal_A(1:L);
denoised_signal_B = denoised_signal_B(1:L);
denoised_signal_C = denoised_signal_C(1:L);
denoised_signal_D = denoised_signal_D(1:L);
denoised_signal_E = denoised_signal_E(1:L);
denoised_signal_F = denoised_signal_F(1:L);

%% 5. Evaluation Metrics for all types

% Function to calculate metrics
calculate_metrics = @(clean, noisy, denoised) struct(...
    'SNR_before', 10 * log10(sum(clean.^2) / sum((noisy - clean).^2)), ...
    'SNR_after', 10 * log10(sum(clean.^2) / sum((denoised - clean).^2)), ...
    'MSE', mean((clean - denoised).^2), ...
    'CC', sum(clean .* denoised) / sqrt(sum(clean.^2) * sum(denoised.^2)));

% Calculate metrics for each type
metrics_A = calculate_metrics(clean_signal_A, noisy_signal_A, denoised_signal_A);
metrics_B = calculate_metrics(clean_signal_B, noisy_signal_B, denoised_signal_B);
metrics_C = calculate_metrics(clean_signal_C, noisy_signal_C, denoised_signal_C);
metrics_D = calculate_metrics(clean_signal_D, noisy_signal_D, denoised_signal_D);
metrics_E = calculate_metrics(clean_signal_E, noisy_signal_E, denoised_signal_E);
metrics_F = calculate_metrics(clean_signal_F, noisy_signal_F, denoised_signal_F);

% %% 6. Plot Results for all types in a single 6x3 figure
type_names = {'Type A', 'Type B', ...
              'Type C', 'Type D', 'Type E', 'Type F'};

clean_signals_all = {clean_signal_A, clean_signal_B, clean_signal_C, ...
                     clean_signal_D, clean_signal_E, clean_signal_F};
noisy_signals_all = {noisy_signal_A, noisy_signal_B, noisy_signal_C, ...
                     noisy_signal_D, noisy_signal_E, noisy_signal_F};
denoised_signals_all = {denoised_signal_A, denoised_signal_B, denoised_signal_C, ...
                        denoised_signal_D, denoised_signal_E, denoised_signal_F};

% Create a single large figure with 6x3 subplots
figure('Position', [100, 100, 1200, 1400]);

for type_idx = 1:6
    % Clean signals (column 1)
    subplot(6, 3, (type_idx-1)*3 + 1);
    plot(t, clean_signals_all{type_idx}, 'LineWidth', 1);
    title([type_names{type_idx} ' - Clean'], 'FontSize', 10);
    xlabel('Time (s)', 'FontSize', 8);
    ylabel('Amplitude', 'FontSize', 8);
    grid on;

    % Noisy signals (column 2)
    subplot(6, 3, (type_idx-1)*3 + 2);
    plot(t, noisy_signals_all{type_idx}, 'LineWidth', 1);
    title([type_names{type_idx} ' - Noisy'], 'FontSize', 10);
    xlabel('Time (s)', 'FontSize', 8);
    ylabel('Amplitude', 'FontSize', 8);
    grid on;

    % Denoised signals (column 3)
    subplot(6, 3, (type_idx-1)*3 + 3);
    plot(t, denoised_signals_all{type_idx}, 'LineWidth', 1);
    title([type_names{type_idx} ' - Denoised'], 'FontSize', 10);
    xlabel('Time (s)', 'FontSize', 8);
    ylabel('Amplitude', 'FontSize', 8);
    grid on;
end

% Add main title for the entire figure
sgtitle('Savitzky-Golay + Wavelet', 'FontSize', 14, 'FontWeight', 'bold');

%% 7. Print Performance for all types
all_metrics = {metrics_A, metrics_B, metrics_C, metrics_D, metrics_E, metrics_F};

for type_idx = 1:6
    fprintf('=== Performance Metrics for %s ===\n', type_names{type_idx});
    fprintf('SNR before denoising: %.2f dB\n', all_metrics{type_idx}.SNR_before);
    fprintf('SNR after denoising : %.2f dB\n', all_metrics{type_idx}.SNR_after);
    fprintf('Mean Square Error (MSE): %.6f\n', all_metrics{type_idx}.MSE);
    fprintf('Correlation Coefficient (CC): %.4f\n\n', all_metrics{type_idx}.CC);
end

%% 8. Summary Statistics across all types
fprintf('=== Summary ===\n');
SNR_improvements = zeros(1, 6);
for i = 1:6
    SNR_improvements(i) = all_metrics{i}.SNR_after - all_metrics{i}.SNR_before;
end

fprintf('Average SNR Improvement: %.2f dB\n', mean(SNR_improvements));

% Calculate average MSE
MSEs = [all_metrics{1}.MSE, all_metrics{2}.MSE, all_metrics{3}.MSE, ...
       all_metrics{4}.MSE, all_metrics{5}.MSE, all_metrics{6}.MSE];
fprintf('Average Mean Square Error: %.4f\n', mean(MSEs));

% Calculate average correlation coefficients
CCs = [all_metrics{1}.CC, all_metrics{2}.CC, all_metrics{3}.CC, ...
       all_metrics{4}.CC, all_metrics{5}.CC, all_metrics{6}.CC];
fprintf('Average Correlation Coefficient: %.4f\n', mean(CCs));


%% Savitzky-Golay + Wavelet Denoising 
function denoised_signal = savgol_wavelet_denoising(signal)
    try
        signal = signal(:)';

        % Savitzky-Golay Parameters
        window_length = max(7, floor(length(signal)/50));  % window
        if mod(window_length, 2) == 0
            window_length = window_length + 1;
        end
        poly_order = min(3, window_length - 2);  % polynomial order
        
        % Minimum requirements for sgolayfilt
        if length(signal) >= window_length && window_length >= 5
            smooth_signal = sgolayfilt(signal, min(3, window_length-2), window_length);
            residual = signal - smooth_signal;
            
            % Second stage: Wavelet denoising on residual
            if length(residual) > 4

                denoised_residual = wdenoise(residual, 'Wavelet', 'db2');
            else
                denoised_residual = residual;
            end
                denoised_signal = smooth_signal * denoised_residual;
         
        else
            % Fallback to simple wavelet denoising
            denoised_signal = wdenoise(signal, 'Wavelet', 'db2');
        end
    catch
        % Ultimate fallback
        try
            denoised_signal = wdenoise(signal, 'Wavelet', 'db2');
        catch
            denoised_signal = signal;
        end
    end
end

%% PCA _Wavelet
function denoised_signal = pca_wavelet_denoising(signal, default_wavelet, max_level, default_num_pcs)
    signal = signal(:);  % Ensure column vector
    L_sig = length(signal);

    %% Step 0: Auto-determine safe max level
    safe_max_level = wmaxlev(L_sig, default_wavelet);
    max_level = min(max_level, safe_max_level);  % Use smaller of user vs safe

    %% Step 1: Best Wavelet Selection based on Shannon Entropy
    candidate_wavelets = {'db4', 'sym4', 'coif5', 'bior4.4'};
    min_entropy = inf;
    best_wavelet = default_wavelet;
    for i = 1:length(candidate_wavelets)
        try
            [C_test, L_test] = wavedec(signal, max_level, candidate_wavelets{i});
            ent = wentropy(C_test, 'shannon');
            if ent < min_entropy
                min_entropy = ent;
                best_wavelet = candidate_wavelets{i};
            end
        catch
            continue;
        end
    end

    %% Step 2: Wavelet Decomposition
    [C, L_dec] = wavedec(signal, max_level, best_wavelet);

    %% Step 3: Extract and denoise detail coefficients using proper indexing
    detail_coeffs = cell(1, max_level);

    % Calculate starting positions for each level
    cum_lengths = cumsum(L_dec);

    for lev = 1:max_level
        % Get the correct indices for this detail level
        if lev == 1
            % First detail level (highest frequency)
            start_idx = cum_lengths(1) + 1;
            end_idx = cum_lengths(1 + lev);
        else
            % Subsequent detail levels
            start_idx = cum_lengths(lev) + 1;
            end_idx = cum_lengths(lev + 1);
        end

        % Safety check
        if start_idx > length(C) || end_idx > length(C) || start_idx < 1
            warning('Skipping level %d due to indexing issue', lev);
            detail_coeffs{lev} = [];
            continue;
        end

        coeff = C(start_idx:end_idx);

        % Adaptive soft thresholding
        if ~isempty(coeff)
            sigma = median(abs(coeff)) / 0.6745;
            T = sigma * sqrt(2 * log(length(coeff)));
            coeff_denoised = wthresh(coeff, 's', T);
            detail_coeffs{lev} = coeff_denoised;
        else
            detail_coeffs{lev} = [];
        end
    end

    %% Step 4: Approximation Coefficient (first L_dec(1) elements)
    approx_len = L_dec(1);
    approx_coeff = C(1:approx_len);

    %% Step 5: Reconstruct signals for each level
    coeff_matrix = [];

    for lev = 1:max_level
        if ~isempty(detail_coeffs{lev})
            % Create coefficient vector with only this level's details
            new_C = zeros(size(C));

            % Insert the denoised detail coefficients at correct position
            if lev == 1
                start_idx = cum_lengths(1) + 1;
                end_idx = cum_lengths(1 + lev);
            else
                start_idx = cum_lengths(lev) + 1;
                end_idx = cum_lengths(lev + 1);
            end

            if start_idx <= length(new_C) && end_idx <= length(new_C)
                new_C(start_idx:end_idx) = detail_coeffs{lev};
                temp_signal = waverec(new_C, L_dec, best_wavelet);
                coeff_matrix = [coeff_matrix; temp_signal(:)'];
            end
        end
    end

    % Add approximation component
    new_C = zeros(size(C));
    new_C(1:approx_len) = approx_coeff;
    approx_signal = waverec(new_C, L_dec, best_wavelet);
    coeff_matrix = [coeff_matrix; approx_signal(:)'];

    %% Step 6-9: PCA and reconstruction (if we have valid components)
    if ~isempty(coeff_matrix) && size(coeff_matrix, 1) > 1
        try
            [coeff, score, latent] = pca(coeff_matrix');

            % Energy-Based Component Selection
            energy = cumsum(latent) / sum(latent);
            num_pcs = find(energy >= 0.95, 1);
            if isempty(num_pcs) || num_pcs > default_num_pcs
                num_pcs = min(default_num_pcs, size(score, 2));
            end

            % K-means Clustering (only if we have enough components)
            if num_pcs >= 2 && size(score, 1) > 2
                try
                    [cluster_idx, ~] = kmeans(score(:, 1:num_pcs), 2, 'Replicates', 5);
                    var1 = var(score(cluster_idx == 1, 1));
                    var2 = var(score(cluster_idx == 2, 1));
                    main_cluster = 1;
                    if var2 > var1
                        main_cluster = 2;
                    end
                    score_filtered = score;
                    score_filtered(cluster_idx ~= main_cluster, :) = 0;
                catch
                    % If k-means fails, use all components
                    score_filtered = score;
                end
            else
                score_filtered = score;
            end

            % Signal Reconstruction
            denoised_components = (score_filtered * coeff')';
            denoised_signal = mean(denoised_components, 1);
            denoised_signal = denoised_signal(1:L_sig);
        catch
            % If PCA fails, fall back to simple wavelet denoising
            warning('PCA failed, using simple wavelet denoising');
            denoised_signal = wdenoise(signal, 'Wavelet', best_wavelet);
            denoised_signal = denoised_signal(1:L_sig);
        end
    else
        % If no valid components, fall back to simple denoising
        warning('No valid wavelet components, using simple denoising');
        denoised_signal = wdenoise(signal, 'Wavelet', best_wavelet);
        denoised_signal = denoised_signal(1:L_sig);
    end

    % Ensure output is row vector to match input format
    denoised_signal = denoised_signal(:)';
end


%% WPD
% function denoised_signal = wavelet_packet_denoising(noisy_signal, wavelet_type, decomp_level)
%     % Perform wavelet packet decomposition
%     wp = wpdec(noisy_signal, decomp_level, wavelet_type);
%     % Estimate noise level
%     noise_level = estimate_noise(noisy_signal);
%     % Adaptive thresholding
%     threshold = 0.7 * noise_level * sqrt(2 * log(length(noisy_signal)));
% 
%     % Apply thresholding to all terminal nodes (leaves) of the wavelet packet tree
%     terminal_nodes = leaves(wp);  % Get the indices of terminal nodes
%     for i = 1:length(terminal_nodes)
%         node_index = terminal_nodes(i);
%         coefs = wpcoef(wp, node_index);            % Get coefficients of the current node
%         coefs = wthresh(coefs, 's', threshold);    % Soft thresholding
%         wp = write(wp, 'data', node_index, coefs); % Update the tree with thresholded coefficients
%     end
% 
%     % Reconstruct signal from the denoised wavelet packet
%     denoised_signal = wprec(wp);
% end
% 
% function noise_std = estimate_noise(signal)
%     % Robust noise standard deviation estimation
%     % Uses Median Absolute Deviation (MAD) method
%     median_abs_dev = median(abs(signal - median(signal)));
%     noise_std = median_abs_dev / 0.6745;
% end

%% POD Denoising 
% function denoised_signal = pod_denoising(signal, energy_threshold)
%     % Improved POD denoising with adaptive parameters and better reconstruction
% 
%     % Adaptive window sizing based on signal characteristics
%     signal_energy = sum(signal.^2);
%     signal_sparsity = sum(abs(signal) > 0.1*max(abs(signal))) / length(signal);
% 
%     % Adjust window size based on signal sparsity and energy
%     if signal_sparsity < 0.05  % Very sparse signals (Type A, C)
%         window_size = 200;
%         overlap = 150;
%     elseif signal_sparsity < 0.15  % Moderately sparse (Type D)
%         window_size = 150;
%         overlap = 100;
%     else  % Dense signals (Type E, F, B)
%         window_size = 100;
%         overlap = 75;
%     end
% 
%     % Ensure minimum window size
%     window_size = max(window_size, 50);
%     overlap = min(overlap, window_size - 10);
% 
%     step = window_size - overlap;
%     num_segments = floor((length(signal) - overlap) / step);
% 
%     % Handle edge case where signal is too short
%     if num_segments < 1
%         window_size = min(length(signal), 50);
%         overlap = 0;
%         step = window_size;
%         num_segments = 1;
%     end
% 
%     % Create segments with proper bounds checking
%     segments = zeros(window_size, num_segments);
%     for i = 1:num_segments
%         start_idx = (i-1)*step + 1;
%         end_idx = min(start_idx + window_size - 1, length(signal));
%         actual_length = end_idx - start_idx + 1;
% 
%         if actual_length == window_size
%             segments(:, i) = signal(start_idx:end_idx);
%         else
%             % Pad shorter segments
%             temp_segment = zeros(window_size, 1);
%             temp_segment(1:actual_length) = signal(start_idx:end_idx);
%             segments(:, i) = temp_segment;
%         end
%     end
% 
%     % Apply POD with improved singular value selection
%     [U, S, V] = svd(segments, 'econ');
%     singular_values = diag(S);
% 
%     % Multiple criteria for mode selection
%     total_energy = sum(singular_values.^2);
%     cumulative_energy = cumsum(singular_values.^2) / total_energy;
% 
%     % Method 1: Energy threshold
%     nModes_energy = find(cumulative_energy >= energy_threshold, 1);
%     if isempty(nModes_energy)
%         nModes_energy = length(singular_values);
%     end
% 
%     % Method 2: Significant drop in singular values (elbow method)
%     if length(singular_values) > 2
%         sv_ratios = singular_values(1:end-1) ./ singular_values(2:end);
%         [~, elbow_idx] = max(sv_ratios);
%         nModes_elbow = elbow_idx;
%     else
%         nModes_elbow = length(singular_values);
%     end
% 
%     % Method 3: Noise floor estimation
%     % Estimate noise level from smallest singular values
%     if length(singular_values) > 10
%         noise_floor = median(singular_values(end-4:end));
%         nModes_noise = sum(singular_values > 3 * noise_floor);
%     else
%         nModes_noise = max(1, floor(0.7 * length(singular_values)));
%     end
% 
%     % Choose the most conservative estimate (but at least 1 mode)
%     nModes = max(1, min([nModes_energy, nModes_elbow, nModes_noise]));
% 
%     % Soft thresholding instead of hard cutoff for smoother reconstruction
%     S_denoised = S;
%     threshold_value = singular_values(min(nModes + 1, length(singular_values)));
% 
%     for i = 1:size(S, 1)
%         if i > nModes && S(i,i) > 0
%             % Apply soft thresholding to remaining modes
%             soft_threshold = max(0, S(i,i) - threshold_value);
%             S_denoised(i,i) = soft_threshold;
%         end
%     end
% 
%     % Reconstruct segments
%     segments_denoised = U * S_denoised * V';
% 
%     % Improved reconstruction with proper weighting
%     data_denoised = zeros(1, length(signal));
%     weight = zeros(1, length(signal));
% 
%     % Tapered window for better overlap handling
%     taper_length = min(10, overlap);
%     taper_window = ones(window_size, 1);
%     if taper_length > 0
%         % Cosine taper at the edges
%         taper_start = 0.5 * (1 + cos(pi * (0:taper_length-1) / taper_length));
%         taper_end = 0.5 * (1 + cos(pi * (taper_length-1:-1:0) / taper_length));
%         taper_window(1:taper_length) = taper_start;
%         taper_window(end-taper_length+1:end) = taper_end;
%     end
% 
%     for i = 1:num_segments
%         start_idx = (i-1)*step + 1;
%         end_idx = min(start_idx + window_size - 1, length(signal));
%         actual_length = end_idx - start_idx + 1;
% 
%         if actual_length == window_size
%             segment_data = segments_denoised(:, i) .* taper_window;
%             data_denoised(start_idx:end_idx) = data_denoised(start_idx:end_idx) + segment_data';
%             weight(start_idx:end_idx) = weight(start_idx:end_idx) + taper_window';
%         else
%             % Handle shorter segments
%             segment_data = segments_denoised(1:actual_length, i) .* taper_window(1:actual_length);
%             data_denoised(start_idx:end_idx) = data_denoised(start_idx:end_idx) + segment_data';
%             weight(start_idx:end_idx) = weight(start_idx:end_idx) + taper_window(1:actual_length)';
%         end
%     end
% 
%     % Normalize by weights
%     denoised_signal = data_denoised ./ (weight + eps);
% 
%     % Post-processing: preserve signal characteristics
%     % Scale to maintain similar amplitude range as input
%     input_max = max(abs(signal));
%     output_max = max(abs(denoised_signal));
%     if output_max > 0 && input_max > 0
%         scale_factor = min(1.2, input_max / output_max);  % Allow slight amplification but limit it
%         denoised_signal = denoised_signal * scale_factor;
%     end
% 
%     % Optional: Apply mild smoothing to very noisy reconstructions
%     signal_to_noise_est = var(signal) / (var(signal - denoised_signal) + eps);
%     if signal_to_noise_est < 2  % If reconstruction seems very noisy
%         % Apply very light smoothing
%         smooth_kernel = [0.25, 0.5, 0.25];
%         denoised_signal = conv(denoised_signal, smooth_kernel, 'same');
%     end
% 
%     % Ensure output is returned as row vector to match input format
%     if size(signal, 1) == 1  % Input was row vector
%         denoised_signal = denoised_signal(:)';
%     else  % Input was column vector
%         denoised_signal = denoised_signal(:);
%     end
% end

% % Sampling and Time Parameters
% fs = 1000e6;        % Sampling frequency (1 GHz)
% t_total = 2e-6;     % Total time in seconds (2 μs)
% 
% % Calculate number of samples (reduced for memory)
% num_samples = round(t_total * fs) + 1;
% disp(['Creating signal with ', num2str(num_samples), ' samples']);
% 
% % Use a more memory-efficient approach for signal generation
% % Instead of creating a full time vector, we'll generate the signal directly
% 
% % Initialize signal
% clean_signal = zeros(num_samples, 1);
% 
% % PD Signal Parameters
% params = {
%     struct('A', 10, 'fc', 20e6, 'tau', 0.1e-6, 'formula', 'D1', 'start_time', 0.2e-6),
%     struct('A', 20, 'fc', 20e6, 'tau', 0.15e-6, 'formula', 'D2', 'start_time', 0.6e-6),
%     struct('A', 10, 'fc', 40e6, 'tau', 0.1e-6, 'formula', 'D1', 'start_time', 1.2e-6),
%     struct('A', 20, 'fc', 40e6, 'tau', 0.15e-6, 'formula', 'D2', 'start_time', 1.6e-6)
% };
% 
% % Generate each pulse separately and add to the signal
% for i = 1:length(params)
%     param = params{i};
% 
%     % Calculate sample indices for this pulse
%     start_sample = round(param.start_time * fs) + 1;
%     end_sample = min(num_samples, start_sample + round(5 * param.tau * fs));
% 
%     % Generate time vector only for this pulse
%     pulse_length = end_sample - start_sample + 1;
%     t_pulse = (0:(pulse_length-1)) / fs;
% 
%     % Generate pulse
%     if strcmp(param.formula, 'D1')
%         pulse = param.A * exp(-t_pulse / param.tau) .* sin(2 * pi * param.fc * t_pulse);
%     else % D2
%         pulse = param.A * (exp(-1.3 * t_pulse / param.tau) - exp(-2.2 * t_pulse / param.tau)) .* sin(2 * pi * param.fc * t_pulse);
%     end
% 
%     % Add pulse to signal
%     clean_signal(start_sample:end_sample) = clean_signal(start_sample:end_sample) + pulse';
% 
%     % Clear temporary variables
%     clear t_pulse pulse;
% end
% % % Add Noise
% desired_snr = -5; % adjust SNR
% noisy_signal = awgn(clean_signal, desired_snr, 'measured');

% 
% % Save memory by using single precision
% clean_signal = single(clean_signal);
% noisy_signal = single(noisy_signal);
% 
% %% Apply ICEEMDAN-MSE-DWT denoising
% disp('Starting denoising process...');
% tic;
% denoised_signal = ICEEMDAN_MSE_DWT_Denoising_HF(noisy_signal, fs);
% processing_time = toc;
% fprintf('Denoising completed in %.2f seconds\n', processing_time);
% 
% %% Evaluate Performance
% % Convert back to double for calculations
% clean_signal = double(clean_signal);
% noisy_signal = double(noisy_signal);
% denoised_signal = double(denoised_signal);
% 
% % Calculate SNR
% clean_power = sum(clean_signal.^2) / length(clean_signal);
% noise_power = sum((noisy_signal - clean_signal).^2) / length(clean_signal);
% denoised_noise_power = sum((denoised_signal - clean_signal).^2) / length(clean_signal);
% 
% original_snr = 10*log10(clean_power / noise_power);
% denoised_snr = 10*log10(clean_power / denoised_noise_power);
% snr_improvement = denoised_snr - original_snr;
% 
% % Calculate MSE and RMSE
% mse_noisy = mean((clean_signal - noisy_signal).^2);
% mse_denoised = mean((clean_signal - denoised_signal).^2);
% mse_improvement = 100 * (1 - mse_denoised/mse_noisy);
% 
% rmse_noisy = sqrt(mse_noisy);
% rmse_denoised = sqrt(mse_denoised);
% 
% % Calculate correlation coefficients
% % Use batch processing for large signals
% batch_size = min(100000, floor(num_samples/10));
% num_batches = ceil(length(clean_signal) / batch_size);
% 
% corr_noisy_sum = 0;
% corr_denoised_sum = 0;
% clean_mean = mean(clean_signal);
% noisy_mean = mean(noisy_signal);
% denoised_mean = mean(denoised_signal);
% 
% clean_var = 0;
% noisy_var = 0;
% denoised_var = 0;
% clean_noisy_cov = 0;
% clean_denoised_cov = 0;
% 
% for batch = 1:num_batches
%     start_idx = (batch-1) * batch_size + 1;
%     end_idx = min(batch * batch_size, length(clean_signal));
% 
%     clean_batch = clean_signal(start_idx:end_idx);
%     noisy_batch = noisy_signal(start_idx:end_idx);
%     denoised_batch = denoised_signal(start_idx:end_idx);
% 
%     clean_var = clean_var + sum((clean_batch - clean_mean).^2);
%     noisy_var = noisy_var + sum((noisy_batch - noisy_mean).^2);
%     denoised_var = denoised_var + sum((denoised_batch - denoised_mean).^2);
% 
%     clean_noisy_cov = clean_noisy_cov + sum((clean_batch - clean_mean).*(noisy_batch - noisy_mean));
%     clean_denoised_cov = clean_denoised_cov + sum((clean_batch - clean_mean).*(denoised_batch - denoised_mean));
% 
%     % Clear temporary variables
%     clear clean_batch noisy_batch denoised_batch;
% end
% 
% corr_noisy = clean_noisy_cov / sqrt(clean_var * noisy_var);
% corr_denoised = clean_denoised_cov / sqrt(clean_var * denoised_var);
% 
% % Print results
% fprintf('\nPerformance Metrics:\n');
% fprintf('Original SNR: %.2f dB\n', original_snr);
% fprintf('Denoised SNR: %.2f dB\n', denoised_snr);
% fprintf('SNR Improvement: %.2f dB\n', snr_improvement);
% fprintf('MSE Improvement: %.2f%%\n', mse_improvement);
% fprintf('RMSE (Noisy): %.4f\n', rmse_noisy);
% fprintf('RMSE (Denoised): %.4f\n', rmse_denoised);
% fprintf('Correlation (Noisy): %.4f\n', corr_noisy);
% fprintf('Correlation (Denoised): %.4f\n', corr_denoised);
% 
% %% Visualize Results (using downsampled signals for plotting)
% % Create downsampled signals for visualization
% ds_factor_viz = max(1, floor(length(clean_signal)/10000));
% t_viz = (0:ds_factor_viz:length(clean_signal)-1)' / fs * 1e6; % Convert to μs
% clean_viz = clean_signal(1:ds_factor_viz:end);
% noisy_viz = noisy_signal(1:ds_factor_viz:end);
% denoised_viz = denoised_signal(1:ds_factor_viz:end);
% 
% figure;
% subplot(3,1,1);
% plot(t_viz, clean_viz);
% title('Original Clean PD Signal');
% ylabel('Amplitude');
% grid on;
% 
% subplot(3,1,2);
% plot(t_viz, noisy_viz);
% title(['Noisy Signal (SNR = ', num2str(desired_snr), ' dB)']);
% ylabel('Amplitude');
% grid on;
% 
% subplot(3,1,3);
% plot(t_viz, denoised_viz);
% title(['Denoised Signal (SNR = ', num2str(denoised_snr, '%.2f'), ' dB)']);
% xlabel('Time (μs)');
% ylabel('Amplitude');
% grid on;
% 
% % Clear visualization variables to save memory
% clear clean_viz noisy_viz denoised_viz t_viz;
% 
% % Plot frequency domain comparison (using fewer points)
% figure;
% max_freq_points = 10000; % Maximum number of frequency points to plot
% 
% % Calculate FFT with downsampling for visualization
% N_fft = min(length(clean_signal), 2^nextpow2(length(clean_signal)/10));
% f = (0:(N_fft/2))*fs/N_fft / 1e6; % Convert to MHz
% 
% % Downsample frequency vector for plotting
% ds_factor_f = max(1, ceil(length(f)/max_freq_points));
% f_plot = f(1:ds_factor_f:end);
% 
% % Calculate FFTs
% F_clean = abs(fft(clean_signal, N_fft));
% F_clean = F_clean(1:N_fft/2+1);
% F_clean_plot = F_clean(1:ds_factor_f:end);
% 
% F_noisy = abs(fft(noisy_signal, N_fft));
% F_noisy = F_noisy(1:N_fft/2+1);
% F_noisy_plot = F_noisy(1:ds_factor_f:end);
% 
% F_denoised = abs(fft(denoised_signal, N_fft));
% F_denoised = F_denoised(1:N_fft/2+1);
% F_denoised_plot = F_denoised(1:ds_factor_f:end);
% 
% % Plot
% subplot(3,1,1);
% semilogy(f_plot, F_clean_plot);
% title('Original Clean Signal - Frequency Domain');
% ylabel('Magnitude');
% grid on;
% 
% subplot(3,1,2);
% semilogy(f_plot, F_noisy_plot);
% title('Noisy Signal - Frequency Domain');
% ylabel('Magnitude');
% grid on;
% 
% subplot(3,1,3);
% semilogy(f_plot, F_denoised_plot);
% title('Denoised Signal - Frequency Domain');
% xlabel('Frequency (MHz)');
% ylabel('Magnitude');
% grid on;
% 
% % Clear FFT variables to save memory
% clear F_clean F_noisy F_denoised F_clean_plot F_noisy_plot F_denoised_plot f f_plot;
% 
% % Create zoomed view of specific pulses
% figure;
% % Find pulse locations
% pulse_times = [0.2e-6, 0.6e-6, 1.2e-6, 1.6e-6]; % From parameters
% window_width = 0.4e-6; % Window width in seconds
% 
% for p = 1:length(pulse_times)
%     % Calculate sample indices
%     center_idx = round(pulse_times(p) * fs);
%     half_width = round(window_width/2 * fs);
%     start_idx = max(1, center_idx - half_width);
%     end_idx = min(length(clean_signal), center_idx + half_width);
% 
%     % Create time vector for this window
%     t_window = ((start_idx:end_idx) - 1) / fs * 1e6; % Convert to μs
% 
%     % Extract signal segments
%     clean_segment = clean_signal(start_idx:end_idx);
%     noisy_segment = noisy_signal(start_idx:end_idx);
%     denoised_segment = denoised_signal(start_idx:end_idx);
% 
%     % Plot
%     subplot(2,2,p);
%     plot(t_window, clean_segment, 'b-', 'LineWidth', 1.5);
%     hold on;
%     plot(t_window, noisy_segment, 'r-', 'LineWidth', 0.5);
%     plot(t_window, denoised_segment, 'g-', 'LineWidth', 1);
%     hold off;
% 
%     title(['Pulse at ', num2str(pulse_times(p)*1e6), ' μs']);
%     xlabel('Time (μs)');
%     ylabel('Amplitude');
%     legend('Clean', 'Noisy', 'Denoised');
%     grid on;
% 
%     % Clear temporary variables
%     clear t_window clean_segment noisy_segment denoised_segment;
% end
% 
% % Display memory usage
% memory_info = whos;
% total_memory_MB = sum([memory_info.bytes]) / 1024^2;
% fprintf('\nTotal memory usage: %.2f MB\n', total_memory_MB);
% 
% % Clear large variables at the end to free memory
% clear clean_signal noisy_signal denoised_signal;
% % clc;clear;
% % %% Simulated Signal
% % 
% % % Sampling and Time Parameters
% % fs = 1000e6; % Sampling frequency (1 GHz for high resolution)
% % t_total = 2e-6; % Total time in seconds for the full signal
% % t = 0:1/fs:1; % Time vector
% % 
% % % PD Signal Parameters
% % params = {
% %     % Pulse A parameters
% %     struct('A', 10, 'fc', 20e6, 'tau', 0.1e-6, 'formula', 'D1', 'start_time', 0.2e-6);
% %     % Pulse B parameters
% %     struct('A', 20, 'fc', 20e6, 'tau', 0.15e-6, 'formula', 'D2', 'start_time', 0.6e-6);
% %     % Pulse C parameters
% %     struct('A', 10, 'fc', 40e6, 'tau', 0.1e-6, 'formula', 'D1', 'start_time', 1.2e-6);
% %     % Pulse D parameters
% %     struct('A', 20, 'fc', 40e6, 'tau', 0.15e-6, 'formula', 'D2', 'start_time', 1.6e-6);
% % };
% % 
% % % Generate Clean Signal
% % clean_signal = zeros(size(t));
% % for i = 1:length(params)
% %     param = params{i};
% %     pulse_t = t - param.start_time;
% %     pulse_t = pulse_t(pulse_t >= 0); % Shift and keep only positive times
% %     if strcmp(param.formula, 'D1')
% %         pulse = param.A * exp(-pulse_t / param.tau) .* sin(2 * pi * param.fc * pulse_t);
% %     elseif strcmp(param.formula, 'D2')
% %         pulse = param.A * (exp(-1.3 * pulse_t / param.tau) - exp(-2.2 * pulse_t / param.tau)) .* sin(2 * pi * param.fc * pulse_t);
% %     end
% %     clean_signal(t >= param.start_time & t < param.start_time + length(pulse)/fs) = pulse;
% % end
% % 
% % 
% % % Add Noise
% % desired_snr= -5; % adjust SNR
% % noisy_signal = awgn(clean_signal, desired_snr, 'measured');
% % 
% % %% 2. Apply ICEEMDAN-MSE-DWT denoising
% % tic;
% % denoised_signal = ICEEMDAN_MSE_DWT_Denoising_HF(noisy_signal, fs);
% % processing_time = toc;
% % fprintf('Denoising completed in %.2f seconds\n', processing_time);
% % 
% % %% 3. Evaluate Performance
% % % Calculate SNR
% % clean_power = sum(clean_signal.^2) / length(clean_signal);
% % noise_power = sum((noisy_signal - clean_signal).^2) / length(clean_signal);
% % denoised_noise_power = sum((denoised_signal - clean_signal).^2) / length(clean_signal);
% % 
% % original_snr = 10*log10(clean_power / noise_power);
% % denoised_snr = 10*log10(clean_power / denoised_noise_power);
% % snr_improvement = denoised_snr - original_snr;
% % 
% % % Calculate MSE and RMSE
% % mse_noisy = mean((clean_signal - noisy_signal).^2);
% % mse_denoised = mean((clean_signal - denoised_signal).^2);
% % mse_improvement = 100 * (1 - mse_denoised/mse_noisy);
% % 
% % rmse_noisy = sqrt(mse_noisy);
% % rmse_denoised = sqrt(mse_denoised);
% % 
% % % Calculate correlation coefficients
% % corr_noisy = corrcoef(clean_signal, noisy_signal);
% % corr_denoised = corrcoef(clean_signal, denoised_signal);
% % 
% % % Print results
% % fprintf('Performance Metrics:\n');
% % fprintf('Original SNR: %.2f dB\n', original_snr);
% % fprintf('Denoised SNR: %.2f dB\n', denoised_snr);
% % fprintf('SNR Improvement: %.2f dB\n', snr_improvement);
% % fprintf('MSE Improvement: %.2f%%\n', mse_improvement);
% % fprintf('RMSE (Noisy): %.4f\n', rmse_noisy);
% % fprintf('RMSE (Denoised): %.4f\n', rmse_denoised);
% % fprintf('Correlation (Noisy): %.4f\n', corr_noisy(1,2));
% % fprintf('Correlation (Denoised): %.4f\n', corr_denoised(1,2));
% % 
% % %% 4. Visualize Results
% % figure;
% % subplot(3,1,1);
% % plot(t*1e6, clean_signal);
% % title('Original Clean PD Signal');
% % ylabel('Amplitude');
% % grid on;
% % 
% % subplot(3,1,2);
% % plot(t*1e6, noisy_signal);
% % title(['Noisy Signal (SNR = ', num2str(desired_snr), ' dB)']);
% % ylabel('Amplitude');
% % grid on;
% % 
% % subplot(3,1,3);
% % plot(t*1e6, denoised_signal);
% % title(['Denoised Signal (SNR = ', num2str(denoised_snr, '%.2f'), ' dB)']);
% % xlabel('Time (μs)');
% % ylabel('Amplitude');
% % grid on;
% % 
% % % Plot frequency domain comparison
% % figure;
% % N = length(t);
% % f = (0:(N/2))*fs/N;
% % 
% % F_clean = fft(clean_signal);
% % F_noisy = fft(noisy_signal);
% % F_denoised = fft(denoised_signal);
% % 
% % subplot(3,1,1);
% % semilogy(f/1e6, abs(F_clean(1:N/2+1)));
% % title('Original Clean Signal - Frequency Domain');
% % ylabel('Magnitude');
% % grid on;
% % 
% % subplot(3,1,2);
% % semilogy(f/1e6, abs(F_noisy(1:N/2+1)));
% % title('Noisy Signal - Frequency Domain');
% % ylabel('Magnitude');
% % grid on;
% % 
% % subplot(3,1,3);
% % semilogy(f/1e6, abs(F_denoised(1:N/2+1)));
% % title('Denoised Signal - Frequency Domain');
% % xlabel('Frequency (MHz)');
% % ylabel('Magnitude');
% % grid on;

%% Real data testing
% 
% load('clean_data_18mm.mat');
% load('noisy_minus5dB_18mm.mat');
% 
% % Select the first data sample to test
% clean_signal = clean_data_18mm(1,:);
% noisy_signal = noisy_data_minus5dB_18mm(1,:);
% 
% % Generate time vector
% fs = 1000e6; % Sampling frequency (1 GHz)
% t_total = length(clean_signal) / fs;
% t = 0:1/fs:t_total-1/fs;
% 
% 
% % Apply different denoising methods
% denoised_data = wdenoise(noisy_signal, 'Wavelet', 'db2', 'DenoisingMethod', 'Minimax', 'ThresholdRule', 'Soft');
% % % Define parameters
% % window_size = 13;
% % wavelet_name = 'sym4';
% % svd_threshold = 0.95;
% % 
% % % Apply hybrid denoising
% % hybrid_denoised = hybrid_denoise(noisy_data, window_size, wavelet_name, svd_threshold);
% 
% % Plotting results
% figure;
% subplot(2,2,2); plot(t, noisy_signal); title('Noisy Signal');
% subplot(2,2,1); plot(t, clean_signal); title('Clean Signal');
% subplot(2,2,3); plot(t, hybrid_denoised); title('Empirical Wavelet Transform (EWT)');
% 
% % Performance Evaluation
% 
% % Ensure dimensions match for clean_signal and wavelet_packet_denoised
% if length(denoised_signal) > length(clean_signal)
%     denoised_signal = denoised_signal(1:length(clean_signal)); % Trim to match length
% elseif length(denoised_signal) < length(clean_signal)
%     denoised_signal = [denoised_signal, zeros(1, length(clean_signal) - length(denoised_signal))]; % Pad with zeros
% end
% 
% SNR = 10 * log10(sum(clean_signal.^2) / sum((noisy_signal - clean_signal).^2));
% 
% % Calculate Signal-to-Noise Ratio (SNR)
% SNR_denoised = 10 * log10(sum(clean_signal.^2) / sum((denoised_signal - clean_signal).^2));
% 
% % Calculate Mean Squared Error (MSE)
% MSE_denoised = mean((clean_signal - denoised_signal).^2);
% 
% % Calculate Correlation Coefficient (CC)
% CC_denoised = sum(clean_signal .* denoised_signal) / sqrt(sum(clean_signal.^2) * sum(denoised_signal.^2));
% 
% fprintf('Performance Metrics:\n');
% fprintf('SNR before denoising: %.2f dB\n', SNR);
% fprintf('SNR after denoising: %.2f dB\n', SNR_denoised);
% fprintf('Mean Square Error (MSE): %.6f\n', MSE_denoised);
% fprintf('Correlation Coeffifient (CC): %.4f\n', CC_denoised);


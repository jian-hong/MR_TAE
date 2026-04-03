% Compare_Models.m
% MATLAB script for comparing MR-TAE-Fusion with conventional wavelet methods.
%
% Loads clean, noisy and deep learning denoised signals, then applies
% conventional wavelet denoising for a fair comparison.
%
% Usage:
%   Run this script after:
%   1. python examples/generate_thesis_data.py
%   2. python examples/run_inference_for_thesis.py
%
% Author: MR-TAE-Fusion Framework
% Date: 2024

clc; clear; close all;

%% Configuration
snr_levels = [-5, -10, -15, -20];
data_dir = 'Thesis_Data/';

% Wavelet parameters (standard for PD analysis)
wavelet_type = 'db4';  % Daubechies-4 (matches MR-TAE-Fusion)
wavelet_level = 5;     % Decomposition levels

fprintf('=================================================================\n');
fprintf('           PD Signal Denoising Comparison\n');
fprintf('=================================================================\n');
fprintf('Wavelet: %s, Level: %d\n\n', wavelet_type, wavelet_level);

%% Initialize Results Tables
results_wavelet = [];
results_dl = [];

%% Load Clean Signals (Reference)
load([data_dir 'Clean_Signals.mat']);
num_signals = size(clean_signals, 1);
fprintf('Loaded %d clean reference signals\n', num_signals);

%% Process Each SNR Level
for snr = snr_levels
    fprintf('\n--- Processing SNR = %d dB ---\n', snr);
    
    % Load Noisy Data
    noisy_file = sprintf('%sNoisy_SNR_%d.mat', data_dir, abs(snr));
    if ~exist(noisy_file, 'file')
        fprintf('  File not found: %s\n', noisy_file);
        continue;
    end
    load(noisy_file);
    
    % Load Deep Learning Results
    dl_file = sprintf('%sDenoised_DL_SNR_%d.mat', data_dir, abs(snr));
    if exist(dl_file, 'file')
        load(dl_file);
        has_dl = true;
    else
        fprintf('  DL file not found: %s\n', dl_file);
        has_dl = false;
    end
    
    % Initialize metric arrays
    metrics_wav = zeros(num_signals, 3);  % [SNR_out, RMSE, NCC]
    metrics_dl = zeros(num_signals, 3);
    
    for i = 1:num_signals
        clean = clean_signals(i, :);
        noisy = noisy_signals(i, :);
        
        %% 1. Conventional Method: Wavelet Soft Threshold
        [C, L] = wavedec(noisy, wavelet_level, wavelet_type);
        
        % Universal threshold (Donoho-Johnstone)
        sigma = median(abs(C)) / 0.6745;
        thr = sigma * sqrt(2 * log(length(noisy)));
        
        % Soft thresholding
        C_soft = wthresh(C, 's', thr);
        wav_out = waverec(C_soft, L, wavelet_type);
        
        % Ensure same length
        wav_out = wav_out(1:length(clean));
        
        %% 2. Calculate Metrics
        metrics_wav(i, :) = calc_stats(clean, wav_out);
        
        if has_dl
            dl_out = denoised_signals(i, :);
            dl_out = dl_out(1:length(clean));
            metrics_dl(i, :) = calc_stats(clean, dl_out);
        end
    end
    
    %% Store Average Results
    avg_wav = mean(metrics_wav, 1);
    avg_dl = mean(metrics_dl, 1);
    
    results_wavelet = [results_wavelet; snr, avg_wav];
    results_dl = [results_dl; snr, avg_dl];
    
    %% Display Results
    fprintf('\nMethod         | SNR_out (dB) | RMSE       | NCC\n');
    fprintf('---------------|--------------|------------|--------\n');
    fprintf('Wavelet (db4)  | %+10.2f   | %.6f   | %.4f\n', avg_wav(1), avg_wav(2), avg_wav(3));
    if has_dl
        fprintf('MR-TAE-Fusion  | %+10.2f   | %.6f   | %.4f\n', avg_dl(1), avg_dl(2), avg_dl(3));
        
        % Calculate improvement
        snr_imp = avg_dl(1) - avg_wav(1);
        if snr_imp > 0
            fprintf('\n  >> MR-TAE-Fusion: +%.2f dB better than Wavelet\n', snr_imp);
        end
    end
end

%% Summary Table
fprintf('\n\n=================================================================\n');
fprintf('                    SUMMARY TABLE\n');
fprintf('=================================================================\n');
fprintf('\nWavelet (db4 Soft Threshold):\n');
fprintf('SNR_in | SNR_out | SNR_imp | RMSE     | NCC\n');
fprintf('-------|---------|---------|----------|--------\n');
for i = 1:size(results_wavelet, 1)
    snr_in = results_wavelet(i, 1);
    snr_out = results_wavelet(i, 2);
    snr_imp = snr_out - snr_in;
    fprintf('%+4d   | %+6.2f  | %+6.2f  | %.6f | %.4f\n', ...
        snr_in, snr_out, snr_imp, results_wavelet(i, 3), results_wavelet(i, 4));
end

if ~isempty(results_dl)
    fprintf('\nMR-TAE-Fusion (Deep Learning):\n');
    fprintf('SNR_in | SNR_out | SNR_imp | RMSE     | NCC\n');
    fprintf('-------|---------|---------|----------|--------\n');
    for i = 1:size(results_dl, 1)
        snr_in = results_dl(i, 1);
        snr_out = results_dl(i, 2);
        snr_imp = snr_out - snr_in;
        fprintf('%+4d   | %+6.2f  | %+6.2f  | %.6f | %.4f\n', ...
            snr_in, snr_out, snr_imp, results_dl(i, 3), results_dl(i, 4));
    end
end

fprintf('\n=================================================================\n');
fprintf('Comparison Complete!\n');
fprintf('=================================================================\n');

%% Visualization (Optional)
figure('Position', [100, 100, 1200, 400]);

% SNR Improvement Plot
subplot(1, 3, 1);
snr_imp_wav = results_wavelet(:, 2) - results_wavelet(:, 1);
snr_imp_dl = results_dl(:, 2) - results_dl(:, 1);
bar([snr_imp_wav, snr_imp_dl]);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%d dB', x), snr_levels, 'UniformOutput', false));
xlabel('Input SNR'); ylabel('SNR Improvement (dB)');
legend('Wavelet', 'MR-TAE-Fusion', 'Location', 'best');
title('SNR Improvement Comparison');
grid on;

% NCC Plot
subplot(1, 3, 2);
bar([results_wavelet(:, 4), results_dl(:, 4)]);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%d dB', x), snr_levels, 'UniformOutput', false));
xlabel('Input SNR'); ylabel('NCC (Correlation)');
legend('Wavelet', 'MR-TAE-Fusion', 'Location', 'best');
title('Shape Preservation (NCC)');
ylim([0, 1]);
grid on;

% RMSE Plot
subplot(1, 3, 3);
bar([results_wavelet(:, 3), results_dl(:, 3)]);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%d dB', x), snr_levels, 'UniformOutput', false));
xlabel('Input SNR'); ylabel('RMSE');
legend('Wavelet', 'MR-TAE-Fusion', 'Location', 'best');
title('Reconstruction Error (RMSE)');
grid on;

saveas(gcf, 'Thesis_Data/Comparison_Results.png');
fprintf('Figure saved: Thesis_Data/Comparison_Results.png\n');

%% Helper Function: Calculate Statistics
function stats = calc_stats(clean, denoised)
    % Ensure row vectors
    clean = clean(:)';
    denoised = denoised(:)';
    
    % RMSE
    rmse = sqrt(mean((clean - denoised).^2));
    
    % NCC (Normalized Cross-Correlation)
    ncc = sum(clean .* denoised) / sqrt(sum(clean.^2) * sum(denoised.^2));
    if isnan(ncc)
        ncc = 0;
    end
    
    % SNR Output
    noise = clean - denoised;
    signal_power = sum(clean.^2);
    noise_power = sum(noise.^2);
    
    if noise_power > 0
        snr_out = 10 * log10(signal_power / noise_power);
    else
        snr_out = 100;  % Perfect reconstruction
    end
    
    stats = [snr_out, rmse, ncc];
end

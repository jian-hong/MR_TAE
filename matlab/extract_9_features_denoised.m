% Extract 9 features for denoised data
clc;clear;

% load noisy data         
load('C:\Users\cyx02\Downloads\Documents\FYP\Classification\noisy_minus5dB_10mm.mat'); 
% load('C:\Users\cyx02\Downloads\Documents\FYP\Classification\noisy_minus5dB_10mm.mat'); 

num_samples = 200; % Number of time points
num_channels = 1000000; % Total time in seconds
t = linspace(0, num_channels, num_samples); % Create time vector

denoised_data = savgol_wavelet_denoising(noisy_data_minus5dB_10mm);

% denoised_data = wavelet_packet_denoising(noisy_signal, 'db4', 8);

%% Savitzky-Golay + Wavelet Denoising 
function denoised_signal = savgol_wavelet_denoising(signal)
    try
        signal = signal(:)';

        % Savitzky-Golay Parameters
        window_length = max(7, floor(length(signal)/50));  % Smaller window
        if mod(window_length, 2) == 0
            window_length = window_length + 1;
        end
        poly_order = min(3, window_length - 2);  % Safe polynomial order
        
        % Ensure minimum requirements for sgolayfilt
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

%% Advanced Wavelet Packet Decomposition Method
% function denoised_signal = wavelet_packet_denoising(noisy_signal, wavelet_type, decomp_level)
%     % Advanced Wavelet Packet Decomposition Denoising
%     % Inputs:
%     %   noisy_signal: Input noisy signal
%     %   wavelet_type: Wavelet type (e.g., 'sym8', 'db4')
%     %   decomp_level: Decomposition level
% 
%     % Perform wavelet packet decomposition
%     wp = wpdec(noisy_signal, decomp_level, wavelet_type);
% 
%     % Estimate noise level
%     noise_level = estimate_noise(noisy_signal);
% 
%     % Adaptive thresholding
%     threshold = noise_level * sqrt(2 * log(length(noisy_signal)));
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

%% Soft Thresholding 
% denoised_data = wdenoise(noisy_data_minus5dB_10mm, 'Wavelet', 'db2', 'DenoisingMethod', 'Minimax', 'ThresholdRule', 'Soft');

%% Method 3 : POD
% denoised_data = pod_denoising(noisy_data, 0.06);
% 
% function data_denoised = pod_denoising(data, energy_threshold)
%     % Check the dimensions of the input data
%     [m, n] = size(data);
% 
%     % Perform SVD on the data
%     [U, S, V] = svd(data, 'econ');
% 
%     % Calculate the cumulative energy of the singular values
%     singular_values = diag(S);
%     cumulative_energy = cumsum(singular_values.^2) / sum(singular_values.^2);
% 
%     % Determine the number of modes to keep based on the energy threshold
%     if nargin < 2
%         % Default energy threshold
%         energy_threshold = 0.01;
%     end
%     nModes = find(cumulative_energy >= energy_threshold, 1);
% 
%     % Reconstruct the data using the most energetic modes
%     S_denoised = S;
%     S_denoised(nModes+1:end, nModes+1:end) = 0;
%     data_denoised = U * S_denoised * V';
% 
% end
% 
% % Method 4 : SSVD_rmse
% denoised_data = SSVD_rmse(noisy_data, 0.1);
% 
% function denoised_data = SSVD_rmse(noisy_data, threshold_rmse)
%     % Step 1: Denoising the noisy data using soft thresholding
%     denoised_data_1 = wdenoise(noisy_data, 'Wavelet', 'sym2', 'DenoisingMethod', 'SURE', 'ThresholdRule', 'Soft');
% 
%     % Perform Singular Value Decomposition (SVD)
%     [U, S, V] = svd(denoised_data_1, 'econ');
% 
%     % Estimate root mean square error (rmse) of SVD modes
%     rmse_modes = zeros(size(S, 2), 1);
%     for i = 1:size(S, 2)
%         rmse_modes(i) = sqrt(mean((denoised_data_1 - U(:, 1:i) * S(1:i, 1:i) * V(:, 1:i)').^2, 'all'));
%     end
% 
%     % Find modes with low enough rmse 
%     low_rmse_modes = find(rmse_modes < threshold_rmse);
%     % This code identifies the SVD modes that have an RMSE below a specified threshold 
% 
%     % Reconstruct clean data using low rmse modes
%     clean_data_svd = U(:, low_rmse_modes) * S(low_rmse_modes, low_rmse_modes) * V(:, low_rmse_modes)';
%     % Reconstructs the clean data using only the SVD modes with low RMSE.
% 
%     % Return the denoised data
%     denoised_data = clean_data_svd;
% end
% 

% Extract features from the denoised data 
% Denoised data 
denoised_data = denoised_data';

RMS_denoised = rms(denoised_data);    
VAR_denoised = var(denoised_data);    
SD_denoised = std(denoised_data);      
WF_denoised = RMS_denoised ./ mean(denoised_data);    % waveform factor (form factor)
K_denoised = kurtosis(denoised_data); 
SK_denoised = skewness(denoised_data);
MAX_PSP_denoised = max(pspectrum(denoised_data));    % Calculate power spectrum along the first dimension

% Define the number of columns for each split (assuming it's even)
split_cols = size(denoised_data, 2) / 4;

% Split the columns into four sets
denoised_data_1 = denoised_data(:, 1:split_cols);
denoised_data_2 = denoised_data(:, split_cols + 1:2 * split_cols);
denoised_data_3 = denoised_data(:, 2 * split_cols + 1:3 * split_cols);
denoised_data_4 = denoised_data(:, 3 * split_cols + 1:end);

% Compute the median frequency for each set of columns
MDF1 = medfreq(denoised_data_1); % Specify dimension 1 for columns
MDF2 = medfreq(denoised_data_2);
MDF3 = medfreq(denoised_data_3);
MDF4 = medfreq(denoised_data_4);
MDF_denoised = [MDF1,MDF2,MDF3,MDF4];
clear MDF1; 
clear MDF2; 
clear MDF3; 
clear MDF4; 

% % % MNF_denoised = meanfreq(denoised_data);   % mean power frequency
MNF_denoised_1 = meanfreq(denoised_data_1);
MNF_denoised_2 = meanfreq(denoised_data_2);
MNF_denoised_3 = meanfreq(denoised_data_3);
MNF_denoised_4 = meanfreq(denoised_data_4);
MNF_denoised =[MNF_denoised_1,MNF_denoised_2,MNF_denoised_3,MNF_denoised_4];
clear MNF_denoised_1;
clear MNF_denoised_2;
clear MNF_denoised_3;
clear MNF_denoised_4;

clear denoised_data_1;
clear denoised_data_2;
clear denoised_data_3;
clear denoised_data_4;

savgol_wavelet_minus5dB_10mm = [RMS_denoised; VAR_denoised; SD_denoised; WF_denoised; K_denoised; SK_denoised; MAX_PSP_denoised; MDF_denoised; MNF_denoised];
save('C:\Users\cyx02\Downloads\Documents\FYP\Classification\savgol_wavelet_9features.mat', 'savgol_wavelet_minus5dB_10mm');


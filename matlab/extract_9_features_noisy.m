
% Extract 9 features for noisy data
clc;clear;

% % load clean data
load('clean_data_10mm.mat'); % 200x1000000 double

% Adjust the desired SNR
snr_noisy = -20;  

noisy_data_minus20dB_10mm = awgn(clean_data_10mm, snr_noisy, 'measured');

save('C:\Users\cyx02\Downloads\Documents\FYP\Classification\noisy_minus20dB_10mm.mat','noisy_data_minus20dB_10mm');

noisy_data = noisy_data_minus20dB_10mm';

%% 9 Features calculation
RMS_noisy = rms(noisy_data);    
VAR_noisy = var(noisy_data);    
SD_noisy = std(noisy_data);     
WF_noisy = RMS_noisy ./ mean(noisy_data);    % waveform factor (form factor)
K_noisy = kurtosis(noisy_data);      
SK_noisy = skewness(noisy_data);
MAX_PSP_noisy = max(pspectrum(noisy_data));    % the maximum power spectrum

% Define the number of columns for each split (assuming it's even)
split_cols = size(noisy_data, 2) / 4;

% Split the columns into four sets
noisy_data_1 = noisy_data(:, 1:split_cols);
noisy_data_2 = noisy_data(:, split_cols + 1:2 * split_cols);
noisy_data_3 = noisy_data(:, 2 * split_cols + 1:3 * split_cols);
noisy_data_4 = noisy_data(:, 3 * split_cols + 1:end);

% Compute the median frequency for each set of columns
MDF1 = medfreq(noisy_data_1); % Specify dimension 1 for columns
MDF2 = medfreq(noisy_data_2);
MDF3 = medfreq(noisy_data_3);
MDF4 = medfreq(noisy_data_4);
MDF_noisy = [MDF1,MDF2,MDF3,MDF4];
clear MDF1; 
clear MDF2; 
clear MDF3; 
clear MDF4; 

% % % MNF_denoised = meanfreq(denoised_data);   % mean power frequency
MNF_noisy_1 = meanfreq(noisy_data_1);
MNF_noisy_2 = meanfreq(noisy_data_2);
MNF_noisy_3 = meanfreq(noisy_data_3);
MNF_noisy_4 = meanfreq(noisy_data_4);
MNF_noisy =[MNF_noisy_1,MNF_noisy_2,MNF_noisy_3,MNF_noisy_4];
clear MNF_noisy_1;
clear MNF_noisy_2;
clear MNF_noisy_3;
clear MNF_noisy_4;

clear noisy_data_1;
clear noisy_data_2;
clear noisy_data_3;
clear noisy_data_4;
clear noisy_data;

noisy_minus20dB_10mm = [RMS_noisy; VAR_noisy; SD_noisy; WF_noisy; K_noisy; SK_noisy; MAX_PSP_noisy; MDF_noisy; MNF_noisy];

save('C:\Users\cyx02\Downloads\Documents\FYP\Classification\noisy_minus20dB_10mm_9features.mat', 'noisy_minus20dB_10mm');


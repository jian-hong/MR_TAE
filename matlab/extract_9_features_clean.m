
% Extract 9 features for clean dat
clc; clear;

Targets = zeros(4,200);
Targets(1,1:50)=1; 
Targets(2,51:100)=1;
Targets(3,101:150)=1;
Targets(4,151:200)=1;
save('Target.mat', 'Targets'); 

% load('data1.0mm.mat');
% load('data1.8mm.mat');
% load('data2.0mm.mat');
  load('data2.5mm.mat');

% Select the first 50 rows from each clean_data_25mm matrix
selected_data1 = data25mm(1:50, :);
selected_data2 = data25mm(51:100, :);
selected_data3 = data25mm(101:150, :);
selected_data4 = data25mm(151:200, :);

% Concatenate the selected rows vertically
clean_data_25mm = [selected_data1; selected_data2; selected_data3; selected_data4];
save('clean_data_25mm.mat', 'clean_data_25mm'); 

%% 9 Features calculation

RMS = rms(clean_data_25mm');    
VAR = var(clean_data_25mm');    
SD = std(clean_data_25mm');     
WF = RMS./mean(clean_data_25mm');    % waveform factor (form factor)
K = kurtosis(clean_data_25mm'); 
SK = skewness(clean_data_25mm');       
MAX_PSP = max(pspectrum(clean_data_25mm'));    % the maximum power spectrum

% % MDF = medfreq(data3_);   % median frequency 
MDF1 = medfreq(selected_data1'); 
MDF2 = medfreq(selected_data2'); 
MDF3 = medfreq(selected_data3'); 
MDF4 = medfreq(selected_data4'); 
MDF = [MDF1, MDF2, MDF3, MDF4]; 

clear MDF1; 
clear MDF2;
clear MDF3;
clear MDF4;

% % % MNF = meanfreq(data3_);   % mean power frequency
MNF1 = meanfreq(selected_data1');
MNF2 = meanfreq(selected_data2');
MNF3 = meanfreq(selected_data3');
MNF4 = meanfreq(selected_data4');
MNF = [MNF1, MNF2, MNF3, MNF4]; 

clear MNF1; 
clear MNF2;
clear MNF3;
clear MNF4;

clean_25mm= [RMS; VAR; SD; WF; K; SK; MAX_PSP; MDF; MNF];

 %% Save feature inputs to '9 features.mat'
save('clean_25mm_9features.mat', 'clean_25mm');          
%load('9features.mat'); 


 







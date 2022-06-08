% #########################################################################
%           Receiver Operating Characteristic Curves
% #########################################################################

% Generate vector of SNR valies
Pfa = 1e-4; % Probability of false alarm
[Pd,SNR] = rocpfa(Pfa,'NumPulses',1,'SignalType','Swerling1','MaxSNR',50,'MinSNR',-50,'NumPoints',201);
% Print values as string
Pdstr = num2str(Pd','%2.10e,')
SNRstr = num2str(SNR','%2.3e,')


% Plot
figure;
plot(SNR,Pd,'-b');
title(["Swerling1", "Receiver Operating Characteristic (ROC) Curves"]);
xlabel('SNR_{1} (dB)');
ylabel('P_{d}');
grid on;

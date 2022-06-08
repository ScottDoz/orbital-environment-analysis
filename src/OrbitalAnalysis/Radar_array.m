% Radar Array Characterization

% Use the MATLAB Phased Array System Toolbox to characterize the radar
% array used in the scenario.


% Radar Array Geometry
% Planar rectangular array with rectangular lattice.
% From Radar1.rd file, total of 4356 elements (66x66) and 0.7505m spacing.
% Beamformer Mvdr.
M = 66;   % Number of elements on each row
N = 66;   % Number of elements on each column
dy = 0.7505; % Spacing between elements on each row (m)
dz = 0.7505; % Spacing between elements on each column (m)
fc = 0.45e9; % Frequency (Hz)
ura = phased.URA([N M],[dz dy]);

% Compute Directivity and gain
disp("Antenna directivity and gain:");
d = directivity(ura,fc,[0;0],'PropagationSpeed',physconst('LightSpeed'))
gain = phased.ArrayGain('SensorArray',ura);
g = gain(fc,[0;0])

% Show the array
figure; viewArray(ura,'Title','Uniform Rectangular Array (URA)');
% Note: Array element positions do not match up exactly.

% Show Directivity pattern
%figure; pattern(ura,fc,'Weights',w); % 3D view
figure; patternAzimuth(ura,fc);
%figure; patternElevation(ura,fc);
figure;pattern(ura,fc,[-180:180],0,'PropagationSpeed',physconst('LightSpeed'),'CoordinateSystem','rectangular');

% Minimum-variance distortionless-response (MVDR) Beamformer.
Np = -133.975; % System Noise (dBW) - from Ts=290K, Ts=1e-7
Sn  = sensorcov(ura.getElementPosition,0,db2pow(-110)); % Sesnor covariance
w = mvdrweights(ura.getElementPosition,0,Sn);
%beamformer = phased.MVDRBeamformer('SensorArray',ura,'OperatingFrequency',0.45e9);


%% Beamforming example -----------------------------------------------------
% https://www.mathworks.com/help/phased/ref/phased.mvdrbeamformer-system-object.html



% Create signal (rectangular pulse)
t = 0:0.001:0.3;                % Time, sampling frequency is 1kHz
xm = zeros(size(t));  
xm = xm(:);                       % Signal in column vector
xm(201:205) = xm(201:205) + 1;    % Define the pulse

% Create signal
t = [0:.1:200]';
fr = 1e-7; %.01; % Signal frequency (Hz)
xm = sin(2*pi*fr*t);

c = physconst('LightSpeed');
fc = 450e6; % Carrier frequency (Hz)
rng('default');
incidentAngle = [0;0]; % Incident angle (az,el)

% Create array
%array = phased.ULA('NumElements',5,'ElementSpacing',0.5);
array = ura;
x = collectPlaneWave(array,xm,incidentAngle,fc,c);
noise = 0.1*(randn(size(x)) + 1j*randn(size(x)));
rx = x + noise;

% Compute beamforming weights
beamformer = phased.MVDRBeamformer('SensorArray',array,...
    'PropagationSpeed',c,'OperatingFrequency',fc,...
    'Direction',incidentAngle,'WeightsOutputPort',true);
[y,w] = beamformer(rx);


%% Plot the signal
plot(t,real(rx(:,3)),'r:',t,real(y))
xlabel('Time')
ylabel('Amplitude')
legend('Original','Beamformed')

%% Plot the array response

figure; 
pattern(array,fc,[-180:180],0,'PropagationSpeed',c,'CoordinateSystem','rectangular');
hold on;
pattern(array,fc,[-180:180],0,'PropagationSpeed',c,'Weights',w,'CoordinateSystem','rectangular');
legend('Original','MVDR');

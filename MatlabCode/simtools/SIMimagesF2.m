function [S1aTnoisy S2aTnoisy S3aTnoisy ...
    S1bTnoisy S2bTnoisy S3bTnoisy ...
    S1cTnoisy S2cTnoisy S3cTnoisy ...
    DIoTnoisy DIoT] = SIMimagesF2(k2,...
    DIo,PSFo,OTFo,ModFac,NoiseLevel,UsePSF,NoiseMode,NoiseFrac)

w = size(DIo,1);
wo = w/2;
x = linspace(0,w-1,w);
y = linspace(0,w-1,w);
[X,Y] = meshgrid(x,y);

%% illunination phase shifts along the three directions
p0Ao = 0*pi/3;
p0Ap = 2*pi/3;
p0Am = 4*pi/3;
p0Bo = 0*pi/3;
p0Bp = 2*pi/3;
p0Bm = 4*pi/3;
p0Co = 0*pi/3;
p0Cp = 2*pi/3;
p0Cm = 4*pi/3;

%% Illuminating patterns
alpha = 0*pi/6;
% orientation direction of illumination patterns
thetaA = 0*pi/3 + alpha;
thetaB = 1*pi/4 + alpha;
thetaC = -1*pi/4 + alpha;
%{
thetaA = 0*pi/3 + alpha; 
thetaB = 1*pi/3 + alpha;
thetaC = 2*pi/3 + alpha;
%}
% illumination frequency vectors
k2a = (k2/w).*[cos(thetaA) sin(thetaA)];
k2b = (k2/w).*[cos(thetaB) sin(thetaB)];
k2c = (k2/w).*[cos(thetaC) sin(thetaC)];
% -------------------------------------------------------
% mean illumination intensity
mA = 1;
mB = 1;
mC = 1;
% amplitude of illumination intensity above mean
aA = mA*ModFac;
aB = mA*ModFac;
aC = mA*ModFac;

% random phase shift errors
NN = 1*(0.5-rand(9,1))*pi/18;

% illunination phase shifts with random errors
psAo = p0Ao + NN(1,1);
psAp = p0Ap + NN(2,1);
psAm = p0Am + NN(3,1);
psBo = p0Bo + NN(4,1);
psBp = p0Bp + NN(5,1);
psBm = p0Bm + NN(6,1);
psCo = p0Co + NN(7,1);
psCp = p0Cp + NN(8,1);
psCm = p0Cm + NN(9,1);

% illunination patterns
sAo = mA + aA*cos(2*pi*(k2a(1,1).*(X-wo)+k2a(1,2).*(Y-wo))+psAo); % illuminated signal (0 phase)
sAp = mA + aA*cos(2*pi*(k2a(1,1).*(X-wo)+k2a(1,2).*(Y-wo))+psAp); % illuminated signal (+ phase)
sAm = mA + aA*cos(2*pi*(k2a(1,1).*(X-wo)+k2a(1,2).*(Y-wo))+psAm); % illuminated signal (- phase)
sBo = mB + aB*cos(2*pi*(k2b(1,1).*(X-wo)+k2b(1,2).*(Y-wo))+psBo); % illuminated signal (0 phase)
sBp = mB + aB*cos(2*pi*(k2b(1,1).*(X-wo)+k2b(1,2).*(Y-wo))+psBp); % illuminated signal (+ phase)
sBm = mB + aB*cos(2*pi*(k2b(1,1).*(X-wo)+k2b(1,2).*(Y-wo))+psBm); % illuminated signal (- phase)
sCo = mC + aC*cos(2*pi*(k2c(1,1).*(X-wo)+k2c(1,2).*(Y-wo))+psCo); % illuminated signal (0 phase)
sCp = mC + aC*cos(2*pi*(k2c(1,1).*(X-wo)+k2c(1,2).*(Y-wo))+psCp); % illuminated signal (+ phase)
sCm = mC + aC*cos(2*pi*(k2c(1,1).*(X-wo)+k2c(1,2).*(Y-wo))+psCm); % illuminated signal (- phase)


%% superposed Objects
s1a = DIo.*sAo; % superposed signal (0 phase)
s2a = DIo.*sAp; % superposed signal (+ phase)
s3a = DIo.*sAm; % superposed signal (- phase)
s1b = DIo.*sBo;
s2b = DIo.*sBp;
s3b = DIo.*sBm;
s1c = DIo.*sCo;
s2c = DIo.*sCp;
s3c = DIo.*sCm;


%% superposed (noise-free) Images
PSFsum = sum(sum(PSFo));
if ( UsePSF == 1 )
    DIoT = conv2(DIo,PSFo,'same')./PSFsum;
    S1aT = conv2(s1a,PSFo,'same')./PSFsum;
    S2aT = conv2(s2a,PSFo,'same')./PSFsum;
    S3aT = conv2(s3a,PSFo,'same')./PSFsum;
    S1bT = conv2(s1b,PSFo,'same')./PSFsum;
    S2bT = conv2(s2b,PSFo,'same')./PSFsum;
    S3bT = conv2(s3b,PSFo,'same')./PSFsum;
    S1cT = conv2(s1c,PSFo,'same')./PSFsum;
    S2cT = conv2(s2c,PSFo,'same')./PSFsum;
    S3cT = conv2(s3c,PSFo,'same')./PSFsum;
else
    DIoT = ifft2( fft2(DIo).*fftshift(OTFo) );
    S1aT = ifft2( fft2(s1a).*fftshift(OTFo) );
    S2aT = ifft2( fft2(s2a).*fftshift(OTFo) );
    S3aT = ifft2( fft2(s3a).*fftshift(OTFo) );
    S1bT = ifft2( fft2(s1b).*fftshift(OTFo) );
    S2bT = ifft2( fft2(s2b).*fftshift(OTFo) );
    S3bT = ifft2( fft2(s3b).*fftshift(OTFo) );
    S1cT = ifft2( fft2(s1c).*fftshift(OTFo) );
    S2cT = ifft2( fft2(s2c).*fftshift(OTFo) );
    S3cT = ifft2( fft2(s3c).*fftshift(OTFo) );

    DIoT = real(DIoT);
    S1aT = real(S1aT);
    S2aT = real(S2aT);
    S3aT = real(S3aT);
    S1bT = real(S1bT);
    S2bT = real(S2bT);
    S3bT = real(S3bT);
    S1cT = real(S1cT);
    S2cT = real(S2cT);
    S3cT = real(S3cT);
end



aNoise = NoiseLevel; % corresponds to 10% noise
pNoise = NoiseLevel/6; % corresponds to 10% noise
beta=2;

%% noise added raw SIM images
scale_ref = double(max(DIo(:))) * (1 + ModFac);
if ~isfinite(scale_ref) || scale_ref <= 0
    scale_ref = 1;
end

DIoTnoisy = local_add_noise_scaled(DIoT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);
S1aTnoisy = local_add_noise_scaled(S1aT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);
S2aTnoisy = local_add_noise_scaled(S2aT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);
S3aTnoisy = local_add_noise_scaled(S3aT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);
S1bTnoisy = local_add_noise_scaled(S1bT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);
S2bTnoisy = local_add_noise_scaled(S2bT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);
S3bTnoisy = local_add_noise_scaled(S3bT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);
S1cTnoisy = local_add_noise_scaled(S1cT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);
S2cTnoisy = local_add_noise_scaled(S2cT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);
S3cTnoisy = local_add_noise_scaled(S3cT, NoiseMode, NoiseLevel, NoiseFrac, scale_ref);

originalImage = im2double(DIoT);
noisyImage = im2double(DIoTnoisy);

[peaksnr, snr] = psnr(noisyImage, originalImage);

fprintf('The SNR in dB is: %.2f\n', snr);


logFilePath = '/your log path';
currentDateTime = datetime("now");

fileID = fopen(logFilePath, 'a');
if fileID == -1
    error('Failed to open the log file. Please check the file path.');
end

logMessage = sprintf('%s - The SNR in dB is: %.2f\n%s - The PSNR in dB is: %.2f\n', ...
    currentDateTime, snr, currentDateTime, peaksnr);
fprintf(fileID, logMessage);
fclose(fileID);
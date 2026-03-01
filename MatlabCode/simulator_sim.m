close all
clc
addpath(genpath('simtools'));

w = 502;
wo = w/2;
x = linspace(0, w-1, w);
y = linspace(0, w-1, w);
[X, Y] = meshgrid(x, y);

% PSF,OTF
scale = 1.3; 
[PSFo, OTFo] = PsfOtf(w, scale);

%% 定义噪声水平
NoiseLevels = 0; %[5, 10, 15, 25, 50];
NoiseMode = 1; % 1'Gaussian' | 2'Poisson' | 3'GaussianPoisson'
NoiseFrac = 0.2; %for NoiseMode=3, 0.2-0.5 Poisson is Noiselevel，Gaussian is NoiseFrac*NoiseLevels

folderPath = '/your folder';
files = dir(fullfile(folderPath, 'e_*.tif'));
[header, data] = XxReadMRC('/your mrc root');

for k = 1:length(files)

    Io1 = imread(fullfile(folderPath, files(k).name));
    Io = Io1;
    DIo = double(Io);

    % 循环不同的噪声水平
    for n = 1:length(NoiseLevels)
        k2 = 75.23; 
        ModFac = 0.8; 
        NoiseLevel = NoiseLevels(n); 
        UsePSF = 0; 
        
        [S1aTnoisy S2aTnoisy S3aTnoisy ...
         S1bTnoisy S2bTnoisy S3bTnoisy ...
         S1cTnoisy S2cTnoisy S3cTnoisy ...
         DIoTnoisy DIoT] = SIMimagesF2(k2, DIo, PSFo, OTFo, ModFac, NoiseLevel, UsePSF, NoiseMode,NoiseFrac);

        Iraw = cat(3, S1aTnoisy, S2aTnoisy, S3aTnoisy, S1bTnoisy, S2bTnoisy, S3bTnoisy, S1cTnoisy, S2cTnoisy, S3cTnoisy);

        gtbaseFileName = sprintf('Cell%s_snr_%02d_gt', files(k).name(5:end-4), n);
        gtfileName = fullfile('G:\models\00MyDataAnalysis-tool\p11\simulation\e_tif\mix_add', [gtbaseFileName '.tif']);
        outDir = fileparts(gtfileName);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        
        [fid, msg] = fopen(fullfile(outDir,'__write_test__.tmp'), 'w');
        if fid < 0
            error('Output folder not writable: %s\nReason: %s', outDir, msg);
        end
        fclose(fid);
        delete(fullfile(outDir,'__write_test__.tmp'));


        XxWriteTiff(im2uint16(mat2gray(DIoT)), gtfileName);

        baseFileName = sprintf('Cell%s_snr_%02d', files(k).name(5:end-4), n);
        save_name = [baseFileName, '.mrc'];
        handle = fopen(['/your path', '/', save_name], 'w+');
        header(1) = 502; 
        header(2) = 502;
        header(3) = 9;
        handle = XxWriteMRC_SmallEndian(handle, Iraw, header);
        fclose(handle);


    end
end

disp('Done!');

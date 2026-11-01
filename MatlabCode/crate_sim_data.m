clc;
clear;
close all;

%% Settings
%% ER  Microtubules  F-actin
Smpl_name = 'Microtubules';
File_dir = 'G:\results\p11\BioSR_data\raw\Microtubules';

Save_file_dir = 'G:\dataset\p11_data\Our dataset\XxMatlabUtils/TEST';

view_count = 4;
Save_file_dir = [Save_file_dir,'/',Smpl_name];
if ~exist(Save_file_dir,'dir')
    mkdir(Save_file_dir);
end

file_list = dir([File_dir,'/*.mrc']);

file_count = length(file_list);

for file_id = 1: 1: file_count
    file_name = file_list(file_id).name;
    
    cur_save_dir = [Save_file_dir,'/',file_name(1:end-4)];
    if ~exist(cur_save_dir,'dir')
        mkdir(cur_save_dir);
    end
    
    [header, data] = XxReadMRC([File_dir,'/', file_name]);

    Nx = double(header(1));
    Ny = double(header(2));
    N_slice = double(header(3));

    data = double(reshape(data,[Nx, Ny, N_slice]));
   
    header_out = header;
    img_raw = data;
    for view_id = 1: 1: view_count
        if view_id == 1
            img_raw_v = img_raw(1:2:end,1:2:end,:);
        elseif view_id == 2
            img_raw_v = img_raw(2:2:end,1:2:end,:);
        elseif view_id == 3
            img_raw_v = img_raw(2:2:end,2:2:end,:);
        elseif view_id == 4
            img_raw_v = img_raw(1:2:end,2:2:end,:);
        else
            print('view id cannot be larger than 4');
        end
        
        img_raw_v = fourier_resize2d_stack(img_raw_v, Nx, Ny);  %fourier_resize2d_stack_padded  fourier_resize2d_stack
        

        if view_id == 1
            img_raw_v = imtranslate(img_raw_v,[-0.5,-0.5]);
        elseif view_id == 2
            img_raw_v = imtranslate(img_raw_v,[-0.5,0.5]);
        elseif view_id == 3
            img_raw_v = imtranslate(img_raw_v,[0.5,0.5]);
        elseif view_id == 4
            img_raw_v = imtranslate(img_raw_v,[0.5,-0.5]);
        else                 
            print('view id cannot be larger than 4');
        end 
        
        save_name = ['view',num2str(view_id),'.mrc'];
        
        handle = fopen([cur_save_dir,'/',save_name],'w+');
        handle = XxWriteMRC_SmallEndian(handle, img_raw_v, header);
        fclose(handle);
    end   
end

function out = fourier_resize2d_stack(in, outNx, outNy)
% Fourier interpolation resize to exact size outNx x outNy
% in:  [Mx, My, K] or [Mx, My]
% out: [outNx, outNy, K]

    if ndims(in) == 2
        in = reshape(in, size(in,1), size(in,2), 1);
    end
    in = double(in);

    [Mx, My, K] = size(in);
    out = zeros(outNx, outNy, K);

    for k = 1:K
        tmp = interpft(in(:,:,k), outNx, 1); 
        tmp = interpft(tmp,       outNy, 2);  
        tmp = real(tmp);
        tmp(tmp < 0) = 0;                
        out(:,:,k) = tmp;
    end

    if size(out,3) == 1
        out = out(:,:,1);
    end
end


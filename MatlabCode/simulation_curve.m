clc; clear;
H=502; W=502;
N=20;
out_dir = '/your root';
if ~exist(out_dir,'dir'); mkdir(out_dir); end

rng(0);

for t=1:N
    nCurves = randi([15, 30]);

    I_a = zeros(H,W);
    for k=1:nCurves
        opts = struct();
        opts.radius = 1;
        opts.blur_sigma = 1.0;


        [I_tube, I_center] = fun_curveimage_gen2_tube(H, opts);
        I_a = max(I_a, I_tube);

    end
    I_a = mat2gray(I_a);

    I_b = apply_notches(I_a, 2, randi([80,180]), 0.05);

    % [dx,dy] = make_elastic_field(H,W,16,14,12);
    % I_d = warp_with_field(I_b,dx,dy);
    I_d = I_b;

    % e
    I_e = imgaussfilt(I_d, 1.0);
    I_e = mat2gray(I_e);

    imwrite(I_e, fullfile(out_dir, sprintf('e_%05d.png', t)));
    %imwrite(uint16(round(I_e*65535)), fullfile(out_dir, sprintf('e_%05d.png', t)));
end

function Iout = apply_notches(Iin, sigmaNotch, nNotches, th)
    I = mat2gray(Iin);
    maskLine = I > 0.05;
    [yy, xx] = find(maskLine);
    if isempty(xx)
        Iout = I;
        return;
    end

    nNotches = min(nNotches, numel(xx));
    pick = randperm(numel(xx), nNotches);

    imp = zeros(size(I));
    imp(sub2ind(size(I), yy(pick), xx(pick))) = 1;

    g = imgaussfilt(imp, sigmaNotch);
    notchMask = (g > th) & maskLine;

    Iout = I;
    Iout(notchMask) = 0;
    Iout = imgaussfilt(Iout, 0.4);
    Iout = mat2gray(Iout);
end

function [I_tube, I_center] = fun_curveimage_gen2_tube(size_img, opts)

if nargin < 2
    opts = struct();
end

% -------- defaults --------
if ~isfield(opts,'num_run');          opts.num_run = 40000; end
if ~isfield(opts,'var_range');        opts.var_range = [0.02, 0.08]; end
if ~isfield(opts,'at_range');         opts.at_range  = [10, 35]; end
if ~isfield(opts,'radius');           opts.radius = 2; end
if ~isfield(opts,'blur_sigma');       opts.blur_sigma = 1.0; end
if ~isfield(opts,'intensity_range');  opts.intensity_range = [0.6, 1.0]; end

H = size_img; W = size_img;

var = opts.var_range(1) + (opts.var_range(2) - opts.var_range(1)) * rand();
a_t = randi(opts.at_range);

% row/col：p(:,1)=row(y), p(:,2)=col(x)
p0 = round([H, W] .* rand(1,2));
p0 = max(min(p0, [H-1, W-1]), [2,2]);

p = zeros(opts.num_run, 2);
v = zeros(opts.num_run, 2);

p(1,:) = p0;

v(1,:) = ([H/2, W/2] - p(1,:)) / H * 2;
v(1,:) = v(1,:) / (norm(v(1,:)) + 1e-12);

a = 0;
last_i = 1;

for i = 2:opts.num_run
    if mod(i, a_t) == 0
        a = rand() - 0.5;
    end

    v(i,:) = v(i-1,:) + var * a * [-v(i-1,2), v(i-1,1)];
    v(i,:) = v(i,:) / (norm(v(i,:)) + 1e-12);

    p(i,:) = p(i-1,:) + v(i,:);

    p(i,1) = max(min(p(i,1), H-1), 2);
    p(i,2) = max(min(p(i,2), W-1), 2);

    last_i = i;

    if p(i,1)==2 || p(i,1)==H-1 || p(i,2)==2 || p(i,2)==W-1
        break;
    end
end

rp = round(p(1:last_i,:));
rp(:,1) = max(min(rp(:,1), H), 1);
rp(:,2) = max(min(rp(:,2), W), 1);

I_center = false(H,W);
idx = sub2ind([H,W], rp(:,1), rp(:,2));
I_center(idx) = true;

se = strel('disk', opts.radius, 0);
I_tube = imdilate(I_center, se);
I_tube = imgaussfilt(double(I_tube), opts.blur_sigma);

inten = opts.intensity_range(1) + (opts.intensity_range(2) - opts.intensity_range(1))*rand();
I_tube = mat2gray(I_tube) * inten;

end

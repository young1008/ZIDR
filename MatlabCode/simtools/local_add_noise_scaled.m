function Out = local_add_noise_scaled(I, NoiseMode, NoiseLevel, NoiseFrac, scale_ref)
    clip01 = @(A) min(max(A, 0), 1);

    I = double(I);

    x = I / double(scale_ref);
    x = clip01(x);

    switch NoiseMode
        case 1  % Gaussian: sigma = noise_level/255
            sigma = double(NoiseLevel) / 255.0;
            y = x + randn(size(x)) * sigma;
            y = clip01(y);

        case 2  % Poisson: Poisson(x*L)/L
            L = double(NoiseLevel);
            if L <= 0
                y = x;
            else
                y = poissrnd(x * L) / L;
                y = clip01(y);
            end

        case 3  % Mixed: Poisson then Gaussian(read noise)
            L = double(NoiseLevel);
            if L <= 0
                y = x;
            else
                y = poissrnd(x * L) / L;
                y = clip01(y);
            end

            sigma = double(NoiseFrac) * double(NoiseLevel) / 255.0;
            y = y + randn(size(y)) * sigma;
            y = clip01(y);

        otherwise
            error('Unknown NoiseMode: %d (1=Gaussian,2=Poisson,3=Mixed)', NoiseMode);
    end

    Out = y * double(scale_ref);
end

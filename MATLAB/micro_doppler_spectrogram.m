function [f, t, SPT] = micro_doppler_spectrogram(F, k_tau_star, N_bins)

    % =====================================
    % 1. IFFT → dominio delay
    % =====================================
    R = ifft(F, [], 1);

    % =====================================
    % 2. Selección de bins de delay
    % =====================================
    % Integrar varios bins para mejorar SNR
    if nargin < 3
        N_bins = 1; % por defecto solo 1 bin
    end

    idx = (k_tau_star - N_bins):(k_tau_star + N_bins);

    % Control de bordes
    idx = idx(idx >= 1 & idx <= size(R,1));

    r = sum(R(idx,:), 1);   % señal en tiempo lento

    % =====================================
    % 3. CWT (Wavelet)
    % =====================================
    fs = 1;  % frecuencia de muestreo (normalizada)

    [wt, f] = cwt(r, 'amor', fs);

    % =====================================
    % 4. Espectrograma en dB
    % =====================================
    SPT = 20 * log10(abs(wt) + 1e-12);

    % Normalización opcional (mejora visual)
    SPT = SPT - max(SPT(:));

    % =====================================
    % 5. Eje temporal
    % =====================================
    t = 1:length(r);

end
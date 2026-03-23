function F_out = ota_compensation_pilots(Yp, Xp, delta_f, N_s, k_tau_sens, delta_T_as, P, N, interpolate)

    % --- 1. Canal en pilotos ---
    Fp = Yp ./ Xp;

    % --- 2. Índices ---
    [Np, M] = size(Fp);
    n = P(:);         % índices reales de subportadora
    m = 0:M-1;

    % --- 3. Corrección de fase SOLO en pilotos ---
    phase = exp(1j * 2 * pi * n * delta_f .* ...
               (k_tau_sens + m * N_s * delta_T_as));

    Fp_tilde = Fp .* phase;

    % --- 4. Interpolar o no ---
    if interpolate
        F_out = interpolate_channel(Fp_tilde, P, N);
    else
        F_out = Fp_tilde;
    end

end
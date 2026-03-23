function F_tilde = ota_compensation(B_UE, B_tx, delta_f, N_s, k_tau_sens, delta_T_as)

    [N, M] = size(B_UE);

    % --- 1. Estimación de canal ---
    F = B_UE ./ B_tx;

    % --- 2. Índices ---
    n = (0:N-1).';   % columna
    m = (0:M-1);     % fila

    % --- 3. Corrección de fase ---
    phase = exp(1j * 2 * pi * n * delta_f .* ...
               (k_tau_sens + m * N_s * delta_T_as));

    % --- 4. Aplicar corrección ---
    F_tilde = F .* phase;

end
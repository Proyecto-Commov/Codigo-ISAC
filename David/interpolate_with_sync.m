function F_final = interpolate_with_sync(Yp, Xp, P, Y_sync, Z_sync, N)

    % =====================================
    % 1. Canal en pilotos
    % =====================================
    Fp = Yp ./ Xp;   % (Np x M)

    M = size(Fp,2);

    % =====================================
    % 2. Interpolación en frecuencia
    % =====================================
    F_interp = zeros(N, M);

    for m = 1:M
        F_interp(:,m) = interp1(P, Fp(:,m), 1:N, 'linear', 'extrap');
    end

    % =====================================
    % 3. Canal en símbolo de sincronización
    % =====================================
    H_sync = Y_sync ./ (Z_sync + 1e-12);   % (N x 1)

    % SOLO usamos la magnitud
    H_sync_mag = abs(H_sync);

    % =====================================
    % 4. Normalización de la interpolación
    % =====================================
    % Calculamos la magnitud media del canal interpolado
    mean_mag = mean(abs(F_interp), 2);   % (N x 1)

    % Evitar divisiones raras
    mean_mag(mean_mag < 1e-12) = 1e-12;

    % Factor de corrección en frecuencia
    correction = H_sync_mag ./ mean_mag;

    % =====================================
    % 5. Aplicar corrección
    % =====================================
    F_final = F_interp .* correction;

end
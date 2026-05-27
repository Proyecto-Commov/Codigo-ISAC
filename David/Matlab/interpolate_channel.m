function F_interp = interpolate_channel(F_pilots, P, N)

    % Interpola en frecuencia (para cada símbolo OFDM)
    M = size(F_pilots, 2);
    F_interp = zeros(N, M);

    for m = 1:M
        F_interp(:, m) = interp1(P, F_pilots(:, m), 1:N, 'linear', 'extrap');
    end

end
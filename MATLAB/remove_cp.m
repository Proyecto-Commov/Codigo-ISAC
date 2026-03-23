function B_noCP = remove_cp(B_time, N, N_cp)

    % B_time: señal en tiempo (vector o matriz)
    % N: tamaño útil OFDM
    % N_cp: tamaño del CP

    % Elimina CP por símbolo
    B_noCP = B_time(N_cp+1:N_cp+N, :);

end
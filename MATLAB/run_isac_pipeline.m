function [f, t, SPT] = run_isac_pipeline(B_UE, B_tx, params)

    % =====================================
    % 1. (Opcional) eliminar CP
    % =====================================
    if params.remove_cp                                %solo si la señal nos la dan en tiempo
        B_UE = remove_cp(B_UE, params.N, params.N_cp);
        B_tx = remove_cp(B_tx, params.N, params.N_cp);
    end

    % =====================================
    % 2. Extraer pilotos
    % =====================================
    [Yp, Xp] = extract_pilots(B_UE, B_tx, params.P);

    % =====================================
    % 3. Canal + compensación + interpolación opcional
    % =====================================
    F = ota_compensation_pilots( ...
        Yp, Xp, ...
        params.delta_f, ...
        params.N_s, ...
        params.k_tau_sens, ...
        params.delta_T_as, ...
        params.P, ...
        params.N, ...
        params.interpolate);

    % =====================================
    % 4. Clutter (opcional)
    % =====================================
    if params.use_mti
        F = mti_clutter_suppression(F);
    end

    % =====================================
    % 5. Espectrograma
    % =====================================
    [f, t, SPT] = micro_doppler_spectrogram( ...
        F, ...
        params.k_tau_star, ...
        params.M_w, ...
        params.M_H);

end

%% PROYECTO COMMOV: FILTRADO DE CLUTTER Y SEPARACIÓN DE FUENTES (E3.4)
% Este script procesa la señal diferencial antes de la STFT.

%% 1. Configuración de Parámetros
fs_slow = 140;          % Frecuencia de muestreo (Nyquist para 70Hz Doppler) [cite: 38]
ema_alpha = 0.95;       % Memoria del filtro (Alta para no borrar el fuego)
corte_doppler_hz = 30;  % Límite para plumas térmicas (Filtro de Velocidad) [cite: 27]

% Supongamos que recibimos la matriz 'S' (Símbolos x Subportadoras)
% tras la diferencia de fase: S[m,k] = H[m,k] * conj(H[m+1,k])
S = randn(1000, 64) + 1j*randn(1000, 64); 
%% 2. Selección de Subportadoras Sensibles (Aislamiento Térmico)
% No todas las subportadoras ven igual la ionización/turbulencia.
% Elegimos las que tienen mayor varianza temporal (donde "pasa algo").
varianza_k = var(abs(S), 0, 1); % Varianza de cada columna (subportadora) [cite: 523] 
[~, idx_top] = sort(varianza_k, 'descend');
subportadoras_seleccionadas = idx_top(1:10); % Nos quedamos con las 10 mejores

% Promediamos solo las mejores para reducir ruido
s_raw = mean(S(:, subportadoras_seleccionadas), 2);

%% 3. Eliminación de Clutter Estático (EMA - Paso Alto)
% El EMA aprende el fondo (paredes) y lo resta.
s_clutter_free = zeros(size(s_raw));
buffer_lento = s_raw(1);

for t = 1:length(s_raw)
    buffer_lento = ema_alpha * buffer_lento + (1 - ema_alpha) * s_raw(t);
    s_clutter_free(t) = s_raw(t) - buffer_lento;
end

%% 4. Filtro de Velocidad (Butterworth Paso Bajo)
% Elimina personas moviéndose rápido (> 3 m/s o > 70 Hz)[cite: 38, 135].
Wn = corte_doppler_hz / (fs_slow/2); 
[b, a] = butter(4, Wn, 'low');
s_filt = filter(b, a, s_clutter_free);

%% 5. Métrica de Separación: Entropía Espectral (Caos vs. Ritmo)
% El fuego es caótico (alta entropía); el humano es rítmico.
% Calculamos la entropía sobre la señal filtrada.
psd = abs(fft(s_filt)).^2;
psd_norm = psd / sum(psd);
entropia = -sum(psd_norm .* log2(psd_norm + eps));

fprintf('Entropía detectada: %.2f (Valores altos indican Fuego)\n', entropia);

%% 6. Generación del Espectrograma Final
nperseg = 64; 
noverlap = 48;
[spec, F, T] = spectrogram(s_filt, hamming(nperseg), noverlap, nperseg, fs_slow, 'centered');

% Visualización
figure;
imagesc(T, F, 10*log10(abs(spec)));
axis xy; colormap('jet'); colorbar;
xlabel('Tiempo (s)'); ylabel('Frecuencia Doppler (Hz)');
title(['Espectrograma Limpio - Entropía: ', num2str(entropia)]);
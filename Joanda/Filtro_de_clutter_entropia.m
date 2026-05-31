%% PROYECTO COMMOV: FILTRADO DE CLUTTER Y SEPARACIÓN DE FUENTES (E3.4)
% Este script procesa la señal diferencial antes de la STFT.

%% 1. Configuración de Parámetros
fs_slow = 140;          % Frecuencia de muestreo (Nyquist para 70Hz Doppler) [cite: 38]
ema_alpha = 0.95;       % Memoria del filtro (Alta para no borrar el fuego) [cite: 133]
corte_doppler_hz = 30;  % Límite para plumas térmicas (Filtro de Velocidad) [cite: 27, 31]

% --- SELECCIÓN DE MODO DE PRUEBA ---
modo = 'fuego'; % Cambia a 'fuego' para ver la diferencia de entropía

if strcmp(modo, 'fuego')
    % Simulación de Fuego (Ruido caótico) 
    S = randn(1000, 64) + j*randn(1000, 64); 
else
    % Simulación de Persona Caminando (Señal rítmica/ordenada) [cite: 135, 460]
    t_vec = (0:999)' / fs_slow;
    f_pasos = 2; % 2 Hz (ritmo de pasos)
    % Creamos una matriz donde todas las subportadoras tienen la misma señal rítmica
    S = repmat(cos(2*pi*f_pasos*t_vec), 1, 64) + 0.1*(randn(1000,64) + 1j*randn(1000,64));
end

%% 2. Selección de Subportadoras Sensibles (Aislamiento Térmico)
% Buscamos las subportadoras con mayor variabilidad[cite: 523].
varianza_k = var(abs(S), 0, 1); 
[~, idx_top] = sort(varianza_k, 'descend');
subportadoras_seleccionadas = idx_top(1:10); 

% Promediamos las seleccionadas para obtener un vector columna (M x 1)
s_raw = mean(S(:, subportadoras_seleccionadas), 2);

%% 3. Eliminación de Clutter Estático (EMA - Paso Alto)
% El EMA estima el DC residual para restarlo[cite: 133, 219].
s_clutter_free = zeros(size(s_raw));
buffer_lento = s_raw(1);
for i = 1:length(s_raw)
    buffer_lento = ema_alpha * buffer_lento + (1 - ema_alpha) * s_raw(i);
    s_clutter_free(i) = s_raw(i) - buffer_lento;
end

%% 4. Filtro de Velocidad (Butterworth Paso Bajo)
% Atenúa componentes rápidas de humanos (> 3 m/s)[cite: 38, 135].
Wn = corte_doppler_hz / (fs_slow/2); 
[b, a] = butter(4, Wn, 'low');
s_filt = filter(b, a, s_clutter_free);

%% 5. Métrica de Separación: Entropía Espectral NORMALIZADA
% El fuego -> 1; el humano -> 0[cite: 135, 529].
psd = abs(fft(s_filt)).^2;
psd_norm = psd / sum(psd + eps); 
entropia_abs = -sum(psd_norm .* log2(psd_norm + eps));
entropia_normalizada = entropia_abs / log2(length(psd_norm));

fprintf('Modo: %s | Entropía Normalizada: %.4f\n', modo, entropia_normalizada);

%% 6. Generación del Espectrograma Final
nperseg = 64; 
noverlap = 48;
[spec, F, T] = spectrogram(s_filt, hamming(nperseg), noverlap, nperseg, fs_slow, 'centered');

% Visualización [cite: 114, 537]
figure('Color', 'w');
imagesc(T, F, 10*log10(abs(spec) + eps));
axis xy; colormap('jet'); colorbar;
ylabel('Frecuencia Doppler (Hz)'); xlabel('Tiempo (s)');
title(['Modo: ', modo, ' - Entropía: ', num2str(entropia_normalizada, '%.4f')]);
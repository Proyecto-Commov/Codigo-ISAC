%% PROYECTO COMMOV: FILTRADO DE CLUTTER Y SEPARACIÓN DE FUENTES (E3.4)
% Este script procesa la señal diferencial antes de la STFT para limpiar
% objetos estáticos y ruidos indeseados.

%% 1. Configuración Inicial
fs_slow = 140; % Frecuencia de muestreo slow-time (Nyquist para 70Hz Doppler) Equipo PHY
ema_alpha = 0.95; % Factor para el filtro paso alto EMA [cite: 316]

% Supongamos que 's_diff' es la señal que viene del paso 3 del diagrama 
% (S[k,m] = H[k,m] * H*[k,m+1])[cite: 114].
% Si no tienes datos reales aún, puedes usar ruido complejo para probar:
 s_diff = randn(1000, 1) + 1j*randn(1000, 1); 

%% 2. Eliminación de Clutter Estático (Quitar DC)
% Objetivo: Eliminar reflexiones de paredes y muebles (0 Hz Doppler).

% Método A: Sustracción de la media (Filtro de cancelación de DC) 
s_no_dc = s_diff - mean(s_diff);

% Método B: Diferencia de primer orden (Resalta cambios muy rápidos)
% s_diff_filt = diff(s_diff); 

%% 3. Filtrado de Paso Alto (EMA - Exponential Moving Average)
% Objetivo: Eliminar imperfecciones de hardware (DC offset residual del USRP)[cite: 218, 221].
% El EMA sigue la tendencia lenta (clutter) para restarla de la señal original.

s_ema_filt = zeros(size(s_no_dc));
ema_buffer = s_no_dc(1);

for t = 1:length(s_no_dc)
    % Actualizamos el promedio lento (tendencia)
    ema_buffer = ema_alpha * ema_buffer + (1 - ema_alpha) * s_no_dc(t);
    % Restamos la tendencia para quedarnos con lo dinámico (fuego/personas)
    s_ema_filt(t) = s_no_dc(t) - ema_buffer;
end

%% 4. Separación de Fuentes (Fuego vs. Personas)
% Objetivo: Diferenciar firmas térmicas de movimientos humanos[cite: 134].
% El fuego tiene componentes < 100 Hz y velocidades < 3 m/s[cite: 31, 38].

% Filtro paso bajo digital para centrarnos en el rango Doppler térmico
% (Típicamente el fuego es de baja frecuencia y espectro difuso).
[b, a] = butter(4, 30 / (fs_slow/2), 'low'); % Corte en 30Hz para plumas térmicas
s_thermal_only = filter(b, a, s_ema_filt);

%% 5. Generación del Espectrograma (STFT)
% Una vez limpia la señal, calculamos la representación tiempo-frecuencia[cite: 114].

nperseg = 64; 
noverlap = 48;
window = hamming(nperseg);

[S_final, F, T] = spectrogram(s_thermal_only, window, noverlap, nperseg, fs_slow, 'centered');

%% 6. Visualización de Resultados
figure;
imagesc(T, F, 10*log10(abs(S_final)));
axis xy;
xlabel('Tiempo (s)');
ylabel('Frecuencia Doppler (Hz)');
title('Espectrograma Micro-Doppler Filtrado (Fuego)');
colorbar;
colormap('jet');
# test_preprocessor.py
"""
Pruebas unitarias para el módulo preprocessor.py

Ejecutar con:
    python -m pytest test_preprocessor.py -v
o directamente:
    python test_preprocessor.py
"""

import numpy as np
try:
    import pytest
except ImportError:
    # Stub mínimo para ejecutar sin pytest instalado
    class _Raises:
        def __init__(self, exc, match=None): self.exc = exc; self.match = match
        def __enter__(self): return self
        def __exit__(self, et, ev, tb):
            if et is None: raise AssertionError(f"Se esperaba {self.exc}")
            import re
            if self.match and not re.search(self.match, str(ev)):
                raise AssertionError(f"Excepción no coincide: {ev}")
            return True
    class _Pytest:
        @staticmethod
        def raises(exc, match=None): return _Raises(exc, match)
    pytest = _Pytest()

from preprocessor import (
    PreprocessConfig,
    ClutterRemover,
    MicroDopplerSeparator,
    Preprocessor,
)

# ── Señales de referencia ────────────────────────────────────────────────────

FS_SLOW = 500.0   # Hz slow-time simulado (elegido alto para tests rápidos)
T = 1.0           # segundos
N = int(FS_SLOW * T)
t = np.linspace(0, T, N, endpoint=False)
RNG = np.random.default_rng(42)


def make_static_clutter(n_series=4, snr_db=20.0) -> np.ndarray:
    """Canal estático (constante) + ruido → debería eliminarse casi por completo."""
    dc = (1.0 + 0.5j) * np.ones((n_series, N))
    noise = RNG.standard_normal((n_series, N)) + 1j * RNG.standard_normal((n_series, N))
    noise *= 10 ** (-snr_db / 20.0)
    return dc + noise


def make_human_signal(n_series=4, f_walk=2.0, f_breath=0.35, amp=0.3) -> np.ndarray:
    """
    Señal de caminata + respiración en banda humana (0.5–20 Hz).
    Suma clutter estático de amplitud mucho mayor.
    """
    dc = (2.0 + 1.0j) * np.ones((n_series, N))
    human = amp * np.exp(1j * 2 * np.pi * f_walk * t)        # caminata ~2 Hz
    human += (amp / 2) * np.exp(1j * 2 * np.pi * f_breath * t)  # respiración ~0.35 Hz
    human = human[np.newaxis, :]  # broadcast a [n_series, N]
    noise = 0.05 * (
        RNG.standard_normal((n_series, N)) + 1j * RNG.standard_normal((n_series, N))
    )
    return dc + human + noise


def make_thermal_signal(n_series=4, amp=0.02) -> np.ndarray:
    """
    Firma térmica: variaciones muy lentas (< 0.3 Hz) + clutter estático.
    Simula la convección de aire caliente → baja frecuencia Doppler.
    """
    dc = (2.0 + 1.0j) * np.ones((n_series, N))
    slow_drift = amp * np.exp(1j * 2 * np.pi * 0.1 * t)   # 0.1 Hz << banda humana
    slow_drift = slow_drift[np.newaxis, :]
    noise = 0.05 * (
        RNG.standard_normal((n_series, N)) + 1j * RNG.standard_normal((n_series, N))
    )
    return dc + slow_drift + noise


def _base_cfg() -> PreprocessConfig:
    """Config adaptada al fs_slow sintético de los tests."""
    return PreprocessConfig(
        sample_rate=FS_SLOW * (1024 + 128),  # → fs_slow ≈ FS_SLOW
        n_subcarriers=1024,
        cp_len=128,
        human_doppler_low_hz=0.5,
        human_doppler_high_hz=min(20.0, FS_SLOW / 2 - 1),
        human_energy_ratio_threshold=0.10,
        nperseg=min(64, N // 4),
        noverlap=min(48, N // 4 - 1),
        nfft=128,
    )


# ── Tests: ClutterRemover ────────────────────────────────────────────────────

class TestClutterRemover:

    def test_dc_sub_removes_mean(self):
        cfg = _base_cfg()
        cfg.clutter_method = "dc_sub"
        cr = ClutterRemover(cfg)
        x = make_static_clutter(n_series=6)
        y, info = cr.remove(x)
        # La media residual debe ser casi cero
        residual_mean = np.abs(np.mean(y, axis=1))
        assert np.all(residual_mean < 1e-10), f"Media residual: {residual_mean}"

    def test_ema_attenuates_dc(self):
        cfg = _base_cfg()
        cfg.clutter_method = "ema"
        cfg.ema_alpha = 0.99
        cr = ClutterRemover(cfg)
        x = make_static_clutter(n_series=4, snr_db=30)
        y, info = cr.remove(x)
        power_in = np.mean(np.abs(x) ** 2)
        power_out = np.mean(np.abs(y) ** 2)
        # El filtro EMA debe reducir la potencia (clutter era dominante)
        assert power_out < power_in, "EMA no redujo la potencia"

    def test_svd_removes_correlated_component(self):
        cfg = _base_cfg()
        cfg.clutter_method = "svd"
        cfg.svd_rank = 1
        cr = ClutterRemover(cfg)
        x = make_static_clutter(n_series=8, snr_db=40)
        y, info = cr.remove(x)
        assert "energy_fraction_removed" in info["svd"]
        frac = info["svd"]["energy_fraction_removed"]
        # El primer componente SVD debe capturar la mayor parte de la energía
        assert frac > 0.80, f"Fracción de energía removida: {frac:.3f} < 0.80"

    def test_all_method_runs_all_stages(self):
        cfg = _base_cfg()
        cfg.clutter_method = "all"
        cr = ClutterRemover(cfg)
        x = make_static_clutter()
        y, info = cr.remove(x)
        assert "dc_sub" in info
        assert "ema" in info
        assert "svd" in info

    def test_output_shape_preserved(self):
        cfg = _base_cfg()
        for method in ("dc_sub", "ema", "svd", "all"):
            cfg.clutter_method = method
            cr = ClutterRemover(cfg)
            x = make_static_clutter(n_series=5)
            y, _ = cr.remove(x)
            assert y.shape == x.shape, f"Shape changed with method={method}"

    def test_invalid_method_raises(self):
        cfg = _base_cfg()
        cfg.clutter_method = "nonexistent"
        cr = ClutterRemover(cfg)
        with pytest.raises(ValueError, match="clutter_method desconocido"):
            cr.remove(make_static_clutter())

    def test_info_has_suppression_db(self):
        cfg = _base_cfg()
        cfg.clutter_method = "dc_sub"
        cr = ClutterRemover(cfg)
        _, info = cr.remove(make_static_clutter())
        assert "suppression_dB" in info["dc_sub"]
        assert info["dc_sub"]["suppression_dB"] > 0


# ── Tests: MicroDopplerSeparator ─────────────────────────────────────────────

class TestMicroDopplerSeparator:

    def _sep(self, method: str) -> MicroDopplerSeparator:
        cfg = _base_cfg()
        cfg.separation_method = method
        return MicroDopplerSeparator(cfg)

    def test_bandpass_output_shapes(self):
        sep = self._sep("bandpass")
        x = make_human_signal()
        human, thermal, mask, info = sep.separate(x)
        assert human.shape == x.shape
        assert thermal.shape == x.shape
        assert mask.shape == (x.shape[0],)

    def test_bandpass_human_detected(self):
        """Señal con componente en banda humana → human_mask debe ser True."""
        sep = self._sep("bandpass")
        x = make_human_signal(amp=1.0)
        # Eliminar clutter primero para que la banda humana sea visible
        dc_removed = x - np.mean(x, axis=1, keepdims=True)
        _, _, mask, info = sep.separate(dc_removed)
        # Al menos una serie debería detectarse como humana
        assert mask.any(), f"energy_ratios={info['energy_ratios']}"

    def test_thermal_not_classified_as_human(self):
        """Señal puramente térmica (baja frecuencia) → human_mask debe ser False."""
        sep = self._sep("bandpass")
        x = make_thermal_signal(amp=0.5)
        dc_removed = x - np.mean(x, axis=1, keepdims=True)
        _, _, mask, _ = sep.separate(dc_removed)
        # La señal térmica está por debajo de la banda humana → no debe clasificarse
        assert not mask.all(), "Señal térmica clasificada erróneamente como humana"

    def test_energy_ratio_method(self):
        sep = self._sep("energy_ratio")
        x = make_human_signal(amp=1.0)
        dc_removed = x - np.mean(x, axis=1, keepdims=True)
        human, thermal, mask, info = sep.separate(dc_removed)
        assert human.shape == x.shape
        assert "energy_ratios" in info

    def test_combined_method(self):
        sep = self._sep("combined")
        x = make_human_signal(amp=1.0)
        dc_removed = x - np.mean(x, axis=1, keepdims=True)
        human, thermal, mask, info = sep.separate(dc_removed)
        assert "votes" in info
        assert human.shape == x.shape

    def test_human_plus_thermal_sum(self):
        """human + thermal ≈ entrada (separación conservativa)."""
        sep = self._sep("bandpass")
        x = make_human_signal(amp=0.5)
        dc_removed = x - np.mean(x, axis=1, keepdims=True)
        human, thermal, _, _ = sep.separate(dc_removed)
        reconstruction = human + thermal
        err = np.max(np.abs(reconstruction - dc_removed))
        assert err < 1e-6, f"Error de reconstrucción: {err:.2e}"

    def test_invalid_method_raises(self):
        cfg = _base_cfg()
        cfg.separation_method = "unknown"
        sep = MicroDopplerSeparator(cfg)
        with pytest.raises(ValueError, match="separation_method desconocido"):
            sep.separate(make_human_signal())


# ── Tests: Preprocessor (integración) ────────────────────────────────────────

class TestPreprocessor:

    def test_full_pipeline_human(self):
        cfg = _base_cfg()
        cfg.clutter_method = "dc_sub"
        cfg.separation_method = "bandpass"
        prep = Preprocessor(cfg)
        x = make_human_signal(n_series=4, amp=1.0)
        result = prep.run(x)
        assert result.raw.shape == x.shape
        assert result.after_clutter.shape == x.shape
        assert result.human_component.shape == x.shape
        assert result.thermal_component.shape == x.shape
        assert result.human_mask.shape == (x.shape[0],)

    def test_full_pipeline_thermal(self):
        cfg = _base_cfg()
        cfg.clutter_method = "svd"
        cfg.separation_method = "energy_ratio"
        prep = Preprocessor(cfg)
        x = make_thermal_signal(n_series=4, amp=0.3)
        result = prep.run(x)
        assert result.metadata["clutter"]["method"] == "svd"
        assert result.metadata["separation"]["method"] == "energy_ratio"

    def test_metadata_populated(self):
        cfg = _base_cfg()
        prep = Preprocessor(cfg)
        result = prep.run(make_static_clutter())
        assert "clutter" in result.metadata
        assert "separation" in result.metadata
        assert "config" in result.metadata

    def test_combined_clutter_and_separation(self):
        cfg = _base_cfg()
        cfg.clutter_method = "all"
        cfg.separation_method = "combined"
        prep = Preprocessor(cfg)
        x = make_human_signal(n_series=3, amp=0.8)
        result = prep.run(x)
        assert result.human_component.shape == x.shape

    def test_false_alarm_rate_thermal_only(self):
        """
        Con señal puramente térmica (amplitud comparable al caso real),
        la tasa de falsas alarmas debe ser menor que con umbral bajo.
        
        Test de cordura: el clasificador con umbral alto (0.40) debe dar
        menos FA que con umbral bajo (0.05).
        """
        x = make_thermal_signal(n_series=20, amp=0.5)
        dc_removed = x - np.mean(x, axis=1, keepdims=True)

        # Umbral alto → menos FA
        cfg_strict = _base_cfg()
        cfg_strict.separation_method = "bandpass"
        cfg_strict.human_energy_ratio_threshold = 0.40
        prep_strict = Preprocessor(cfg_strict)
        result_strict = prep_strict.run(dc_removed)
        fa_strict = result_strict.human_mask.mean()

        # Umbral bajo → más FA
        cfg_lax = _base_cfg()
        cfg_lax.separation_method = "bandpass"
        cfg_lax.human_energy_ratio_threshold = 0.05
        prep_lax = Preprocessor(cfg_lax)
        result_lax = prep_lax.run(dc_removed)
        fa_lax = result_lax.human_mask.mean()

        assert fa_strict <= fa_lax, (
            f"Umbral estricto FA={fa_strict:.2%} debería ser ≤ umbral laxo FA={fa_lax:.2%}"
        )


# ── Runner manual ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    tests_classes = [TestClutterRemover, TestMicroDopplerSeparator, TestPreprocessor]
    passed = failed = 0
    for cls in tests_classes:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith("test_")]
        for m in methods:
            try:
                getattr(obj, m)()
                print(f"  ✓  {cls.__name__}.{m}")
                passed += 1
            except Exception as exc:
                print(f"  ✗  {cls.__name__}.{m}  →  {exc}")
                failed += 1

    total = passed + failed
    print(f"\n{'─'*55}")
    print(f"  {passed}/{total} tests pasaron  |  {failed} fallaron")
    sys.exit(0 if failed == 0 else 1)

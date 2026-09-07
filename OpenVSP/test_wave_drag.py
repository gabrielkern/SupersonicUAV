"""
Standalone unit tests for suave_corrected_wave_drag() in wave_drag.py.

No pytest dependency in this repo -- uses stdlib unittest. Run with:
    cd supersonicUAV/OpenVSP && python3 -m unittest test_wave_drag

Injects a fake variable_plane_wave_drag module into sys.modules so these
tests run without a real OpenVSP installation, mirroring the lazy import
suave_corrected_wave_drag() itself does.
"""
import sys
import types
import unittest
import numpy as np

FAKE_CD_WAVE = 0.05


def _install_fake_variable_plane_wave_drag():
    fake_module = types.ModuleType('variable_plane_wave_drag')

    def fake_main(config, mach, filename=None, num_slices=20, num_rots=10):
        return FAKE_CD_WAVE

    fake_module.main = fake_main
    sys.modules['variable_plane_wave_drag'] = fake_module


class SuaveCorrectedWaveDragTests(unittest.TestCase):

    def setUp(self):
        _install_fake_variable_plane_wave_drag()
        import wave_drag
        self.wave_drag = wave_drag

    def tearDown(self):
        sys.modules.pop('variable_plane_wave_drag', None)

    def _base_config(self, mach, effective_sweep_deg=55.0):
        return {
            'mach_start': mach,
            'wing_area': 3.201,
            'model_unit': 'in',
            'effective_sweep': effective_sweep_deg,
        }

    def test_accepts_plain_list_of_CL_at_subsonic_mach(self):
        """aero_results['CL'] is a plain Python list in production; must not raise."""
        config = self._base_config(mach=0.5)
        CL = [0.1, 0.2, 0.3]

        result = self.wave_drag.suave_corrected_wave_drag(config, CL, vspfile='dummy.vsp3')

        self.assertEqual(len(result), len(CL))
        # mach <= begin_drag_rise_mach (default 0.87) -> subsonic branch, zero drag
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_accepts_scalar_CL_matching_main_demo_usage(self):
        """__main__'s suave_sweep() demo passes a bare float for CL."""
        config = self._base_config(mach=0.5)

        result = self.wave_drag.suave_corrected_wave_drag(config, 0.15, vspfile='dummy.vsp3')

        self.assertEqual(len(result), 1)

    def test_does_not_raise_keyerror_on_effective_sweep(self):
        config = self._base_config(mach=0.5, effective_sweep_deg=55.0)
        CL = [0.2]

        # Should not raise KeyError: 'effective_sweep'
        self.wave_drag.suave_corrected_wave_drag(config, CL, vspfile='dummy.vsp3')

    def test_effective_sweep_degrees_are_converted_to_radians(self):
        """M_DD must use cos(deg2rad(eff_sweep)), not cos(eff_sweep) directly."""
        eff_sweep_deg = 55.0
        CL = [0.2]
        # Mach comfortably inside the lerp region so the CL-dependent M_DD
        # actually affects the returned value via the CubicSpline anchor.
        config = self._base_config(mach=1.0, effective_sweep_deg=eff_sweep_deg)

        result = self.wave_drag.suave_corrected_wave_drag(config, CL, vspfile='dummy.vsp3')

        # Hand-computed expected M_CR using the correct (radians) formula.
        kappa = 0.1
        dCD_dM, m, z = 0.1, 4, 20
        begin_drag_rise_mach = 0.87
        cos_term_correct = np.cos(np.deg2rad(eff_sweep_deg)) ** 3
        cos_term_wrong = np.cos(eff_sweep_deg) ** 3
        self.assertNotAlmostEqual(cos_term_correct, cos_term_wrong)

        M_DD_correct = begin_drag_rise_mach - (kappa * CL[0] / cos_term_correct)
        M_CR_correct = M_DD_correct - (dCD_dM / m / z) ** (1 / (m - 1))

        peak_mach, end_drag_rise_mach = 1.04, 1.2
        cd_peak = FAKE_CD_WAVE * 1.25
        from scipy.interpolate import CubicSpline
        expected = CubicSpline(
            [M_CR_correct, peak_mach, end_drag_rise_mach],
            [0.0, cd_peak, FAKE_CD_WAVE],
        )(1.0)

        np.testing.assert_allclose(result[0], expected, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()

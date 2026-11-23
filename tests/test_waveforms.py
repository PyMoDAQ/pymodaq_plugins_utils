"""The waveform factory: samples compared with values worked out by hand, the registry and the settings"""
import numpy as np
import pytest
from pymodaq_gui.parameter import Parameter

from pymodaq_data import Q_
from pymodaq_plugins_utils.hardware import waveforms as wf
from pymodaq_plugins_utils.hardware.waveforms import WaveformFactory, Waveform, waveform_field, parse_breakpoints


def create_waveform(name, **parameters):
    return WaveformFactory().get(name, **parameters)


def test_registry_has_the_standard_forms():
    assert WaveformFactory.names()[:8] == ['dc', 'sine', 'square', 'triangle', 'sawtooth', 'ramp', 'pulse',
                                           'piecewise']
    assert 'array' in WaveformFactory.names()
    assert 'sine' in WaveformFactory().keys  # the usual interface of the PyMoDAQ factories
    assert WaveformFactory.get_class('sine') is wf.Sine and wf.Sine.name == 'sine'
    with pytest.raises(KeyError, match='Unknown waveform'):
        create_waveform('nope')


def test_a_plugin_can_register_its_own_waveform():
    import dataclasses

    @WaveformFactory.register('test_constant2')
    @dataclasses.dataclass
    class Constant2(Waveform):
        value: float = waveform_field(2., 'V', 'Value')

        def _realize(self, sample_rate):
            return np.full(4, self.value), None

    try:
        assert Constant2.name == 'test_constant2'
        assert 'test_constant2' in WaveformFactory.names()
        assert create_waveform('test_constant2', value=Q_(3000, 'mV')).realize(1e3).samples.tolist() == [3.] * 4
    finally:
        wf.WaveformFactory._builders['WaveformFactory'].pop('test_constant2')


def test_dc():
    realized = create_waveform('dc', level=1.5, duration=5e-3).realize(1000.)
    assert realized.samples.tolist() == [1.5] * 5
    assert not realized.periodic and realized.frequency is None


def test_sine_one_period():
    # 800 Hz sampling, 100 Hz: 8 samples per period, the quarter of the period is the 3rd sample
    realized = create_waveform('sine', amplitude=2., frequency=100., offset=1.).realize(800.)
    assert realized.periodic and len(realized.samples) == 8
    assert realized.samples[0] == pytest.approx(1.)
    assert realized.samples[2] == pytest.approx(3.)
    assert realized.samples[4] == pytest.approx(1.)
    assert realized.samples[6] == pytest.approx(-1.)
    assert realized.frequency == pytest.approx(100.) and realized.frequency_error == pytest.approx(0.)


def test_sine_phase():
    realized = create_waveform('sine', amplitude=1., frequency=100., phase=90.).realize(800.)
    assert realized.samples[0] == pytest.approx(1.)  # starts at the top


def test_frequency_is_rounded_to_a_whole_number_of_samples():
    realized = create_waveform('sine', frequency=30.).realize(1000.)  # 33.3 samples per period
    assert len(realized.samples) == 33
    assert realized.frequency == pytest.approx(1000. / 33)
    assert realized.frequency_error == pytest.approx(1000. / 33 - 30.)
    assert realized.duration == pytest.approx(0.033)


def test_frequency_above_nyquist_is_refused():
    with pytest.raises(ValueError, match='half the sample rate'):
        create_waveform('sine', frequency=600.).realize(1000.)


def test_square():
    realized = create_waveform('square', amplitude=2., frequency=100., offset=1., duty_cycle=0.25).realize(800.)
    assert realized.samples.tolist() == [3., 3., -1., -1., -1., -1., -1., -1.]


def test_triangle():
    realized = create_waveform('triangle', amplitude=2., frequency=100.).realize(800.)
    assert realized.samples == pytest.approx([0., 1., 2., 1., 0., -1., -2., -1.])


def test_triangle_symmetry():
    # rises during a quarter of the period: starts at 0 on the rising edge, reaches the top after 1/8 of the period
    realized = create_waveform('triangle', frequency=100., symmetry=0.25).realize(800.)
    assert realized.samples.max() == pytest.approx(1.) and realized.samples.min() == pytest.approx(-1.)
    assert realized.samples[0] == pytest.approx(0.)
    assert int(np.argmax(realized.samples)) == 1


def test_sawtooth():
    realized = create_waveform('sawtooth', amplitude=4., frequency=100.).realize(800.)
    assert realized.samples == pytest.approx([0., 1., 2., 3., -4., -3., -2., -1.])


def test_ramp_and_pulse():
    assert create_waveform('ramp', start=0., stop=1., duration=5e-3).realize(1000.).samples == \
           pytest.approx([0., 0.25, 0.5, 0.75, 1.])
    pulse = create_waveform('pulse', baseline=-1., amplitude=3., delay=2e-3, width=3e-3, duration=8e-3)
    assert pulse.realize(1000.).samples.tolist() == [-1., -1., 2., 2., 2., -1., -1., -1.]


def test_piecewise():
    wave = create_waveform('piecewise', points='0:0, 2e-3:4\n4e-3:4, 5e-3:0')
    assert wave.realize(1000.).samples == pytest.approx([0., 2., 4., 4., 4., 0.])


@pytest.mark.parametrize('text, message', [('0:0', 'two'), ('0:0, 1e-3', 'time:value'),
                                           ('0:0, 0:1', 'increase'), ('a:b, c:d', 'time:value')])
def test_breakpoints_errors(text, message):
    with pytest.raises(ValueError, match=message):
        parse_breakpoints(text)
    with pytest.raises(ValueError, match=message):
        create_waveform('piecewise', points=text)


def test_array_from_values_and_from_files(tmp_path):
    values = np.array([0., 1., 0.5])
    assert create_waveform('array', values=values).realize(100.).samples.tolist() == values.tolist()
    np.save(tmp_path / 'wave.npy', values)
    assert create_waveform('array', path=str(tmp_path / 'wave.npy')).realize(100.).samples.tolist() == values.tolist()
    (tmp_path / 'wave.csv').write_text('0\n1\n0.5\n')
    assert create_waveform('array', path=str(tmp_path / 'wave.csv')).realize(100.).samples.tolist() == values.tolist()
    with pytest.raises(ValueError, match='No file'):
        create_waveform('array').realize(100.)


def test_pint_quantities_are_converted():
    wave = create_waveform('sine', amplitude=Q_(500, 'mV'), frequency=Q_(2, 'kHz'), phase=Q_(0.5, 'turn'))
    assert (wave.amplitude, wave.frequency, wave.phase) == pytest.approx((0.5, 2000., 180.))
    assert create_waveform('sine', frequency=100.).realize(Q_(0.8, 'kHz')).sample_rate == 800.


def test_limits():
    with pytest.raises(ValueError, match='out of its limits'):
        create_waveform('square', duty_cycle=1.5)
    with pytest.raises(ValueError, match='positive'):
        create_waveform('dc').realize(0.)


@pytest.mark.parametrize('name', ['dc', 'sine', 'square', 'triangle', 'sawtooth', 'ramp', 'pulse', 'piecewise',
                                  'array'])
def test_settings_roundtrip(name):
    """The params make a settings group and the waveform is read back from it"""
    cls = wf.WaveformFactory.get_class(name)
    group = Parameter.create(name='waveform', type='group', children=cls.params())
    wave = cls.from_settings(group)
    defaults = {f.name: f.default for f in cls._parameters()}
    assert {f.name: getattr(wave, f.name) for f in cls._parameters()} == defaults


def test_settings_carry_units_and_limits():
    params = {param['name']: param for param in wf.Square.params()}
    assert params['frequency']['suffix'] == 'Hz' and params['frequency']['siPrefix'] is True
    assert params['phase']['suffix'] == 'deg' and params['phase']['siPrefix'] is False
    assert (params['duty_cycle']['min'], params['duty_cycle']['max']) == (0., 1.)
    assert params['duty_cycle']['suffix'] == ''

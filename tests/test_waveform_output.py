"""The waveform output contract: settings groups of the plugins, read back from the settings of their control module"""
import numpy as np
import pytest
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_utils.hardware.waveform_output import (WaveformOutput, OutputMode, waveform_output_params,
                                                            fill_waveform_output_settings, WAVEFORM_OUTPUT_VERSION)


class FakeModule:
    """What an application has: the DAQ_Move control module, with the settings tree of its plugin"""
    _hw_settings_name = 'actuator_settings'

    def __init__(self, children=None):
        children = waveform_output_params() if children is None else children
        self.settings = Parameter.create(name='settings', type='group', children=[
            {'name': self._hw_settings_name, 'type': 'group', 'children': children}])

    @property
    def plugin_settings(self):
        return self.settings.child(self._hw_settings_name)


def fill(module, **kwargs):
    values = dict(n_channels=2, min_rate=1., max_rate=1e6, max_samples=1000, min_voltage=-5., max_voltage=5.,
                  modes=[OutputMode.FINITE, OutputMode.CONTINUOUS], triggers=['PFI0', 'PFI1'], live_update=True)
    values.update(kwargs)
    fill_waveform_output_settings(module.plugin_settings, **values)


def test_not_initialized_yet():
    assert WaveformOutput.from_module(FakeModule()) is None


def test_no_such_groups():
    assert WaveformOutput.from_module(FakeModule(children=[])) is None


def test_roundtrip():
    module = FakeModule()
    fill(module)
    module.plugin_settings.child('waveform_output', 'sample_rate').setValue(5e4)
    module.plugin_settings.child('waveform_output', 'mode').setValue('continuous')
    assert WaveformOutput.from_module(module) == WaveformOutput(
        n_channels=2, min_rate=1., max_rate=1e6, max_samples=1000, min_voltage=-5., max_voltage=5.,
        modes=[OutputMode.FINITE, OutputMode.CONTINUOUS], triggers=['PFI0', 'PFI1'], live_update=True,
        sample_rate=5e4, mode=OutputMode.CONTINUOUS, repetitions=1)


def test_the_modes_the_user_can_choose_are_the_ones_of_the_instrument():
    module = FakeModule()
    fill(module, modes=[OutputMode.FINITE])
    assert module.plugin_settings.child('waveform_output', 'mode').opts['limits'] == ['finite']


def test_other_version_is_refused(caplog):
    module = FakeModule()
    fill(module)
    module.plugin_settings.child('waveform_capabilities', 'version').setValue(WAVEFORM_OUTPUT_VERSION + 1)
    assert WaveformOutput.from_module(module) is None


def make_output(**kwargs):
    module = FakeModule()
    fill(module, **kwargs)
    module.plugin_settings.child('waveform_output', 'sample_rate').setValue(1e3)
    return WaveformOutput.from_module(module)


def test_check_accepts_what_the_instrument_can_do():
    make_output().check([np.zeros(100), np.ones(100)])


@pytest.mark.parametrize('samples, kwargs, message', [
    ([np.zeros(100)], {}, '1 waveforms for 2 channels'),
    ([np.zeros(100), np.zeros(50)], {}, 'same number of samples'),
    ([np.zeros(2000), np.zeros(2000)], {}, 'at most 1000'),
    ([np.zeros(10), np.zeros(10)], dict(sample_rate=2e6), 'Sample rate'),
    ([np.zeros(10), np.zeros(10)], dict(mode='repeat'), 'mode repeat is not available'),
    ([np.zeros(10), np.full(10, 6.)], {}, 'goes from 0 V to 6 V.*-5 V to 5 V'),
    ([np.zeros(10), np.full(10, -6.)], {}, 'goes from -6 V to 0 V'),
])
def test_check_says_what_is_wrong(samples, kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_output().check(samples, **kwargs)


def test_no_limit_of_samples_when_zero():
    make_output(max_samples=0).check([np.zeros(10 ** 6), np.zeros(10 ** 6)])

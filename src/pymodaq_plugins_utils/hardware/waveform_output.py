"""Common interface of the instrument plugins generating waveforms (analog outputs, arbitrary generators)

A waveform output plugin is a DAQ_Move actuator whose position is a :class:`~pymodaq.utils.data.DataActuator` holding
one array of samples (volts) per channel, all of the same length. It adds the settings groups returned by
:func:`waveform_output_params` to its parameters and fills the capabilities once the instrument is initialized
(:func:`fill_waveform_output_settings`). The applications (as the waveform extension) do not call the plugin, which
lives in another thread: they read these groups from the settings of its control module
(:meth:`WaveformOutput.from_module`), then check what they are going to send (:meth:`WaveformOutput.check`) and send
the samples as the position of the actuator.

Two groups: ``waveform_capabilities`` (read only, what the instrument can do) and ``waveform_output`` (what the user
chooses: sample rate, mode, repetitions; the plugin applies them when it receives samples). The names of the groups and
of their children, and :data:`WAVEFORM_OUTPUT_VERSION`, are the contract. Evolution rule: only add optional children,
never rename or remove one; a breaking change increments :data:`WAVEFORM_OUTPUT_VERSION`.
"""
import dataclasses
from typing import Optional, Sequence

import numpy as np

from pymodaq_utils.enums import StrEnum
from pymodaq_utils.logger import set_logger, get_module_name

logger = set_logger(get_module_name(__file__))

WAVEFORM_OUTPUT_VERSION = 1
CAPABILITIES_NAME = 'waveform_capabilities'
OUTPUT_NAME = 'waveform_output'


class OutputMode(StrEnum):
    FINITE = 'finite'          # the samples are generated once
    REPEAT = 'repeat'          # the samples are generated ``repetitions`` times
    CONTINUOUS = 'continuous'  # the samples are generated again and again until the actuator is stopped


def waveform_capabilities_params() -> dict:
    """Group describing the instrument, read only. The lists (modes, triggers) are comma separated texts"""
    return {'title': 'Waveform capabilities:', 'name': CAPABILITIES_NAME, 'type': 'group', 'children': [
        {'title': 'Version:', 'name': 'version', 'type': 'int', 'value': WAVEFORM_OUTPUT_VERSION, 'readonly': True,
         'visible': False},
        {'title': 'Channels:', 'name': 'n_channels', 'type': 'int', 'value': 0, 'readonly': True},
        {'title': 'Min sample rate:', 'name': 'min_rate', 'type': 'float', 'value': 0., 'suffix': 'Hz',
         'siPrefix': True, 'readonly': True},
        {'title': 'Max sample rate:', 'name': 'max_rate', 'type': 'float', 'value': 0., 'suffix': 'Hz',
         'siPrefix': True, 'readonly': True},
        {'title': 'Max samples:', 'name': 'max_samples', 'type': 'int', 'value': 0, 'readonly': True,
         'tip': 'Per channel, 0 for no limit'},
        {'title': 'Min voltage:', 'name': 'min_voltage', 'type': 'float', 'value': -10., 'suffix': 'V',
         'readonly': True},
        {'title': 'Max voltage:', 'name': 'max_voltage', 'type': 'float', 'value': 10., 'suffix': 'V',
         'readonly': True},
        {'title': 'Modes:', 'name': 'modes', 'type': 'str', 'value': OutputMode.FINITE.value, 'readonly': True},
        {'title': 'Triggers:', 'name': 'triggers', 'type': 'str', 'value': '', 'readonly': True,
         'tip': 'Names of the trigger sources the instrument accepts, set in the settings of the plugin'},
        {'title': 'Live update:', 'name': 'live_update', 'type': 'bool', 'value': False, 'readonly': True,
         'tip': 'True if new samples replace the ones being generated without a break of the signal'},
    ]}


def waveform_output_settings_params() -> dict:
    """Group with the choices of the user"""
    return {'title': 'Waveform output:', 'name': OUTPUT_NAME, 'type': 'group', 'children': [
        {'title': 'Version:', 'name': 'version', 'type': 'int', 'value': WAVEFORM_OUTPUT_VERSION, 'readonly': True,
         'visible': False},
        {'title': 'Sample rate:', 'name': 'sample_rate', 'type': 'float', 'value': 1e4, 'min': 0., 'suffix': 'Hz',
         'siPrefix': True},
        {'title': 'Mode:', 'name': 'mode', 'type': 'list', 'limits': [m.value for m in OutputMode],
         'value': OutputMode.FINITE.value},
        {'title': 'Repetitions:', 'name': 'repetitions', 'type': 'int', 'value': 1, 'min': 1,
         'tip': 'Used in the repeat mode'},
    ]}


def waveform_output_params() -> list[dict]:
    """The groups to add to the parameters of a waveform output plugin"""
    return [waveform_capabilities_params(), waveform_output_settings_params()]


def fill_waveform_output_settings(settings, n_channels: int, min_rate: float, max_rate: float, max_samples: int = 0,
                                  min_voltage: float = -10., max_voltage: float = 10.,
                                  modes: Sequence[OutputMode] = (OutputMode.FINITE,), triggers: Sequence[str] = (),
                                  live_update: bool = False) -> None:
    """Fill the capabilities of a plugin in its ``settings`` (to be called in ``ini_stage``)

    The modes of the user choice are limited to the ones the instrument has.
    """
    modes = [OutputMode(mode) for mode in modes]
    capabilities = settings.child(CAPABILITIES_NAME)
    for key, value in (('n_channels', n_channels), ('min_rate', min_rate), ('max_rate', max_rate),
                       ('max_samples', max_samples), ('min_voltage', min_voltage), ('max_voltage', max_voltage),
                       ('modes', ','.join(mode.value for mode in modes)), ('triggers', ','.join(triggers)),
                       ('live_update', live_update)):
        capabilities.child(key).setValue(value)
    settings.child(OUTPUT_NAME, 'mode').setLimits([mode.value for mode in modes])


@dataclasses.dataclass
class WaveformOutput:
    """What an application knows of a waveform output plugin: its capabilities and the choices of the user"""
    n_channels: int
    min_rate: float
    max_rate: float
    max_samples: int
    min_voltage: float
    max_voltage: float
    modes: list[OutputMode]
    triggers: list[str]
    live_update: bool
    sample_rate: float
    mode: OutputMode
    repetitions: int

    @classmethod
    def from_settings(cls, capabilities, output) -> 'WaveformOutput':
        """Build it from the two groups (Parameter objects) of a waveform output plugin"""
        return cls(n_channels=int(capabilities['n_channels']), min_rate=float(capabilities['min_rate']),
                   max_rate=float(capabilities['max_rate']), max_samples=int(capabilities['max_samples']),
                   min_voltage=float(capabilities['min_voltage']), max_voltage=float(capabilities['max_voltage']),
                   modes=[OutputMode(mode) for mode in capabilities['modes'].split(',') if mode],
                   triggers=[trigger for trigger in capabilities['triggers'].split(',') if trigger],
                   live_update=bool(capabilities['live_update']), sample_rate=float(output['sample_rate']),
                   mode=OutputMode(output['mode']), repetitions=int(output['repetitions']))

    @classmethod
    def from_module(cls, module) -> Optional['WaveformOutput']:
        """Read it from a DAQ_Move control module

        None if its plugin has no such groups, is not initialized yet (no channel) or follows another version of the
        contract (an error is logged in that case).
        """
        try:
            settings = module.settings.child(module._hw_settings_name)
            capabilities, output = settings.child(CAPABILITIES_NAME), settings.child(OUTPUT_NAME)
        except (KeyError, AttributeError):
            return None
        versions = [group['version'] if 'version' in [c.name() for c in group.children()] else None
                    for group in (capabilities, output)]
        if versions != [WAVEFORM_OUTPUT_VERSION] * 2:
            logger.error(f'The waveform output groups of {module} have versions {versions}, version '
                         f'{WAVEFORM_OUTPUT_VERSION} is expected: update the plugin or pymodaq_plugins_utils')
            return None
        waveform_output = cls.from_settings(capabilities, output)
        return waveform_output if waveform_output.n_channels > 0 else None

    def check(self, samples: Sequence[np.ndarray], sample_rate: Optional[float] = None,
              mode: Optional[OutputMode] = None) -> None:
        """Raise a ValueError saying what the instrument cannot do with these samples (one array per channel)

        ``sample_rate`` and ``mode`` are the ones of the user choices if not given.
        """
        sample_rate = self.sample_rate if sample_rate is None else sample_rate
        mode = self.mode if mode is None else OutputMode(mode)
        if len(samples) != self.n_channels:
            raise ValueError(f'{len(samples)} waveforms for {self.n_channels} channels')
        if len({len(channel) for channel in samples}) != 1:
            raise ValueError('The waveforms of the channels should have the same number of samples')
        n_samples = len(samples[0])
        if self.max_samples and n_samples > self.max_samples:
            raise ValueError(f'{n_samples} samples, the instrument holds at most {self.max_samples} per channel')
        if not self.min_rate <= sample_rate <= self.max_rate:
            raise ValueError(f'Sample rate {sample_rate} Hz out of the range of the instrument: '
                             f'{self.min_rate:g} to {self.max_rate:g} Hz')
        if mode not in self.modes:
            raise ValueError(f'The mode {mode.value} is not available, the instrument has: '
                             f'{", ".join(mode.value for mode in self.modes)}')
        low, high = min(float(np.min(channel)) for channel in samples), max(float(np.max(channel))
                                                                           for channel in samples)
        if low < self.min_voltage or high > self.max_voltage:
            raise ValueError(f'The waveform goes from {low:.3g} V to {high:.3g} V, the instrument gives '
                             f'{self.min_voltage:g} V to {self.max_voltage:g} V')

"""Sine"""
import dataclasses
from typing import ClassVar

import numpy as np

from pymodaq_plugins_utils.hardware.waveforms.waveform_factory import (
    Waveform, WaveformFactory, waveform_field, _period, _phase)


@WaveformFactory.register('sine')
@dataclasses.dataclass
class Sine(Waveform):
    title: ClassVar[str] = 'Sine'
    periodic: ClassVar[bool] = True
    amplitude: float = waveform_field(1., 'V', 'Amplitude', tip='Peak value, half of the peak to peak')
    frequency: float = waveform_field(1e3, 'Hz', 'Frequency', min=0.)
    offset: float = waveform_field(0., 'V', 'Offset')
    phase: float = waveform_field(0., 'deg', 'Phase')

    def _realize(self, sample_rate):
        n, frequency = _period(self.frequency, sample_rate)
        return self.offset + self.amplitude * np.sin(2 * np.pi * _phase(n, self.phase)), frequency

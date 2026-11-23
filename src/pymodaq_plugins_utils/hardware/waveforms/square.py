"""Square with an adjustable duty cycle"""
import dataclasses
from typing import ClassVar

import numpy as np

from pymodaq_plugins_utils.hardware.waveforms.waveform_factory import (
    Waveform, WaveformFactory, waveform_field, _period, _phase)


@WaveformFactory.register('square')
@dataclasses.dataclass
class Square(Waveform):
    title: ClassVar[str] = 'Square'
    periodic: ClassVar[bool] = True
    amplitude: float = waveform_field(1., 'V', 'Amplitude', tip='Peak value, half of the peak to peak')
    frequency: float = waveform_field(1e3, 'Hz', 'Frequency', min=0.)
    offset: float = waveform_field(0., 'V', 'Offset')
    phase: float = waveform_field(0., 'deg', 'Phase')
    duty_cycle: float = waveform_field(0.5, 'dimensionless', 'Duty cycle', min=0., max=1.,
                                       tip='Fraction of the period spent high')

    def _realize(self, sample_rate):
        n, frequency = _period(self.frequency, sample_rate)
        high = _phase(n, self.phase) < self.duty_cycle
        return self.offset + np.where(high, self.amplitude, -self.amplitude), frequency

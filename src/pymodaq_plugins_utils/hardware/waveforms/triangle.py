"""Triangle with an adjustable symmetry"""
import dataclasses
from typing import ClassVar

import numpy as np

from pymodaq_plugins_utils.hardware.waveforms.waveform_factory import (
    Waveform, WaveformFactory, waveform_field, _period, _phase)


@WaveformFactory.register('triangle')
@dataclasses.dataclass
class Triangle(Waveform):
    title: ClassVar[str] = 'Triangle'
    periodic: ClassVar[bool] = True
    amplitude: float = waveform_field(1., 'V', 'Amplitude', tip='Peak value, half of the peak to peak')
    frequency: float = waveform_field(1e3, 'Hz', 'Frequency', min=0.)
    offset: float = waveform_field(0., 'V', 'Offset')
    phase: float = waveform_field(0., 'deg', 'Phase')
    symmetry: float = waveform_field(0.5, 'dimensionless', 'Symmetry', min=0.01, max=0.99,
                                     tip='Fraction of the period spent rising')

    def _realize(self, sample_rate):
        n, frequency = _period(self.frequency, sample_rate)
        s = self.symmetry
        x = (_phase(n, self.phase) + s / 2) % 1.  # a symmetric triangle crosses zero rising at phase 0
        y = np.where(x < s, -1 + 2 * x / s, 1 - 2 * (x - s) / (1 - s))
        return self.offset + self.amplitude * y, frequency

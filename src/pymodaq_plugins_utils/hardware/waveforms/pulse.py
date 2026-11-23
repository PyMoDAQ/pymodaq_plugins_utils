"""Single pulse"""
import dataclasses
from typing import ClassVar

import numpy as np

from pymodaq_plugins_utils.hardware.waveforms.waveform_factory import (Waveform, WaveformFactory, waveform_field)


@WaveformFactory.register('pulse')
@dataclasses.dataclass
class Pulse(Waveform):
    title: ClassVar[str] = 'Pulse'
    baseline: float = waveform_field(0., 'V', 'Baseline')
    amplitude: float = waveform_field(1., 'V', 'Amplitude', tip='Height of the pulse above the baseline')
    delay: float = waveform_field(1e-4, 's', 'Delay', min=0., tip='From the start to the rising edge')
    width: float = waveform_field(1e-4, 's', 'Width', min=0.)
    duration: float = waveform_field(1e-3, 's', 'Duration', min=0., tip='Total, the signal returns to the baseline')

    def _realize(self, sample_rate):
        t = np.arange(self.n_samples(sample_rate)) / sample_rate
        high = (t >= self.delay) & (t < self.delay + self.width)
        return self.baseline + self.amplitude * high, None

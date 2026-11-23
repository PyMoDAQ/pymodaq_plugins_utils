"""Linear ramp"""
import dataclasses
from typing import ClassVar

import numpy as np

from pymodaq_plugins_utils.hardware.waveforms.waveform_factory import (Waveform, WaveformFactory, waveform_field)


@WaveformFactory.register('ramp')
@dataclasses.dataclass
class Ramp(Waveform):
    title: ClassVar[str] = 'Ramp'
    start: float = waveform_field(0., 'V', 'Start')
    stop: float = waveform_field(1., 'V', 'Stop')
    duration: float = waveform_field(1e-3, 's', 'Duration', min=0.)

    def _realize(self, sample_rate):
        return np.linspace(self.start, self.stop, self.n_samples(sample_rate)), None

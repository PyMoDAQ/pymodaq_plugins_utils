"""Constant level"""
import dataclasses
from typing import ClassVar

import numpy as np

from pymodaq_plugins_utils.hardware.waveforms.waveform_factory import (Waveform, WaveformFactory, waveform_field)


@WaveformFactory.register('dc')
@dataclasses.dataclass
class DC(Waveform):
    title: ClassVar[str] = 'DC'
    level: float = waveform_field(0., 'V', 'Level')
    duration: float = waveform_field(1e-3, 's', 'Duration', min=0.)

    def _realize(self, sample_rate):
        return np.full(self.n_samples(sample_rate), self.level), None

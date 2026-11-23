"""Samples from a file or an array"""
import dataclasses
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np

from pymodaq_plugins_utils.hardware.waveforms.waveform_factory import Waveform, WaveformFactory, waveform_field


@WaveformFactory.register('array')
@dataclasses.dataclass
class Array(Waveform):
    """Samples taken as they are, from a file (``.npy`` or text) or given as ``values``; they are played at the sample
    rate of the output"""
    title: ClassVar[str] = 'Array'
    path: str = waveform_field('', None, 'File', tip='.npy file or text file (csv) with one column of samples in V')
    values: Optional[np.ndarray] = None  # not a parameter: the samples when no file is given

    def _realize(self, sample_rate):
        values = self.values
        if values is None:
            if not self.path:
                raise ValueError('No file and no values for the array waveform')
            path = Path(self.path)
            values = np.load(path) if path.suffix == '.npy' else np.loadtxt(path, delimiter=',')
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError('The samples should be a 1D array of at least two values')
        return values, None

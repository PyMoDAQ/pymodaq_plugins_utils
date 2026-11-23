"""Straight segments between time:value points"""
import dataclasses
import re
from typing import ClassVar

import numpy as np

from pymodaq_plugins_utils.hardware.waveforms.waveform_factory import Waveform, WaveformFactory, waveform_field


_NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'


def parse_breakpoints(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Times (s) and values (V) of ``'0:0, 1e-3:5, 3e-3:5, 4e-3:0'``, entries separated by a comma or a new line

    Raises ValueError if an entry is not ``time:value`` or the times do not increase.
    """
    entries = [entry for entry in re.split(r'[,\n;]+', text) if entry.strip()]
    times, values = [], []
    for entry in entries:
        match = re.fullmatch(rf'\s*({_NUMBER})\s*:\s*({_NUMBER})\s*', entry)
        if match is None:
            raise ValueError(f'"{entry.strip()}" is not a time:value pair')
        times.append(float(match.group(1)))
        values.append(float(match.group(2)))
    if len(times) < 2:
        raise ValueError('At least two time:value pairs are needed')
    if np.any(np.diff(times) <= 0):
        raise ValueError('The times should increase')
    return np.array(times), np.array(values)


@WaveformFactory.register('piecewise')
@dataclasses.dataclass
class Piecewise(Waveform):
    """Straight segments between points given as ``time:value`` pairs (seconds and volts)"""
    title: ClassVar[str] = 'Piecewise linear'
    points: str = waveform_field('0:0, 1e-3:1, 2e-3:0', None, 'Points',
                                 tip='time:value pairs in s and V, separated by commas or new lines. The signal '
                                     'goes in a straight line from one point to the next, from the first to the '
                                     'last time')

    def __post_init__(self):
        super().__post_init__()
        parse_breakpoints(self.points)  # fail at creation

    def _realize(self, sample_rate):
        times, values = parse_breakpoints(self.points)
        t = np.arange(int(round((times[-1] - times[0]) * sample_rate)) + 1) / sample_rate + times[0]
        return np.interp(t, times, values), None

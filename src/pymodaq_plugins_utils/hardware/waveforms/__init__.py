"""Waveforms: the factory (:mod:`.waveform_factory`) and one module per form

Importing this package imports all its submodules, so every form is registered in the
:class:`~.waveform_factory.WaveformFactory`. To add a form to the package, add a module here: nothing else to update.
"""
import importlib
import pkgutil

from pymodaq_plugins_utils.hardware.waveforms.waveform_factory import (  # noqa: F401
    WAVEFORM_VERSION, RealizedWaveform, Waveform, WaveformFactory, waveform_field)

for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f'{__name__}.{_module.name}')

from pymodaq_plugins_utils.hardware.waveforms.array import Array  # noqa: E402,F401
from pymodaq_plugins_utils.hardware.waveforms.dc import DC  # noqa: E402,F401
from pymodaq_plugins_utils.hardware.waveforms.piecewise import Piecewise, parse_breakpoints  # noqa: E402,F401
from pymodaq_plugins_utils.hardware.waveforms.pulse import Pulse  # noqa: E402,F401
from pymodaq_plugins_utils.hardware.waveforms.ramp import Ramp  # noqa: E402,F401
from pymodaq_plugins_utils.hardware.waveforms.sawtooth import Sawtooth  # noqa: E402,F401
from pymodaq_plugins_utils.hardware.waveforms.sine import Sine  # noqa: E402,F401
from pymodaq_plugins_utils.hardware.waveforms.square import Square  # noqa: E402,F401
from pymodaq_plugins_utils.hardware.waveforms.triangle import Triangle  # noqa: E402,F401

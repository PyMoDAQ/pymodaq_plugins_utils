"""Base class of the camera plugins built on a camera of `pylablib <https://pylablib.readthedocs.io>`_.

The generic part (acquisition, ROI, binning, ...) is in :mod:`~pymodaq_plugins_utils.hardware.camera_base`: the
pylablib cameras follow its driver contract as they are. The names below are kept here for the plugins importing them
from this module.
"""
from pymodaq.control_modules.viewer_utility_classes import comon_parameters

from pymodaq_plugins_utils.hardware.camera_base import CameraBase, CameraCallback, Grab, cam_params


class CameraBasePyLabLib(CameraBase):
    """
    Base implementation for Camera using pylablib framework. Works for TSI and uc480 thorlabs camera and rpobaly others
    """
    serial_numbers = []

    serial_params = [{'title': 'Serial number:', 'name': 'serial_number', 'type': 'list', 'limits': serial_numbers}]

    params = comon_parameters + serial_params + cam_params


__all__ = ['CameraBasePyLabLib', 'CameraCallback', 'Grab', 'cam_params']

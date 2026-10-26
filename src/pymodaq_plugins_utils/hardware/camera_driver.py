"""Contract between a camera plugin and its driver.

:class:`~pymodaq_plugins_utils.hardware.camera_base.CameraBase` talks to its ``controller`` only through the methods
listed in :class:`CameraDriverBase`. They are named and behave as in the camera drivers of
`pylablib <https://pylablib.readthedocs.io/en/latest/devices/cameras.html>`_, so those drivers can be used as they are,
without any adapter. To write the driver of a camera that pylablib does not know, inherit from
:class:`CameraDriverBase` and reimplement the methods that raise ``NotImplementedError``.

The class has no dependency (neither Qt nor pylablib): it can be tested without hardware.
"""
from typing import Optional, Tuple

import numpy as np


class CameraDriverBase:
    """Methods a camera driver provides to :class:`~pymodaq_plugins_utils.hardware.camera_base.CameraBase`.

    The ROI is described as in pylablib: ``(hstart, hend, vstart, vend, hbin, vbin)``, the coordinates being pixels of
    the full sensor (``hend`` and ``vend`` excluded) and the binnings integers.

    Images are 2D arrays of shape ``(height, width)``. The driver fills its own buffer while the acquisition runs
    (typically in a thread of the driver or of the vendor library) and :meth:`wait_for_frame` blocks until new frames
    are available there.
    """

    # --- to reimplement ------------------------------------------------------------------------------------------

    def get_detector_size(self) -> Tuple[int, int]:
        """Size of the full sensor in pixels, as ``(width, height)``."""
        raise NotImplementedError

    def get_data_dimensions(self) -> Tuple[int, int]:
        """Shape of one image with the current ROI and binning, as ``(height, width)``."""
        raise NotImplementedError

    def get_roi(self) -> Tuple[int, int, int, int, int, int]:
        """Current ROI as ``(hstart, hend, vstart, vend, hbin, vbin)``."""
        raise NotImplementedError

    def set_roi(self, hstart: int = 0, hend: Optional[int] = None, vstart: int = 0, vend: Optional[int] = None,
                hbin: int = 1, vbin: int = 1):
        """Set the ROI and the binning. ``None`` for ``hend`` or ``vend`` means up to the end of the sensor."""
        raise NotImplementedError

    def set_exposure(self, exposure: float):
        """Set the exposure time, in seconds."""
        raise NotImplementedError

    def setup_acquisition(self):
        """Allocate the buffer for the current ROI. Called after each change of ROI or binning."""
        raise NotImplementedError

    def clear_acquisition(self):
        """Stop the acquisition if it runs and release the buffer."""
        raise NotImplementedError

    def start_acquisition(self, nframes: int = 1):
        """Start filling the buffer with new frames."""
        raise NotImplementedError

    def acquisition_in_progress(self) -> bool:
        """Whether the acquisition is running."""
        raise NotImplementedError

    def wait_for_frame(self, since: str = 'lastread', nframes: int = 1, timeout: float = 20.0):
        """Block until ``nframes`` new frames are available in the buffer.

        ``since`` tells from where the new frames are counted (``'lastread'``, ``'lastwait'``, ``'now'`` or
        ``'start'``). Raise an exception on timeout.
        """
        raise NotImplementedError

    def read_multiple_images(self, rng=None, peek: bool = False, missing_frame: str = 'skip',
                             return_rng: bool = False):
        """Read the frames available in the buffer.

        Return an array of shape ``(nframes, height, width)``, and the ``(first, last)`` indices of the frames
        if ``return_rng`` is ``True``.
        """
        raise NotImplementedError

    def read_newest_image(self, peek: bool = False) -> Optional[np.ndarray]:
        """Read the last acquired frame, an array of shape ``(height, width)``, or ``None`` if there is none."""
        raise NotImplementedError

    def close(self):
        """Release the camera."""
        raise NotImplementedError

    # --- optional, with a harmless default ---------------------------------------------------------------------------

    def get_device_info(self):
        """Object with a ``name`` (or ``model``) attribute, shown in the settings."""
        return None

    def get_roi_limits(self):
        """Limits of the ROI, not used by :class:`CameraBase` itself but called once at initialization."""
        return None

    def set_frame_format(self, fmt: str):
        """Format of the returned frames, ``'array'`` for numpy arrays (the only one used here)."""

    def get_pixel_size(self) -> Optional[Tuple[float, float]]:
        """Size of a sensor pixel in micrometers as ``(x, y)``, or ``None`` if unknown.

        When given, the axes of the 2D viewer are calibrated in micrometers instead of pixels. pylablib cameras
        do not have this method, they keep axes in pixels.
        """
        return None

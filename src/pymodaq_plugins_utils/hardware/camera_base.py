"""Base class of the 2D detector plugins driving a camera.

The plugin talks to its ``controller`` (the camera driver) through the methods listed in
:class:`~pymodaq_plugins_utils.hardware.camera_driver.CameraDriverBase`, which are those of the pylablib cameras.
It provides the acquisition in a thread, the ROI shared with the viewer, the binning, the exposure, the FPS and the
colour conversion; a plugin only has to create the controller in :meth:`CameraBase.ini_detector_custom`.
"""
import dataclasses
from typing import Type, Optional, Tuple

try:
    import cv2
except ImportError:  # only needed for the colour conversion
    cv2 = None

from pymodaq_data import DataToExport
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_utils.utils import ThreadCommand
from pymodaq_gui.parameter import Parameter
from pymodaq_gui.utils.widgets import Status
try:
    from pymodaq_gui.plotting.items.roi import RoiInfo  # pymodaq > 5.1.x
except ImportError:
    from pymodaq_gui.plotting.utils.plot_utils import RoiInfo

from pymodaq.utils.data import DataFromPlugins, Axis
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main

from qtpy import QtWidgets, QtCore
import numpy as np
from time import perf_counter


logger = set_logger(get_module_name(__file__))


# States of the LEDs of the ROI buttons (Status workflow): grey when the button is released, blue while the change is
# applied, green when it is, orange on error. They are set by the plugin, the user only presses or releases the button.
ROI_LED_STATES = [Status.OFF, Status.RUNNING, Status.IDLE, Status.ERROR]

cam_params = [
    {'title': 'Camera name:', 'name': 'camera_name', 'type': 'str', 'value': '', 'readonly': True},
    {'title': 'Color Conversion:', 'name': 'color_conversion', 'type': 'list', 'value': 'None',
     'limits': ['None', 'RGB2GRAY', 'BAYER_BG2RGB', 'BAYER_BG2GRAY'] if cv2 is not None else ['None']},
    {'title': 'ROI', 'name': 'roi', 'type': 'group', 'children': [
        {'title': 'Update ROI from Viewer', 'name': 'update_roi', 'type': 'action_led',
         'value': Status.OFF.value, 'label': 'Update', 'limits': ROI_LED_STATES, 'checkable': True},
        {'title': 'Apply ROI', 'name': 'apply_roi', 'type': 'action_led',
         'value': Status.OFF.value, 'label': 'Apply', 'limits': ROI_LED_STATES, 'checkable': True},
        {'title': 'Clear ROI+Bin', 'name': 'clear_roi', 'type': 'bool_push', 'value': False},
        {'title': 'ROI:', 'name': 'roi_slices', 'type': 'str', 'value': '', 'readonly': True},
        {'title': 'X binning', 'name': 'x_binning', 'type': 'int', 'value': 1},
        {'title': 'Y binning', 'name': 'y_binning', 'type': 'int', 'value': 1},
    ], },
    {'title': 'Image width', 'name': 'hdet', 'type': 'int', 'value': 1, 'readonly': True},
    {'title': 'Image height', 'name': 'vdet', 'type': 'int', 'value': 1, 'readonly': True},
    {'title': 'Timing', 'name': 'timing_opts', 'type': 'group', 'children':
        [{'title': 'Exposure Time (ms)', 'name': 'exposure_time', 'type': 'int', 'value': 1},
         {'title': 'Compute FPS', 'name': 'fps_on', 'type': 'bool', 'value': True},
         {'title': 'FPS', 'name': 'fps', 'type': 'float', 'value': 0.0, 'readonly': True}]
     },
    {'title': 'Buffer', 'name': 'buffer', 'type': 'group', 'children': [
        {'title': 'Size:', 'name': 'size', 'type': 'int', 'value': 10},
        {'title': 'mode:', 'name': 'mode', 'type': 'list', 'value': 'now',
         'limits': ['now', 'lastread', 'lastwait', 'start']},
    ]},
]


@dataclasses.dataclass
class Grab:
    do_acquisition: bool = True
    snap: bool = False
    since: str = 'now'
    nframes: int = 1
    n_average: int = 1


class CameraCallback(QtCore.QObject):
    """Callback object """
    data_sig = QtCore.Signal(np.ndarray)
    error = QtCore.Signal()

    def __init__(self, controller):
        super().__init__()
        # Set the wait function
        self.controller = controller
        self.do_acquisition = True

    def set_do_grab(self, mode: Grab):
        self.do_acquisition = mode.do_acquisition
        if mode.do_acquisition:
            self.wait_for_acquisition(mode)

    def wait_for_acquisition(self, mode: Grab):
        while self.do_acquisition:
            try:
                ind_average = 0
                while ind_average < mode.n_average:
                    ind_frames = 0
                    while ind_frames < mode.nframes:
                        self.controller.wait_for_frame(since='now')
                        new_frames, rng = self.controller.read_multiple_images(missing_frame='skip', return_rng=True)
                        if rng[1] == rng[0]:  # no new frame: the acquisition was stopped while waiting for it
                            QtWidgets.QApplication.processEvents()  # let the signal that stops this loop be handled
                            if not self.do_acquisition or not self.controller.acquisition_in_progress():
                                return
                            continue

                        # in case hardware has a random buffer size (ex: Andor CCD cameras)
                        if new_frames.shape[0] != mode.nframes:
                            new_frames = np.expand_dims(new_frames[-mode.nframes:], axis=0)

                        if ind_average == 0 and ind_frames == 0:
                            shape = list(new_frames.shape[1:])
                            shape = [mode.n_average, mode.nframes] + shape
                            frames = np.zeros(shape, dtype=new_frames.dtype)
                        nacq = rng[1] - rng[0]
                        frames[ind_average, ind_frames:nacq, ...] = new_frames
                        ind_frames += nacq
                    ind_average += 1
                self.data_sig.emit(frames)
                QtCore.QThread.msleep(10)
            except Exception as e:
                logger.exception(str(e))
                self.error.emit()
                break
            QtWidgets.QApplication.processEvents()
            if not self.do_acquisition or mode.snap:
                break



class CameraBase(DAQ_Viewer_base):
    """Base implementation of a camera plugin

    The camera driver (``self.controller``, created in :meth:`ini_detector_custom`) follows
    :class:`~pymodaq_plugins_utils.hardware.camera_driver.CameraDriverBase`. If its ``get_pixel_size`` method returns
    a size, the axes of the viewer are in micrometers, otherwise in pixels.

    ``live_mode_available`` and ``hardware_averaging`` are independent (see the *Live Mode* section of the PyMoDAQ
    documentation): in live mode the viewer calls ``grab_data`` once and the camera thread emits the frames until
    ``stop`` is called, and ``hardware_averaging`` tells who averages the frames.
    """
    params = comon_parameters + cam_params

    callback_signal = QtCore.Signal(Grab)
    live_mode_available = True
    # The camera thread already averages Naverage frames (Grab.n_average): once the pymodaq releases that contain the fix
    # of the live mode (PyMoDAQ #1421) are the minimum required, this should be True, otherwise the viewer averages
    # the averaged frames a second time.
    hardware_averaging = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the activation of the buttons is sent by the DAQ_Viewer to the thread of the plugin (pymodaq >= 5.3.9)
        for name, slot in (('update_roi', self._on_update_roi), ('apply_roi', self._on_apply_roi)):
            try:
                self.settings.child('roi', name).sigActivated.connect(slot)
            except KeyError:  # a subclass without these parameters
                pass

    def ini_attributes(self):
        self.controller = None
        self.callback_thread: QtCore.QThread = None
        self.is_live: bool = False
        self.Naverage: int = 1

        self.x_axis: Axis = None
        self.y_axis: Axis = None

        self.roi_select_info: RoiInfo = None
        self.roi_sensor_slices: Optional[Tuple[slice, slice]] = None  # (rows, columns), in pixels of the full sensor

        self.last_tick = 0.0  # time counter used to compute FPS
        self.fps = 0.0

        self.data_shape: str = ''


    @property
    def pixel_size(self) -> Optional[Tuple[float, float]]:
        """Size (x, y) of a sensor pixel in micrometers if the driver knows it, else None"""
        get_pixel_size = getattr(self.controller, 'get_pixel_size', None)
        return get_pixel_size() if get_pixel_size is not None else None

    def _set_roi_led(self, name: str, status: Status):
        """Show the state of the 'update_roi' or 'apply_roi' LED"""
        self.settings.child('roi', name).setValue(status)

    def _roi_led_is(self, name: str, status: Status) -> bool:
        return self.settings['roi', name] == status

    def _fail_roi_button(self, name: str, message: str):
        """The change asked by the button could not be done: orange LED, the button is released"""
        self._set_roi_led(name, Status.ERROR)
        self.settings.child('roi', name).setOpts(checked=False)
        self.emit_status(ThreadCommand('Update_Status', [message, 'log']))

    def _on_update_roi(self, param):
        """The user pressed or released 'Update ROI from Viewer'

        Pressed: take the ROI now in the viewer (blue LED while it is taken, green once it is, orange if there is
        none to take) and follow the ROI of the viewer. Released: stop following it (grey LED).
        """
        if not param.opts.get('checked', False):
            self._set_roi_led('update_roi', Status.OFF)
            return
        if self.roi_select_info is None:
            self._fail_roi_button('update_roi', 'No ROI selected in the viewer')
            return
        self._set_roi_led('update_roi', Status.RUNNING)
        self.update_roi_from_viewer()
        self._set_roi_led('update_roi', Status.IDLE)
        if self._roi_led_is('apply_roi', Status.IDLE):
            self.apply_roi()

    def _on_apply_roi(self, param):
        """The user pressed or released 'Apply ROI'

        Pressed: apply the ROI selected in the viewer (the one last taken, or the one now in the viewer if none was
        taken), and apply it again when it changes. Released: the camera goes back to the full sensor (grey LED).
        """
        if not param.opts.get('checked', False):
            self._set_roi_led('apply_roi', Status.RUNNING)
            self.release_roi()
            self._set_roi_led('apply_roi', Status.OFF)
            return
        if self.roi_sensor_slices is None and self.roi_select_info is not None:
            self.update_roi_from_viewer()
        self.apply_roi()

    def release_roi(self):
        """Full sensor, with the binning, the ROI selected in the viewer is kept"""
        wdet, hdet = self.controller.get_detector_size()
        self.update_rois((0, wdet, 0, hdet,
                          self.settings['roi', 'x_binning'], self.settings['roi', 'y_binning']))

    def update_roi_from_viewer(self):
        """Compute the ROI of the sensor, in pixels, from the one last selected in the viewer"""
        # the slices are in the values of the axes of the viewer (pixels of the sensor, or um if the pixel size is known).
        # They are kept as floats: a truncation to integers would be wrong in um.
        rows, columns = self.roi_select_info.to_slices(as_integer=False)
        x_size, y_size = self.pixel_size if self.pixel_size is not None else (1, 1)
        rows = slice(int(round(rows.start / y_size)), int(round(rows.stop / y_size)))
        columns = slice(int(round(columns.start / x_size)), int(round(columns.stop / x_size)))
        self.roi_sensor_slices = (rows, columns)
        self.settings['roi', 'roi_slices'] = str(list(self.roi_sensor_slices))

    def roi_select(self, roi_info: RoiInfo, ind_viewer: int = 0):
        """ Automatically called when a user use the RoiSelect ROi from a 2D viewer"""
        self.roi_select_info = roi_info
        self.roi_select_viewer_index = ind_viewer

        if self._roi_led_is('update_roi', Status.IDLE):  # the ROI of the viewer is followed
            self.update_roi_from_viewer()
            if self._roi_led_is('apply_roi', Status.IDLE):
                self.apply_roi()

    def apply_roi(self):
        """Set the ROI of the camera from the one last selected in the viewer

        The 'apply_roi' LED is blue while the ROI is applied, then green, or orange if it cannot be applied.
        """
        if self.roi_sensor_slices is None:
            self._fail_roi_button('apply_roi', 'No ROI selected in the viewer')
            return
        width, height = self.controller.get_detector_size()
        rows, columns = self.roi_sensor_slices
        hstart, hend = max(columns.start, 0), min(columns.stop, width)
        vstart, vend = max(rows.start, 0), min(rows.stop, height)
        if hend <= hstart or vend <= vstart:
            self._fail_roi_button('apply_roi', 'The ROI is outside of the sensor')
            return
        self._set_roi_led('apply_roi', Status.RUNNING)
        try:
            self.update_rois((hstart, hend, vstart, vend,
                              self.settings['roi', 'x_binning'], self.settings['roi', 'y_binning']))
        except Exception as exc:
            logger.exception(exc)
            self._fail_roi_button('apply_roi', f'The ROI could not be applied: {exc}')
            return
        self._set_roi_led('apply_roi', Status.IDLE)

    def compute_axes(self):
        """Axes of the viewer from the current ROI and binning of the camera"""
        (hstart, hend, vstart, vend, hbin, vbin) = self.controller.get_roi()
        height, width = self.controller.get_data_dimensions()
        self.settings.child('roi', 'roi_slices').setValue(str([slice(vstart, vend, vbin), slice(hstart, hend, hbin)]))
        self.settings.child('hdet').setValue(int(width))
        self.settings.child('vdet').setValue(int(height))

        pixel_size = self.pixel_size
        x_size, y_size = pixel_size if pixel_size is not None else (1, 1)
        units = 'µm' if pixel_size is not None else ''
        self.x_axis = Axis('x_axis', units=units, offset=hstart * x_size, scaling=hbin * x_size, size=int(width),
                           index=1)
        self.y_axis = Axis('y_axis', units=units, offset=vstart * y_size, scaling=vbin * y_size, size=int(height),
                           index=0)

    def clear_roi(self):
        """Full sensor, no binning"""
        wdet, hdet = self.controller.get_detector_size()
        self.settings.child('roi', 'x_binning').setValue(1)
        self.settings.child('roi', 'y_binning').setValue(1)

        self.update_rois((0, wdet, 0, hdet, 1, 1))
        self.roi_sensor_slices = None
        for name in ('update_roi', 'apply_roi'):
            self._set_roi_led(name, Status.OFF)
            self.settings.child('roi', name).setOpts(checked=False)

    def update_rois(self, new_roi: Tuple[int, int, int, int, int, int]):
        """Set the ROI and binning of the camera if they changed

        Parameters
        ----------
        new_roi: tuple of int
            As in pylablib: (hstart, hend, vstart, vend, hbin, vbin)
        """
        new_roi = tuple(int(value) for value in new_roi)
        if new_roi != tuple(self.controller.get_roi()):
            (hstart, hend, vstart, vend, hbin, vbin) = new_roi
            was_live = self.is_live
            self.stop()  # most cameras do not change the ROI while they acquire
            self.controller.set_roi(hstart=hstart, hend=hend, vstart=vstart, vend=vend, hbin=hbin, vbin=vbin)
            self.emit_status(ThreadCommand('Update_Status', [f'Changed ROI: {new_roi}']))
            self.controller.setup_acquisition()
            # Finally, prepare view for displaying the new data
            self.compute_axes()
            self._prepare_view()
            if was_live:
                self.grab_data(self.Naverage, live=True)

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "exposure_time":
            self.controller.set_exposure(param.value()/1000)

        elif param.name() == "fps_on":
            self.settings.child('timing_opts', 'fps').setOpts(visible=param.value())

        elif param.name() in ['x_binning', 'y_binning']:
            # We handle ROI and binning separately for clarity
            (hstart, hend, vstart, vend, *_) = self.controller.get_roi()  # Get current ROI
            xbin = self.settings['roi', 'x_binning']
            ybin = self.settings['roi', 'y_binning']
            self.update_rois((hstart, hend, vstart, vend, xbin, ybin))

        elif param.name() == "clear_roi":
            if param.value():   # Switching on ROI
                self.clear_roi()
                param.setValue(False)

    def ini_detector_custom(self, controller=None):
        """Create (master) or store (slave) the camera driver in ``self.controller``, to reimplement"""
        raise NotImplementedError

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """

        self.ini_detector_custom(controller)

        self.get_device_info()
        self.get_set_main_parameters()
        self.setup_callback_thread()
        self.controller.set_frame_format("array")

        info = "Initialized camera"
        initialized = True
        return info, initialized

    def get_device_info(self):

        device_info = self.controller.get_device_info()

        # Get camera name/model
        if hasattr(device_info, 'name'):
            self.settings.child('camera_name').setValue(device_info.name)
        elif hasattr(device_info, 'model'):
            self.settings.child('camera_name').setValue(device_info.model)

    def get_set_main_parameters(self):
        # Set exposure time
        self.controller.set_exposure(self.settings['timing_opts', 'exposure_time']/1000)

        # FPS visibility
        self.settings.child('timing_opts', 'fps').setOpts(visible=self.settings['timing_opts', 'fps_on'])

        # get roi limits
        self.controller.get_roi_limits()

        # Update image parameters
        (hstart, hend, vstart, vend, hbin, vbin) = self.controller.get_roi()
        height, width = self.controller.get_data_dimensions()
        self.settings.child('roi', 'x_binning').setValue(hbin)
        self.settings.child('roi', 'y_binning').setValue(vbin)
        self.settings.child('hdet').setValue(width)
        self.settings.child('vdet').setValue(height)
        self.compute_axes()

    @property
    def callback(self) -> Type[CameraCallback]:
        """ Return the class handling the wait for acquisition and signal emission

        Should be reimplemented as well as CameraCallback if needed
        """
        return CameraCallback

    def setup_callback_thread(self):
        callback = self.callback(self.controller)
        self.settings.child('buffer', 'mode').setReadonly(True)


        self.callback_thread = QtCore.QThread(self)  # creation of a Qt5 thread
        callback.moveToThread(self.callback_thread)  # callback object will live within this thread

        callback.data_sig.connect(
            self.emit_data)  # when the wait for acquisition returns (with data taken), emit_data will be fired
        callback.error.connect(self.handle_error)

        self.callback_signal.connect(callback.set_do_grab)
        self.callback_thread.callback = callback
        self.callback_thread.start()

        self._prepare_view()

    def handle_error(self):
        self.stop()

    def _prepare_view(self):
        """Preparing a data viewer by emitting temporary data. Typically, needs to be called whenever the
        ROIs are changed"""

        height, width = self.controller.get_data_dimensions()

        self.settings.child('hdet').setValue(width)
        self.settings.child('vdet').setValue(height)
        mock_data = np.zeros((height, width))

        if width != 1 and height != 1:
            data_shape = 'Data2D'
        else:
            data_shape = 'Data1D'

        if data_shape != self.data_shape:
            self.data_shape = data_shape
            # init the viewers
            self.dte_signal_temp.emit(DataToExport('Camera',
                                                   data=[DataFromPlugins(name='Camera',
                                                                         data=[np.squeeze(mock_data)],
                                                                         dim=self.data_shape,
                                                                         labels=[f'Camera_{self.data_shape}'])]))
            QtWidgets.QApplication.processEvents()

    def grab_data(self, Naverage=1, **kwargs):
        """
        Grabs the data. ASynchronous method (kinda).
        ----------
        Naverage: (int) Number of averaging
        kwargs: (dict) of others optionals arguments
        """
        try:
            # Warning, acquisition_in_progress returns 1,0 and not a real bool
            self.is_live = kwargs.get('live', False)
            self.Naverage = Naverage

            self.n_frames = 1

            if not self.controller.acquisition_in_progress():
                self.controller.clear_acquisition()
                self.controller.start_acquisition(nframes=self.n_frames)
            #Then start the acquisition
            self.callback_signal.emit(Grab(do_acquisition=True,
                                           snap=not self.is_live,
                                           n_average=Naverage,
                                           nframes=self.n_frames,
                                           since=self.settings['buffer', 'mode']))

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))

    def emit_data(self, frame: np.ndarray=None):
        """ Function used to emit data obtained by callback.

        Parameter
        ---------
        status: bool
            If True a frame is available, If False, a Timeout occurred while waiting for the frame

        See Also
        --------
        daq_utils.ThreadCommand
        """
        try:
            # Get  data from buffer
            if frame is None:
                frame = self.controller.read_newest_image()
            # Emit the frame.
            if frame is not None:
                conversion_str = self.settings['color_conversion']
                if conversion_str not in (None, "None") and cv2 is not None:
                    for ind_average in range(frame.shape[0]):
                        for ind_frame in range(frame.shape[1]):
                            if ind_frame == 0 and ind_average == 0:
                                new_frame = cv2.cvtColor(frame[ind_average, ind_frame, ...],
                                                         getattr(cv2, f'COLOR_{conversion_str}'))
                                shape = [frame.shape[0], frame.shape[1]] + list(new_frame.shape)
                                out_frames = np.zeros(shape, dtype=new_frame.dtype)
                                out_frames[ind_average, ind_frame, ...] = new_frame
                            else:
                                cv2.cvtColor(frame[ind_average, ind_frame, ...],
                                             getattr(cv2, f'COLOR_{conversion_str}'),
                                             out_frames[ind_average, ind_frame, ...])
                else:
                    out_frames = frame
                if self.Naverage > 1:
                    out_frames = np.sum(out_frames, axis=0) / self.Naverage
                else:
                    out_frames = out_frames[0, ...]

                if self.n_frames > 1:
                    pass
                    #todo handle chunks of frames in ND data
                else:
                    out_frames = out_frames[0, ...]

                if out_frames.shape[-1] == 3:
                    data_arrays = [np.atleast_1d(out_frames[..., ind]) for ind in range(3)]
                    labels = ['Red', 'Green', 'Blue']
                else:
                    labels = ['Intensity']
                    data_arrays = [out_frames]

                self.dte_signal.emit(
                    DataToExport('Camera',
                                 data=[DataFromPlugins(name='Camera',
                                                       data=data_arrays,
                                                       dim=self.data_shape,
                                                       labels=labels,
                                                       axes=[self.y_axis, self.x_axis])]))
            if self.settings.child('timing_opts', 'fps_on').value():
                self.update_fps()

            # To make sure that timed events are executed in continuous grab mode
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    def update_fps(self):
        current_tick = perf_counter()
        frame_time = current_tick-self.last_tick

        if self.last_tick != 0.0 and frame_time != 0.0:
            # We don't update FPS for the first frame, and we also avoid divisions by zero

            if self.fps == 0.0:
                self.fps = 1 / frame_time
            else:
                # If we already have an FPS calculated, we smooth its evolution
                self.fps = 0.9 * self.fps + 0.1 / frame_time

        self.last_tick = current_tick

        # Update reading
        self.settings.child('timing_opts', 'fps').setValue(round(self.fps, 1))

    def close(self):
        """
        Terminate the communication protocol
        """
        # Terminate the communication

        self.stop()
        if self.callback_thread is not None:
            self.callback_thread.quit()
            self.callback_thread.wait()

        self.controller.close()
        self.settings.child('buffer', 'mode').setReadonly(False)

    def stop(self):
        """Stop the acquisition."""
        self.is_live = False
        self.callback_signal.emit(Grab(do_acquisition=False))
        QtWidgets.QApplication.processEvents()
        self.controller.clear_acquisition()
        return ''





import dataclasses
from typing import Type

import cv2

from pymodaq_data import DataToExport
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_utils.utils import ThreadCommand
from pymodaq_gui.parameter import Parameter
try:
    from pymodaq_gui.plotting.items.roi import RoiInfo  # pymodaq > 5.1.x
except ImportError:
    from pymodaq_gui.plotting.utils.plot_utils import RoiInfo

from pymodaq.utils.data import DataFromPlugins, Axis
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main

from pylablib.devices.interface.camera import trim_frames

from qtpy import QtWidgets, QtCore
import numpy as np
from time import perf_counter


logger = set_logger(get_module_name(__file__))


cam_params = [
    {'title': 'Camera name:', 'name': 'camera_name', 'type': 'str', 'value': '', 'readonly': True},
    {'title': 'Color Conversion:', 'name': 'color_conversion', 'type': 'list',
     'limits': ['None', 'RGB2GRAY', 'BAYER_BG2RGB', 'BAYER_BG2GRAY']},
    {'title': 'ROI', 'name': 'roi', 'type': 'group', 'children': [
        {'title': 'Update ROI from Viewer', 'name': 'update_roi', 'type': 'led', 'value': False},
        {'title': 'Apply ROI', 'name': 'apply_roi', 'type': 'led', 'value': False},
        {'title': 'Clear ROI+Bin', 'name': 'clear_roi', 'type': 'bool_push', 'value': False},
        {'title': 'ROI:', 'name': 'roi_slices', 'type': 'str', 'value': ''},
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



class CameraBasePyLabLib(DAQ_Viewer_base):
    """
    Base implementation for Camera using pylablib framework. Works for TSI and uc480 thorlabs camera and rpobaly others
    """
    serial_numbers = []

    serial_params = [{'title': 'Serial number:', 'name': 'serial_number', 'type': 'list', 'limits': serial_numbers}]

    params = comon_parameters + serial_params + cam_params

    callback_signal = QtCore.Signal(Grab)
    live_mode_available = True
    hardware_averaging = True

    def ini_attributes(self):
        self.controller = None
        self.callback_thread: QtCore.QThread = None
        self.is_live: bool = False
        self.Naverage: int = 1

        self.x_axis: Axis = None
        self.y_axis: Axis = None

        self.roi_select_info: RoiInfo = None

        self.last_tick = 0.0  # time counter used to compute FPS
        self.fps = 0.0

        self.data_shape: str = ''


    def roi_select(self, roi_info: RoiInfo, ind_viewer: int = 0):
        """ Automatically called when a user use the RoiSelect ROi from a 2D viewer"""
        self.roi_select_info = roi_info
        self.roi_select_viewer_index = ind_viewer

        if self.settings['roi', 'update_roi']:
            self.settings['roi', 'roi_slices'] = str(roi_info.to_slices())
            if self.settings['roi', 'apply_roi']:
                self.apply_roi()

    def apply_roi(self):
        roi_info = RoiInfo.from_slices(eval(self.settings['roi', 'roi_slices']))
        new_roi = (roi_info.origin[1], roi_info.size[1], self.settings['roi', 'x_binning'],
                   roi_info.origin[0], roi_info.size[0], self.settings['roi', 'y_binning'])
        self.update_rois(new_roi)

    def compute_axes(self):
        (hstart, hend, vstart, vend, hbin, vbin) = self.controller.get_roi()
        slices = [slice(vstart, vend, vbin), slice(hstart, hend, hbin)]
        self.settings.child('roi', 'roi_slices').setValue(str(slices))
        roi_info = RoiInfo.from_slices(slices)

        self.x_axis = Axis('x_axis', offset=roi_info.origin[1],
                           scaling=self.settings['roi', 'x_binning'],
                           size=int(roi_info.size[1]),
                           index=1)
        self.y_axis = Axis('y_axis', offset=roi_info.origin[0],
                           scaling=self.settings['roi', 'y_binning'],
                           size=int(roi_info.size[0]),
                           index=0)

    def clear_roi(self):
        wdet, hdet = self.controller.get_detector_size()
        self.settings.child('roi', 'x_binning').setValue(1)
        self.settings.child('roi', 'y_binning').setValue(1)

        new_roi = (0, wdet, 1, 0, hdet, 1)
        self.update_rois(new_roi)

    def update_rois(self, new_roi):
        # In pylablib, ROIs compare as tuples
        (new_x, new_width, new_xbinning, new_y, new_height, new_ybinning) = new_roi
        if new_roi != self.controller.get_roi():
            # self.controller.set_attribute_value("ROIs",[new_roi])
            self.controller.set_roi(hstart=new_x, hend=new_x + new_width, vstart=new_y, vend=new_y + new_height,
                                    hbin=new_xbinning, vbin=new_ybinning)
            self.emit_status(ThreadCommand('Update_Status', [f'Changed ROI: {new_roi}']))
            self.controller.clear_acquisition()
            self.controller.setup_acquisition()
            # Finally, prepare view for displaying the new data
            self._prepare_view()
            self.compute_axes()

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

        elif param.name() == "apply_roi":
            if param.value():   # Switching on ROI
                self.apply_roi()
            else:
                self.clear_roi()

        elif param.name() in ['x_binning', 'y_binning']:
            # We handle ROI and binning separately for clarity
            (x0, w, y0, h, *_) = self.controller.get_roi()  # Get current ROI
            xbin = self.settings['roi', 'x_binning']
            ybin = self.settings['roi', 'y_binning']
            new_roi = (x0, w, xbin, y0, h, ybin)
            self.update_rois(new_roi)

        elif param.name() == "clear_roi":
            if param.value():   # Switching on ROI
                self.clear_roi()
                param.setValue(False)

    def ini_detector_custom(self, controller=None):
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
        slices = [slice(vstart, vend, vbin), slice(hstart, hend, hbin)]
        self.settings.child('roi', 'roi_slices').setValue(str(slices))
        self.compute_axes()

    @property
    def callback(self) -> Type[CameraCallback]:
        """ Return the class handling the wait for acquisition and signal emission

        Should be reimplement as well as CameraCallback if needed
        """
        return CameraCallback

    def setup_callback_thread(self):
        # Way to define a wait function with arguments
        wait_func = lambda: self.controller.wait_for_frame(since=self.settings['buffer', 'mode'],
                                                           nframes=1, timeout=20.0)
        callback = CameraCallback(self.controller)
        self.settings.child('buffer', 'mode').setReadonly(True)


        self.callback_thread = QtCore.QThread()  # creation of a Qt5 thread
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
            self.data_grabed_signal_temp.emit([DataFromPlugins(name='Thorlabs Camera',
                                                               data=[np.squeeze(mock_data)],
                                                               dim=self.data_shape,
                                                               labels=[f'ThorCam_{self.data_shape}'])])
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
                if conversion_str != "None":
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
        self.callback_signal.emit(Grab(do_acquisition=False))
        QtWidgets.QApplication.processEvents()
        self.controller.clear_acquisition()
        return ''





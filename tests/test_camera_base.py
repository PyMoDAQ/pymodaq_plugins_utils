"""The camera plugin base class, driven by a fake camera driver (no pylablib, no hardware)"""
import importlib.util
import time

import numpy as np
import pytest

from pymodaq_plugins_utils.hardware.camera_driver import CameraDriverBase

try:
    available = importlib.util.find_spec('pymodaq.utils.plugin_testing') is not None
except ModuleNotFoundError:  # pymodaq is not installed
    available = False
if not available:
    pytest.skip('The harness needs PyMoDAQ >= 5.3.2', allow_module_level=True)

from pymodaq.utils.plugin_testing import make_detector, grab_and_wait, wait_for_signal

try:
    from pymodaq_gui.plotting.items.roi import RoiInfo  # pymodaq > 5.1.x
except ImportError:
    from pymodaq_gui.plotting.utils.plot_utils import RoiInfo

from pymodaq_plugins_utils.hardware.camera_base import CameraBase

WIDTH, HEIGHT = 40, 30


class FakeCamera(CameraDriverBase):
    """A sensor whose pixel (row, column) is worth 100 * row + column, with ROI and binning"""

    def __init__(self):
        self.sensor = 100. * np.arange(HEIGHT)[:, None] + np.arange(WIDTH)[None, :]
        self.roi = (0, WIDTH, 0, HEIGHT, 1, 1)
        self.exposure = None
        self.running = False
        self.closed = False

    def get_detector_size(self):
        return WIDTH, HEIGHT

    def get_data_dimensions(self):
        hstart, hend, vstart, vend, hbin, vbin = self.roi
        return (vend - vstart) // vbin, (hend - hstart) // hbin

    def get_roi(self):
        return self.roi

    def set_roi(self, hstart=0, hend=None, vstart=0, vend=None, hbin=1, vbin=1):
        if self.running:
            raise RuntimeError('The ROI cannot change during the acquisition')  # as most cameras
        self.roi = (hstart, WIDTH if hend is None else hend, vstart, HEIGHT if vend is None else vend, hbin, vbin)

    def set_exposure(self, exposure):
        self.exposure = exposure

    def setup_acquisition(self):
        pass

    def clear_acquisition(self):
        self.running = False

    def start_acquisition(self, nframes=1):
        self.running = True

    def acquisition_in_progress(self):
        return self.running

    def wait_for_frame(self, since='lastread', nframes=1, timeout=20.0):
        if self.running:
            time.sleep(0.005)  # the exposure
        return self.running  # as pylablib, False if the acquisition is stopped

    def read_newest_image(self, peek=False):
        hstart, hend, vstart, vend, hbin, vbin = self.roi
        height, width = self.get_data_dimensions()
        crop = self.sensor[vstart:vstart + height * vbin, hstart:hstart + width * hbin]
        return crop.reshape(height, vbin, width, hbin).mean(axis=(1, 3))

    def read_multiple_images(self, rng=None, peek=False, missing_frame='skip', return_rng=False):
        if self.running:
            frames, rng = self.read_newest_image()[None, ...], (0, 1)
        else:
            frames, rng = np.zeros((0,) + self.get_data_dimensions()), (0, 0)
        return (frames, rng) if return_rng else frames

    def close(self):
        self.closed = True


class FakeCameraWithPixelSize(FakeCamera):
    def get_pixel_size(self):
        return 5.0, 10.0


class FakeCameraSeveralFrames(FakeCamera):
    """A camera that kept several frames since the last read (as when nobody reads while it acquires)"""
    n_unread = 3

    def read_multiple_images(self, rng=None, peek=False, missing_frame='skip', return_rng=False):
        frames = np.repeat(self.read_newest_image()[None, ...], self.n_unread, axis=0)
        return (frames, (0, self.n_unread)) if return_rng else frames


class DAQ_2DViewer_Fake(CameraBase):
    driver_class = FakeCamera

    def ini_detector_custom(self, controller=None):
        self.controller = self.driver_class() if self.is_master else controller


class DAQ_2DViewer_FakeSeveralFrames(DAQ_2DViewer_Fake):
    driver_class = FakeCameraSeveralFrames


class DAQ_2DViewer_FakeWithPixelSize(DAQ_2DViewer_Fake):
    driver_class = FakeCameraWithPixelSize


@pytest.fixture
def camera(qapp):
    plugin = make_detector(DAQ_2DViewer_Fake)
    yield plugin
    plugin.close()


@pytest.fixture
def calibrated_camera(qapp):
    plugin = make_detector(DAQ_2DViewer_FakeWithPixelSize)
    yield plugin
    plugin.close()


def click_led(plugin, name: str, pressed: bool = True):
    """What the user does: press (or release) the button of the parameter, which the DAQ_Viewer sends to the plugin"""
    param = plugin.settings.child('roi', name)
    param.opts['checked'] = pressed
    param.activate()


def select_roi(plugin, rows: slice, columns: slice):
    """What the user does: draw a ROI in the viewer, update it, then apply it"""
    plugin.roi_select(RoiInfo.from_slices([rows, columns]))
    click_led(plugin, 'update_roi')
    click_led(plugin, 'apply_roi')


def roi_leds(plugin):
    return plugin.settings['roi', 'update_roi'], plugin.settings['roi', 'apply_roi']


def set_binning(plugin, xbin: int, ybin: int):
    for name, value in (('x_binning', xbin), ('y_binning', ybin)):
        param = plugin.settings.child('roi', name)
        param.setValue(value)
        plugin.commit_settings(param)


class TestSeveralUnreadFrames:
    def test_a_snap_gives_one_image_when_the_camera_kept_several_frames(self, qapp):
        plugin = make_detector(DAQ_2DViewer_FakeSeveralFrames)
        try:
            dte = grab_and_wait(plugin)
            assert dte[0].dim.name == 'Data2D'
            assert dte[0].shape == (HEIGHT, WIDTH)
        finally:
            plugin.close()


class TestDriverContract:
    @pytest.mark.parametrize('method, args', [
        ('get_detector_size', ()), ('get_data_dimensions', ()), ('get_roi', ()), ('set_roi', ()),
        ('set_exposure', (0.1,)), ('setup_acquisition', ()), ('clear_acquisition', ()), ('start_acquisition', ()),
        ('acquisition_in_progress', ()), ('wait_for_frame', ()), ('read_multiple_images', ()),
        ('read_newest_image', ()), ('close', ()),
    ])
    def test_what_a_driver_has_to_implement(self, method, args):
        with pytest.raises(NotImplementedError):
            getattr(CameraDriverBase(), method)(*args)

    def test_optional_methods_are_harmless(self):
        driver = CameraDriverBase()
        assert driver.get_device_info() is None
        assert driver.get_roi_limits() is None
        assert driver.get_pixel_size() is None
        driver.set_frame_format('array')


class TestCameraBase:
    def test_the_camera_averages_so_the_viewer_does_not(self):
        """The frames are averaged by the camera thread (Grab.n_average): the viewer averaging them again would grab
        Naverage x Naverage frames"""
        assert CameraBase.hardware_averaging is True
        assert CameraBase.live_mode_available is True

    def test_grab_a_full_frame(self, camera):
        dte = grab_and_wait(camera)
        assert dte[0].shape == (HEIGHT, WIDTH)
        assert dte[0][0][3, 7] == pytest.approx(307)
        assert (camera.x_axis.size, camera.y_axis.size) == (WIDTH, HEIGHT)

    def test_exposure_is_sent_in_seconds(self, camera):
        param = camera.settings.child('timing_opts', 'exposure_time')
        param.setValue(50)
        camera.commit_settings(param)
        assert camera.controller.exposure == pytest.approx(0.05)

    def test_axes_are_in_pixels_when_the_driver_does_not_know_the_pixel_size(self, camera):
        assert camera.x_axis.units == ''
        assert camera.x_axis.scaling == 1
        assert camera.x_axis.offset == 0

    def test_roi_selected_in_the_viewer_becomes_the_camera_roi(self, camera):
        select_roi(camera, rows=slice(5, 15), columns=slice(10, 30))
        assert camera.controller.get_roi() == (10, 30, 5, 15, 1, 1)
        dte = grab_and_wait(camera)
        assert dte[0].shape == (10, 20)
        assert dte[0][0][0, 0] == pytest.approx(5 * 100 + 10)
        assert camera.x_axis.offset == 10
        assert camera.y_axis.offset == 5

    def test_roi_is_kept_inside_the_sensor(self, camera):
        select_roi(camera, rows=slice(-5, 15), columns=slice(10, 100))
        assert camera.controller.get_roi() == (10, WIDTH, 0, 15, 1, 1)

    def test_leds_are_grey_until_activated(self, camera):
        assert roi_leds(camera) == ('off', 'off')

    def test_leds_are_green_once_the_roi_is_updated_and_applied(self, camera):
        select_roi(camera, rows=slice(5, 15), columns=slice(10, 30))
        assert roi_leds(camera) == ('idle', 'idle')

    def test_updating_takes_the_roi_now_in_the_viewer(self, camera):
        camera.roi_select(RoiInfo.from_slices([slice(5, 15), slice(10, 30)]))  # drawn before the click
        assert camera.roi_sensor_slices is None

        click_led(camera, 'update_roi')

        assert camera.roi_sensor_slices == (slice(5, 15), slice(10, 30))
        assert roi_leds(camera) == ('idle', 'off')
        assert camera.controller.get_roi() == (0, WIDTH, 0, HEIGHT, 1, 1)  # not applied yet

    def test_updating_without_roi_in_the_viewer_is_an_error(self, camera):
        click_led(camera, 'update_roi')
        assert roi_leds(camera) == ('error', 'off')
        assert not camera.settings.child('roi', 'update_roi').opts['checked']  # the button is released

    def test_releasing_update_stops_following_the_viewer(self, camera):
        select_roi(camera, rows=slice(5, 15), columns=slice(10, 30))
        click_led(camera, 'update_roi', pressed=False)
        assert roi_leds(camera) == ('off', 'idle')

        camera.roi_select(RoiInfo.from_slices([slice(0, 10), slice(0, 10)]))
        assert camera.controller.get_roi() == (10, 30, 5, 15, 1, 1)

    def test_releasing_apply_goes_back_to_the_full_sensor(self, camera):
        select_roi(camera, rows=slice(5, 15), columns=slice(10, 30))
        click_led(camera, 'apply_roi', pressed=False)
        assert roi_leds(camera) == ('idle', 'off')
        assert camera.controller.get_roi() == (0, WIDTH, 0, HEIGHT, 1, 1)
        assert camera.settings['hdet'] == WIDTH
        assert camera.roi_sensor_slices == (slice(5, 15), slice(10, 30))  # applying again gives the same ROI

        click_led(camera, 'apply_roi')
        assert camera.controller.get_roi() == (10, 30, 5, 15, 1, 1)
        assert roi_leds(camera) == ('idle', 'idle')

    def test_applying_computes_the_image_size(self, camera):
        select_roi(camera, rows=slice(5, 15), columns=slice(10, 30))
        assert camera.settings['hdet'] == 20
        assert camera.settings['vdet'] == 10

    def test_applying_a_roi_outside_the_sensor_is_an_error(self, camera):
        select_roi(camera, rows=slice(5, 15), columns=slice(WIDTH + 10, WIDTH + 30))
        assert roi_leds(camera)[1] == 'error'
        assert camera.controller.get_roi() == (0, WIDTH, 0, HEIGHT, 1, 1)

    def test_applying_with_a_failing_camera_is_an_error(self, camera):
        camera.controller.set_roi = lambda **kwargs: (_ for _ in ()).throw(RuntimeError('refused'))
        camera.roi_select(RoiInfo.from_slices([slice(5, 15), slice(10, 30)]))
        click_led(camera, 'apply_roi')
        assert roi_leds(camera)[1] == 'error'

    def test_applied_roi_follows_the_viewer_until_cleared(self, camera):
        select_roi(camera, rows=slice(5, 15), columns=slice(10, 30))

        camera.roi_select(RoiInfo.from_slices([slice(0, 10), slice(0, 10)]))
        assert camera.controller.get_roi() == (0, 10, 0, 10, 1, 1)

        clear = camera.settings.child('roi', 'clear_roi')
        clear.setValue(True)
        camera.commit_settings(clear)
        assert roi_leds(camera) == ('off', 'off')
        camera.roi_select(RoiInfo.from_slices([slice(0, 5), slice(0, 5)]))
        assert camera.controller.get_roi() == (0, WIDTH, 0, HEIGHT, 1, 1)

    def test_changing_the_binning_keeps_the_roi(self, camera):
        select_roi(camera, rows=slice(5, 15), columns=slice(10, 30))
        set_binning(camera, 2, 5)
        assert camera.controller.get_roi() == (10, 30, 5, 15, 2, 5)
        dte = grab_and_wait(camera)
        assert dte[0].shape == (2, 10)
        assert camera.x_axis.size == 10
        assert camera.x_axis.scaling == 2
        assert camera.y_axis.size == 2
        assert camera.y_axis.scaling == 5

    def test_clear_roi_goes_back_to_the_full_sensor(self, camera):
        select_roi(camera, rows=slice(5, 15), columns=slice(10, 30))
        set_binning(camera, 2, 2)
        clear = camera.settings.child('roi', 'clear_roi')
        clear.setValue(True)
        camera.commit_settings(clear)
        assert camera.controller.get_roi() == (0, WIDTH, 0, HEIGHT, 1, 1)
        assert grab_and_wait(camera)[0].shape == (HEIGHT, WIDTH)

    def test_roi_change_during_live_restarts_it_with_the_new_roi(self, camera):
        wait_for_signal(camera.dte_signal, lambda: camera.grab_data(1, live=True), timeout_ms=3000)
        assert camera.is_live
        select_roi(camera, rows=slice(5, 15), columns=slice(10, 30))
        assert camera.controller.get_roi() == (10, 30, 5, 15, 1, 1)
        dte = wait_for_signal(camera.dte_signal, lambda: None, timeout_ms=3000)
        while dte[0].shape != (10, 20):  # images of the old ROI may still be on their way
            dte = wait_for_signal(camera.dte_signal, lambda: None, timeout_ms=3000)
        camera.stop()
        assert not camera.is_live
        assert not camera.controller.acquisition_in_progress()

    def test_close_releases_the_camera(self, qapp):
        plugin = make_detector(DAQ_2DViewer_Fake)
        controller = plugin.controller
        plugin.close()
        assert controller.closed


class TestPixelSize:
    def test_axes_are_calibrated(self, calibrated_camera):
        dte = grab_and_wait(calibrated_camera)
        assert calibrated_camera.x_axis.units == 'µm'
        assert calibrated_camera.x_axis.scaling == pytest.approx(5.0)
        assert calibrated_camera.y_axis.scaling == pytest.approx(10.0)
        assert dte[0].shape == (HEIGHT, WIDTH)

    def test_roi_with_fractional_values_is_rounded_in_pixels_not_truncated_in_um(self, calibrated_camera):
        # 5 um x 10 um pixels: 52.4 -> 10.48 and 157.6 -> 31.52 pixels (columns), 53 -> 5.3 and 147 -> 14.7 (rows)
        select_roi(calibrated_camera, rows=slice(53.0, 147.0), columns=slice(52.4, 157.6))
        assert calibrated_camera.controller.get_roi() == (10, 32, 5, 15, 1, 1)

    def test_roi_drawn_on_calibrated_axes_is_converted_to_pixels(self, calibrated_camera):
        select_roi(calibrated_camera, rows=slice(50, 150), columns=slice(50, 150))  # um: 5 to 15 rows, 10 to 30 cols
        assert calibrated_camera.controller.get_roi() == (10, 30, 5, 15, 1, 1)
        assert calibrated_camera.x_axis.offset == pytest.approx(50.0)
        assert calibrated_camera.y_axis.offset == pytest.approx(50.0)

    def test_binning_multiplies_the_scaling(self, calibrated_camera):
        set_binning(calibrated_camera, 2, 1)
        assert calibrated_camera.x_axis.scaling == pytest.approx(10.0)
        assert calibrated_camera.x_axis.size == WIDTH // 2

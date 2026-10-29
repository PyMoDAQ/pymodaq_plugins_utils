"""The shaper contract: settings groups of the plugins, read back from the settings of their control module"""
import pytest
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_utils.hardware.shaper import (Shaper, ModulatorType, shaper_params, fill_shaper_settings,
                                                   SHAPER_VERSION)


class FakeModule:
    """What an application has: the DAQ_Move control module, with the settings tree of its plugin"""
    _hw_settings_name = 'actuator_settings'

    def __init__(self, children=None):
        children = shaper_params() if children is None else children
        self.settings = Parameter.create(name='settings', type='group', children=[
            {'name': self._hw_settings_name, 'type': 'group', 'children': children}])

    @property
    def plugin_settings(self):
        return self.settings.child(self._hw_settings_name)


def fill(module, **kwargs):
    values = dict(name='SLM', width=1920, height=1080, pixel_size_x=8., pixel_size_y=9.)
    values.update(kwargs)
    fill_shaper_settings(module.plugin_settings, **values)


def test_not_initialized_yet():
    assert Shaper.from_module(FakeModule()) is None


def test_roundtrip():
    module = FakeModule()
    fill(module, has_internal_calibration=True, modulator_type='both')
    shaper = Shaper.from_module(module)
    assert shaper == Shaper('SLM', 1920, 1080, 8., True, True, ModulatorType.BOTH, 9.)
    assert shaper.shape == (1080, 1920)


def test_pixel_size_y_defaults_to_x():
    module = FakeModule()
    fill(module, pixel_size_y=None)
    assert Shaper.from_module(module).pixel_size_y == 8.


def test_use_internal_calibration_is_the_choice_of_the_user():
    module = FakeModule()
    fill(module, has_internal_calibration=True)
    module.plugin_settings.child('shaper_properties', 'use_internal_calibration').setValue(False)
    shaper = Shaper.from_module(module)
    assert shaper.has_internal_calibration and not shaper.use_internal_calibration


def test_plugin_without_the_groups():
    assert Shaper.from_module(FakeModule([])) is None
    assert Shaper.from_module(object()) is None


def test_unknown_version_is_refused():
    module = FakeModule()
    fill(module)
    module.plugin_settings.child('shaper_geometry', 'version').setValue(SHAPER_VERSION + 1)
    assert Shaper.from_module(module) is None


def test_missing_version_is_refused():
    params = shaper_params()
    params[1]['children'] = [c for c in params[1]['children'] if c['name'] != 'version']
    module = FakeModule(params)
    fill_params = module.plugin_settings.child('shaper_geometry')
    for key, value in (('width', 4), ('height', 4), ('pixel_size_x', 8.)):
        fill_params.child(key).setValue(value)
    assert Shaper.from_module(module) is None


def test_unknown_modulator_type():
    with pytest.raises(ValueError):
        Shaper('x', 1, 1, 1., False, False, 'unknown')

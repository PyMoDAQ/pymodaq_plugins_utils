"""Common interface of the instrument plugins driving a *shaper* (SLM, DMD...)

A shaper plugin adds the settings groups returned by :func:`shaper_params` to its parameters and fills them once the
instrument is initialized (:func:`fill_shaper_settings`). The applications (as the BeamShaping extension) do not call the
plugin, which lives in another thread: they read these groups from the settings of its control module
(:meth:`Shaper.from_module`). The names of the groups and of their children, and :data:`SHAPER_VERSION`, are the
contract. Evolution rule: only add optional children, never rename or remove one; a breaking change increments
:data:`SHAPER_VERSION`.
"""
import dataclasses
from typing import Optional

from pymodaq_utils.enums import StrEnum
from pymodaq_utils.logger import set_logger, get_module_name

logger = set_logger(get_module_name(__file__))

SHAPER_VERSION = 1
GEOMETRY_NAME = 'shaper_geometry'
PROPERTIES_NAME = 'shaper_properties'


class ModulatorType(StrEnum):
    PHASE = 'phase'
    AMPLITUDE = 'amplitude'
    BOTH = 'both'


def shaper_geometry_params() -> dict:
    """Group with the number of pixels and their size (micrometer), read only"""
    return {'title': 'Shaper geometry:', 'name': GEOMETRY_NAME, 'type': 'group', 'children': [
        {'title': 'Version:', 'name': 'version', 'type': 'int', 'value': SHAPER_VERSION, 'readonly': True,
         'visible': False},
        {'title': 'Width (pixels):', 'name': 'width', 'type': 'int', 'value': 0, 'readonly': True},
        {'title': 'Height (pixels):', 'name': 'height', 'type': 'int', 'value': 0, 'readonly': True},
        {'title': 'Pixel size x (um):', 'name': 'pixel_size_x', 'type': 'float', 'value': 0., 'readonly': True},
        {'title': 'Pixel size y (um):', 'name': 'pixel_size_y', 'type': 'float', 'value': 0., 'readonly': True},
    ]}


def shaper_properties_params() -> dict:
    """Group with what the shaper does: its name, the type of modulation and whether it converts itself the phases
    (radians) into its grey levels (``has_internal_calibration``), which the user may or not use"""
    return {'title': 'Shaper properties:', 'name': PROPERTIES_NAME, 'type': 'group', 'children': [
        {'title': 'Version:', 'name': 'version', 'type': 'int', 'value': SHAPER_VERSION, 'readonly': True,
         'visible': False},
        {'title': 'Name:', 'name': 'name', 'type': 'str', 'value': '', 'readonly': True},
        {'title': 'Modulator type:', 'name': 'modulator_type', 'type': 'list', 'limits': [m.value for m in ModulatorType],
         'value': ModulatorType.PHASE.value, 'readonly': True},
        {'title': 'Has internal calibration:', 'name': 'has_internal_calibration', 'type': 'bool', 'value': False,
         'readonly': True},
        {'title': 'Use internal calibration:', 'name': 'use_internal_calibration', 'type': 'bool', 'value': False,
         'tip': 'Send phases in radians and let the shaper convert them, instead of using a calibration made '
                'outside of the shaper'},
    ]}


def shaper_params() -> list[dict]:
    """The groups to add to the parameters of a shaper plugin"""
    return [shaper_geometry_params(), shaper_properties_params()]


def fill_shaper_settings(settings, name: str, width: int, height: int, pixel_size_x: float,
                         pixel_size_y: Optional[float] = None,
                         modulator_type: ModulatorType = ModulatorType.PHASE,
                         has_internal_calibration: bool = False) -> None:
    """Fill the groups of :func:`shaper_params` in the ``settings`` of a plugin (to be called in ``ini_stage``)

    ``use_internal_calibration`` is set to ``has_internal_calibration``, the user can change it afterwards.
    """
    geometry = settings.child(GEOMETRY_NAME)
    for key, value in (('width', width), ('height', height), ('pixel_size_x', pixel_size_x),
                       ('pixel_size_y', pixel_size_x if pixel_size_y is None else pixel_size_y)):
        geometry.child(key).setValue(value)
    properties = settings.child(PROPERTIES_NAME)
    for key, value in (('name', name), ('modulator_type', ModulatorType(modulator_type).value),
                       ('has_internal_calibration', has_internal_calibration),
                       ('use_internal_calibration', has_internal_calibration)):
        properties.child(key).setValue(value)


@dataclasses.dataclass
class Shaper:
    name: str
    width: int
    height: int
    pixel_size: float  # in micrometer, along x
    has_internal_calibration: bool
    use_internal_calibration: bool
    modulator_type: ModulatorType
    pixel_size_y: Optional[float] = None  # in micrometer, same as pixel_size if None

    def __post_init__(self):
        if self.pixel_size_y is None:
            self.pixel_size_y = self.pixel_size
        self.modulator_type = ModulatorType(self.modulator_type)

    @property
    def shape(self) -> tuple[int, int]:
        """The shape (height, width) of the arrays to send to the shaper"""
        return self.height, self.width

    @classmethod
    def from_settings(cls, geometry, properties) -> 'Shaper':
        """Build it from the two groups (Parameter objects) of a shaper plugin"""
        return cls(name=properties['name'], width=int(geometry['width']), height=int(geometry['height']),
                   pixel_size=float(geometry['pixel_size_x']), pixel_size_y=float(geometry['pixel_size_y']),
                   has_internal_calibration=bool(properties['has_internal_calibration']),
                   use_internal_calibration=bool(properties['use_internal_calibration']),
                   modulator_type=ModulatorType(properties['modulator_type']))

    @classmethod
    def from_module(cls, module) -> Optional['Shaper']:
        """Read the shaper from a DAQ_Move control module

        None if its plugin has no shaper groups, is not initialized yet (empty geometry) or follows another version
        of the contract (an error is logged in that case).
        """
        try:
            settings = module.settings.child(module._hw_settings_name)
            geometry, properties = settings.child(GEOMETRY_NAME), settings.child(PROPERTIES_NAME)
        except (KeyError, AttributeError):
            return None
        versions = [group['version'] if 'version' in [c.name() for c in group.children()] else None
                    for group in (geometry, properties)]
        if versions != [SHAPER_VERSION] * 2:
            logger.error(f'The shaper groups of {module} have versions {versions}, version {SHAPER_VERSION} is '
                         f'expected: update the plugin or pymodaq_plugins_utils')
            return None
        shaper = cls.from_settings(geometry, properties)
        if shaper.width <= 0 or shaper.height <= 0 or shaper.pixel_size <= 0:
            return None
        return shaper

pymodaq_plugins_utils
#####################

.. the following must be adapted to your developed package, links to pypi, github  description...

.. image:: https://img.shields.io/pypi/v/pymodaq_plugins_utils.svg
   :target: https://pypi.org/project/pymodaq_plugins_utils/
   :alt: Latest Version

.. image:: https://readthedocs.org/projects/pymodaq/badge/?version=latest
   :target: https://pymodaq.readthedocs.io/en/stable/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/PyMoDAQ/pymodaq_plugins_utils/workflows/Upload%20Python%20Package/badge.svg
   :target: https://github.com/PyMoDAQ/pymodaq_plugins_utils
   :alt: Publication Status

.. image:: https://github.com/PyMoDAQ/pymodaq_plugins_utils/actions/workflows/Test.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_plugins_utils/actions/workflows/Test.yml



Authors
=======

* Sebastien J. Weber (sebastien.weber@cemes.fr)
* Constant Schouder (constant.schouder@universite-paris-saclay.fr)

.. if needed use this field

    Contributors
    ============

    * First Contributor
    * Other Contributors

.. if needed use this field

  Depending on the plugin type, delete/complete the fields below


Utilities
=========

* pysvisa stuff
* CameraBase: Base plugin class inheriting from DAQ_Viewer_base and implementing the acquisition in a thread, ROI
  management (shared with the viewer), binning, exposure and FPS for any camera. The camera driver follows
  CameraDriverBase, whose methods are those of the pylablib cameras: write the driver of your camera against it,
  or use a pylablib one as it is.
* CameraBasePyLabLib: CameraBase for the cameras of the pylablib package. Mostly for (S)CMOS camera
* shaper: common interface of the plugins driving a shaper (SLM, DMD...): settings groups with the number of pixels,
  their size and the properties of the modulator, filled by the plugin and read back by the applications (BeamShaping
  extension) from the control module, see ``pymodaq_plugins_utils.hardware.shaper``

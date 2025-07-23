
import omni.ext
import importlib
import os

from .ogn import *

SCENES_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data"))


# Any class derived from `omni.ext.IExt` in a top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when the extension is enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() will be called.
class OceandeformerwarpExtension(omni.ext.IExt):
    # ext_id is the current extension id. It can be used with the extension manager to query additional information,
    # such as where this extension is located in the filesystem.
    def on_startup(self, ext_id):
        print("[OceanDeformerWARP] OceandeformerwarpExtension startup", flush=True)

        try:
            importlib.import_module("omni.kit.browser.sample").register_sample_folder(
                SCENES_PATH,
                "Simulation/Ocean"
            )
        except ImportError as e:
            print("Warning: sample browser not enabled.")



    def on_shutdown(self):
        print("[OceanDeformerWARP] OceandeformerwarpExtension shutdown", flush=True)

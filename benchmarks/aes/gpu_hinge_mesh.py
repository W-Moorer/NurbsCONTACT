"""Compatibility entry point for the installed GPU backend and the source checkout."""

import _gpu_source
from nurbscontact_gpu.backend import export_scene, make_windows, module, pack_mesh, prepare, run, signature

from nurbscontact_gpu.cli import main

if __name__ == "__main__":
    main()

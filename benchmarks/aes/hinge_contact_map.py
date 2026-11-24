"""Compatibility entry point for the installed GPU backend and the source checkout."""

import _gpu_source
from nurbscontact_gpu.geometry import prepare

from nurbscontact_gpu.geometry import main

if __name__ == "__main__":
    main()

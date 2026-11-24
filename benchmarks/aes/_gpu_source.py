"""Locate the source package for standalone scripts in this benchmark directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nurbscontact_gpu.scene import equations, read_model

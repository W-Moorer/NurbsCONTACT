"""Device-resident solve for frictionless contact islands using a frozen normal Jacobian."""
from pathlib import Path
import cupy as cp
import numpy as np


class NormalContactSolve:
    """Solve W = J M_eff^-1 J^T for independent islands, each with the same row count.

    free_gap is the predicted end-step gap in metres, positive when separated.
    Each row selects linear implicit penalty, unilateral multiplier or augmented
    multiplier. Generalized impulses are J^T p. All outputs stay on the device.
    Status codes are 0 (converged), 1 (residual above tolerance), 2 (invalid row).
    """
    PENALTY, LAGRANGE, AUGMENTED_LAGRANGE = range(3)

    def __init__(self, islands, rows):
        if islands <= 0 or rows <= 0:
            raise ValueError("Expected at least one island and contact row")
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        self.shared_bytes = 5 * rows * 8
        if self.shared_bytes > props["sharedMemPerBlock"]:
            raise ValueError("Contact island exceeds the device shared memory capacity")
        self.shape = (islands, rows)
        self.impulse = cp.empty(self.shape, cp.float64)
        self.residual = cp.empty(islands, cp.float64)
        self.sweeps = cp.empty(islands, cp.int32)
        self.status = cp.empty(islands, cp.int32)
        src = Path(__file__).resolve().parents[2] / "src/gpu_normal_contact.cu"
        self.kernel = cp.RawKernel(src.read_text(), "solve_normal_rows", options=("--std=c++14",))

    def enqueue(self, mobility, free_gap, laws, stiffness, step_s, tolerance_m_s=1e-9, max_sweeps=4000):
        islands, rows = self.shape
        for a, shape, dtype in ((mobility, (islands, rows, rows), cp.float64),
                                (free_gap, self.shape, cp.float64), (laws, self.shape, cp.int32),
                                (stiffness, self.shape, cp.float64)):
            if not isinstance(a, cp.ndarray) or a.shape != shape or a.dtype != dtype or not a.flags.c_contiguous:
                raise ValueError("Unexpected device array shape, type or layout")
        if not (np.isfinite(step_s) and step_s > 0 and np.isfinite(tolerance_m_s) and tolerance_m_s > 0):
            raise ValueError("Expected a finite positive step and velocity tolerance")
        if not 1 <= max_sweeps <= np.iinfo(np.int32).max:
            raise ValueError("Invalid iteration limit")
        self.kernel((islands,), (32,), (mobility, free_gap, laws, stiffness, np.int32(rows),
                    np.float64(step_s), np.float64(tolerance_m_s), np.int32(max_sweeps),
                    self.impulse, self.residual, self.sweeps, self.status), shared_mem=self.shared_bytes)
        return self.impulse

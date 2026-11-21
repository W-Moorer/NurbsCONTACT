# Native rigid-body engine comparison

The runner uses MuJoCo 4.0.1, Drake 1.58.0, and the C++ `nc_contact`
executable. Each engine computes its own contacts and integrates its own
states. The runner tests a sphere against a plane and a translating
two-sphere collision. Motion is along a shared axis. Each moving body has
mass 1 kg and nominal radius 35 mm. The incident speed is 0.45 m/s.
The initial separation is 18 mm beyond a 12 mm compliant layer.

Collision proxies encode that layer as a plane offset or two radius
offsets. The normal law is 5000 N/m times the overlap of those proxies.
Gravity, tangential forces, and contact dissipation are zero. MuJoCo uses
its one-dimensional constraint with direct stiffness and constant
impedance. Drake uses continuous point contact with zero Hunt–Crossley
dissipation. Its two 10,000 N/m contact stiffnesses combine in series.
`nc_contact` uses directional gap probes on its original STEP proxies
and the implicit HHT solver. [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/) · [Drake compliant contact](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html)

The reference is the analytical half-sine normal response at reduced mass
1 or 0.5 kg, with conserved center-of-mass velocity in the two-body case.
Each run covers 160 ms at 1, 0.5, or 0.25 ms. The CSVs retain every
time-grid sample, including the analytical position and normal force.
Before scoring, the runner verifies the force evaluated on each
engine's own trajectory against the specified law. The report includes
seven integration timings for each configuration. Model construction,
setup, and CSV export are outside each timed region. Each method's
contact evaluation, solve, and step logic are inside it. The fixtures
isolate normal contact and time integration. The Chen OBB and triangle
BVH comparisons address trimmed geometry and continuous clearance.

## Reproduce on a common Linux host

On Ubuntu 24.04, the build needs Open CASCADE development libraries
(`libocct-data-exchange-dev`, `libocct-visualization-dev`, and
`libtbb-dev`). Run the three backends sequentially on the same idle host.

```bash
cmake -S . -B build/linux-engines -DCMAKE_BUILD_TYPE=Release -DNC_WITH_OCCT=ON
cmake --build build/linux-engines --target nc_contact -j 6
python3 -m venv build/engine-venv
build/engine-venv/bin/python -m pip install mujoco==4.0.1 drake==1.58.0 matplotlib
build/engine-venv/bin/python benchmarks/engines/rigid.py build/engines/nurbs --backend nurbs --nc-contact build/linux-engines/nc_contact
build/engine-venv/bin/python benchmarks/engines/rigid.py build/engines/mujoco --backend mujoco
build/engine-venv/bin/python benchmarks/engines/rigid.py build/engines/drake --backend drake
build/engine-venv/bin/python benchmarks/engines/report.py build/engines/nurbs build/engines/mujoco build/engines/drake results/engine-comparison
```

[Raw traces and reported metrics](../../results/engine-comparison/) · [Comparison figure](../../results/engine-comparison/engine-comparison.pdf)

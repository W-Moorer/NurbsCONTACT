# Common untrimmed patch comparison with TDIB-CCD

The harness calls Chen et al.'s official OBB solver at commit b70aa3e95ab4f75d2166df7e0cbe52b64515a3cb. Both solvers receive the same untrimmed biquadratic control nets and linear control-point velocities. The fixed surface is the unit square at zero height. The moving graph has height \(0.375+0.25u-t\) in the planar-entry case and \(0.375+0.5[(u-0.5)^2+(v-0.5)^2]-t\) in the curved-entry case. The curved-clear case reduces the approach speed to 0.125. Thus both zero-distance entries occur at \(t=0.375\). [Chen et al. paper](https://doi.org/10.1145/3687960) · [Official implementation](https://github.com/xw-c/TDIB-CCD/tree/b70aa3e95ab4f75d2166df7e0cbe52b64515a3cb)

The OBB solver's deltaDist is \(5\times10^{-4}\) in combined parameter and candidate-time widths. The native solver requests \(5\times10^{-4}\) in distance and time, with a 25,000-task cap. The OBB solver returns a candidate time and the centers of the retained parameter cells. The native solver reports a certified prefix and requires a verified original-surface pair for a witnessed entry. Runtime observations include query work and exclude control-net construction. Nine measured repetitions follow one warm-up, alternating the order. They characterize these fixtures and termination rules.

Both solvers exclude the curved-clear case. The OBB candidates for the entry cases are near the analytic time, with center point-pair residuals of \(4.92\times10^{-4}\) and \(6.91\times10^{-4}\). The native safe prefixes reach 0.375000 and 0.374832, respectively. Exact zero-distance witnesses remain unresolved at the task cap. The native-only run at positive clearance 0.025 uses separate outputs.

- [Zero-distance observations](common-observations.csv)
- [Separate positive-margin observations](positive-observations.csv)

## Reproduce

~~~powershell
git clone https://github.com/xw-c/TDIB-CCD.git build/tdib-ccd
git -C build/tdib-ccd checkout b70aa3e95ab4f75d2166df7e0cbe52b64515a3cb
cmake -S benchmarks/chen -B build/chen-common -DTDIB_SOURCE_DIR="$PWD/build/tdib-ccd" -DEIGEN_INCLUDE_DIR=C:/vcpkg/installed/x64-windows/include/eigen3
cmake --build build/chen-common --config Release
build/chen-common/Release/nc_chen_common.exe
build/chen-common/Release/nc_chen_common.exe --positive-only
~~~

Substitute the location of the local Eigen headers when needed. The [rotation follow-up](../clearance-followup/README.md) adds a chord-error bound, outward verification of proposed exclusions and an original-trajectory witness check around the official OBB search.

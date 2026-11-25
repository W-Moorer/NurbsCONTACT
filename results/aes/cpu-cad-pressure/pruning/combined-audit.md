# 同时启用距离剪枝和射线剪枝的开发对照

本页中的“新版”指提交 `492174b` 的试验组合。正式 CPU 入口选用距离剪枝，最终的模型计时和响应见[默认配置核验](README.md)。

同一减速器、同一隐式积分器和共享罚刚度，从初态分别推进 5 s，主步长及输出间隔均为 0.1 ms。纯网格使用 OBB 粗筛和三角面接触区，CAD 使用原 STEP 接触区积分，步内采用已有的预测姿态更新。三组均关闭独立 CCD。模型、载荷、运行入口和原先的 0.5 ms 力矩核验可在上一级报告中复查。[上一版与模型归档](../README.md)

## 谁省了多少时间

Windows 11，AMD Ryzen 9 5900X（12 核），Release 构建。计时含读模、预处理、全部动力学步骤及响应写出。首轮依次运行旧 CAD、新 CAD、网格，期间做过几何诊断。反序复测期间单独运行每个算例，依次运行新 CAD、旧 CAD。每次均从头初始化。历史报告里另外一轮旧版 CAD 用时 464.43 s。

| 整段 5 s 耗时 | 原 STEP | 通用剪枝 STEP（ours） | 纯网格 OBB |
|---|---:|---:|---:|
| 首轮（s） | 519.79 | 407.87 | 134.83 |
| 反序复测（s） | 445.10 | 446.36 | — |

反序复测中前 20 步仍从 94.60 降到 22.31 s，随后 49,980 步从 322.92 增到 400.43 s，最终整机用时相差 1.25 s。反序运行时连共用的粗筛步骤累计耗时也从 99.39 变为 121.57 s。两轮都能复现启动核验提速。持续啮合阶段的用时变化方向相反，五秒闭环的稳定加速比例尚未建立。与本轮网格的 134.83 s 相比，两次新版 CAD 用时分别为其 3.03 和 3.31 倍。[反序复测和响应差](isolated-repeat.json) · [复测旧版分项](repeat-old-run.json) · [复测新版分项](repeat-new-run.json)

再做一次完整消融，只启用第 1、2 项距离剪枝，恢复原有射线查询和每个线程的曲面缓存。该版本耗时 **374.22 s**，启动 20 步耗时 19.68 s，随后 49,980 步耗时 331.78 s。相对相邻旧版长跑节省 70.88 s（15.9%），闭环 0.5 ms 同为 433.2725 N·m。相应数字来自一次单独的五秒运行。[距离剪枝消融](distance-only.json)

首轮节省 111.93 s（21.5%）。仅前 20 步的积分就从 97.91 降至 22.10 s，贡献其中 75.80 s。为分离产生差别的原因，在完全相同的初始姿态，以旧、新、新、旧的次序单独重测有上限的 CAD 距离查询。一级进入 OCCT 最近距离核的特征对从 423 减为 116，耗时从 3.54–3.77 降至 1.51–1.53 s。二级从 1,254 减为 87，耗时从 4.13–4.15 降至 0.84–0.90 s。独立整壳距离分别为 0.046797 和 0.040085 mm，两对均满足本次分离检查的 0.00001 mm 上限。[初态距离、调用计数与耗时](startup-static.json) · [整机统计](dynamics.json)

另用每级八个连续窗口在相同姿态上交替运行，两种 CAD 均启用原有索引，每级保留 640 个预热后的姿态。数值为单级每姿态的平均毫秒数。网格窗口包含区域深度计算，CAD 窗口仅测搜索与分组。整机速度比较使用上表相同求解任务的墙钟时间。[窗口配置、P95 与程序哈希](paired-window-timing.json) · [分项图](search-and-grouping.png)

| 级别和方法 | 粗筛（ms） | 局部搜索（ms） | 接触区分组（ms） | 合计（ms） |
|---|---:|---:|---:|---:|
| 一级原 STEP | 0.428 | 0.585 | 0.048 | 1.061 |
| 一级通用剪枝（ours） | 0.414 | 0.488 | 0.037 | 0.939 |
| 二级原 STEP | 0.749 | 0.983 | 1.485 | 3.217 |
| 二级通用剪枝（ours） | 0.725 | 0.895 | 1.395 | 3.016 |

接触稳定后，二级窗口内约 46% 的 CAD 几何查询时间花在确定哪些面共同承载。首轮新版闭环累计粗筛 109.10 s、局部搜索 131.74 s、分组 136.58 s、承载面积积分 111.55 s。两级并发，这些分项累加会超过实际墙钟时间。[新版累计计数](new-run.json)

## 具体改动

1. 对 B 样条和 Bézier 面、边，投影方向上的极值由全部控制点给出包围范围。顶点和线段直接使用其几何点，其余类型保留 OCCT 包围盒。只有两个包围范围之间的距离已经超过本次查询所需的上限并留足容差，才跳过这一对特征的最近距离计算。
2. 对带裁剪的有限平面，仅保留对侧特征可能靠近平面的那部分控制点凸包。将其投影到平面参数域，检查投影是否与原 CAD 裁剪曲线的包围范围相交。在远离原曲线的位置，可以用裁剪弦段的绕数奇偶判定投影是否位于孔中或外轮廓外。候选靠近边界、曲线类型缺少可靠包围、曲面包容等情况继续按原路径核验。载荷、法向和力矩仍使用原 STEP 曲面。
3. 曲面射线查询按原刚体对共享只读的样条单元包围数据。所有线程保留各自的求交器。检验线段时，先缩短到其可能位于单元包围盒内的区间，再用这个区间检查曲面法向投影范围。三项优化均由几何数据触发，使用同一个 `cad-patch` 入口。[距离剪枝](../../../../src/cad_boundary_distance.cpp) · [平面裁剪](../../../../src/cad_planar_trim.hpp) · [射线查询](../../../../src/cad_contact_regions.cpp)

## 几何和响应核对

将旧、新 CAD 放在网格闭环导出的同一组成对姿态上，每级 50,001 帧。全部帧命中面编号、每一区域包含的面对、区域数和报告的法向均一致，接触点位置差最大为 2.78 × 10⁻¹⁷ m，深度差最大为 1.93 × 10⁻¹⁷ m。该检查能发现剪枝改变搜索结果的情况。与完整 STEP 实体独立积分的几何对照仍采用上一级报告的审计姿态。[全程同姿态核对](same-poses.json)

进一步用分布压力对一级 24 个、二级 32 个审计姿态逐一重算。56 处全部得到结果，按归档精度比较，三维合力、三维力矩和从动轴力矩的新旧差值均为零。固定在网格轨迹的 0.5 ms 姿态，二级输出轴力矩仍为 533.953 N·m，完整 STEP 实体独立积分为 533.53 N·m，同姿态网格为 411.95 N·m。[56 处同姿态载荷](fixed-loads.json) · [完整 STEP、网格及局部法原始对照](../wrenches.csv)

各自从初态推进时，0.5 ms 输出轴力矩新、旧均为 433.2725 N·m，网格为 395.3675 N·m。首轮新旧 CAD 的五秒输出轴力矩 RMS 差为 0.154 N·m，最大差 10.266 N·m 出现在 3.5377 s。固定在旧轨迹该时刻的轴姿态，两版重算均为 149.468 N·m。换成新轨迹的姿态，两版又都给出 138.336 N·m，均找到三个接触区。同一旧可执行文件两次独立积分的最大力矩差也为 10.266 N·m，RMS 差 0.156 N·m。固定姿态核对能解释这里的两档力矩，最初造成闭环轨迹分歧的步骤尚未定位。相对同模型 RecurDyn，五秒输出轴力矩 RMS 差为旧 CAD 42.164、新 CAD 42.163、网格 29.142 N·m。[两条已接受姿态的交叉重算](peak-poses.json) · [端到端耗时与响应曲线](cpu-pruning.png) · [逐步输出轴力矩，单位 N·m](output-torque.npz)

第二种几何采用两颗自由球的正碰，沿用相同主步长和 5 s 时长。两版 50,001 帧响应文件的 SHA-256 完全一致，峰值 22.49913 N（解析值 22.5 N），冲量 0.4500013 N·s（解析值 0.45 N·s）。用时 17.18 和 17.39 s。这个算例说明原有极点附近的整实体核验仍能发现接触，也显示简单双球未获得同样的效率收益。另有核验直接覆盖有孔平面的内外边缘和斜穿接触、刚体旋转，以及有理样条球体的分离、相切和相交。[双球解析对照](sphere.json) · [几何回归测试](../../../../tests/cad_solids.cpp)

## 复现

使用上一级报告的模型和 Release 构建方式。旧版本取 `68c099c`，两套原生可执行文件各自放入独立目录。PowerShell 中 `$oldBin`、`$bin` 为这两个目录，`$scene` 为减速器场景，`$work` 为新输出目录。准备姿态、探测与积分结果均可以从归档模型重建。各原生进程限时 1,800 s。

```powershell
$env:OMP_NUM_THREADS = "2"
$run = "benchmarks/aes/cpu_backend_run.py"
$replay = "benchmarks/aes/cpu_cad_pressure.py"
$audit = "benchmarks/aes/cpu_cad_pruning.py"
$settings = @("--solver","penalty","--penalty-stiffness","shared","--geometry-update","predictor","--spectral-radius","0")
python $run "$oldBin/nc_contact.exe" $scene "$work/old" --method cad-patch @settings
python $run "$bin/nc_contact.exe" $scene "$work/new" --method cad-patch @settings
python $run "$bin/nc_contact.exe" $scene "$work/mesh" --method mesh-obb --scope contour-ray @settings
python $replay prepare $scene "$work/old-replay"
python $replay indexed $scene "$work/old-replay" --exe "$oldBin/nc_cad_probes.exe"
python $replay prepare $scene "$work/replay"
python $replay indexed $scene "$work/replay" --exe "$bin/nc_cad_probes.exe"
python $audit static $scene "$work/replay" "$oldBin/nc_cad_distance_benchmark.exe" "$bin/nc_cad_distance_benchmark.exe" "$work/report"
python $audit spheres "$work/spheres" "$oldBin/nc_contact.exe" "$bin/nc_contact.exe" "$work/report"
python $replay area $scene "$work/replay" --order 0 --exe "$bin/nc_cad_probes.exe"
python $audit loads $scene "$work/replay" "$work/report"
python $audit peaks $scene $work "$oldBin/nc_cad_probes.exe" "$bin/nc_cad_probes.exe" "$work/report"
python $replay windows $scene "$work/replay" --exe "$bin/nc_cad_probes.exe" --old-exe "$oldBin/nc_cad_probes.exe" --old-indexed --mesh-exe "$bin/nc_mesh.exe"
python $run "$bin/nc_contact.exe" $scene "$work/repeat-new" --method cad-patch @settings
python $run "$oldBin/nc_contact.exe" $scene "$work/repeat-old" --method cad-patch @settings
python $audit repeat-report $scene $work "$work/report"
python $audit report $scene $work "$work/old-replay" "$work/replay" "$work/report"
& "$bin/nc_cad_solid_tests.exe"
& "$bin/nc_cad_solid_tests.exe" --chen-2026
```

若复算仅启用距离剪枝的消融，在单独检出的新版代码中用 `git restore --source 68c099c -- src/cad_contact_regions.cpp` 恢复旧射线实现，编译独立的 `nc_contact`。使用上述同一条 CAD 命令将输出写入 `$work/distance-only`，再运行 `repeat-report`。

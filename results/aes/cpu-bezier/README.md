# CPU 减速器的 Bézier 曲面接触对照

同一份两级减速器模型通过 `--method mesh-obb`、`--method cad-patch` 和 `--method cad-bezier` 切换几何后端。四个零件、初始姿态、驱动力矩、共享罚刚度、预测姿态更新和 HHT 设置相同。网格使用 Gmsh 三角片、OBB 搜索和已有的 contour-ray 接触区规则。CAD 两列使用相同的接触区搜索、分组和分布载荷规则。

本次借鉴陈等使用曲面控制点包围范围排除候选的思想，将其用于给定刚体姿态下的有限射线求交。每个零件保存正权重 Bézier 块的控制点及导数范围。两种方法分别处理的问题是运动曲面的连续碰撞时刻，以及承载期间当前姿态的接触深度、法向和力臂。[Chen 等，2024](https://xw-c.github.io/publication/siga24/) · [本次实现](../../../src/cad_bezier_ray.cpp)

## 同姿态回放

从已存档的五秒轴姿态中，每级选取八个窗口。每窗连续计算 100 个姿态，前 20 帧用于恢复接触历史，其余 80 帧计入查询时间。首次加载和 0.5 ms 力矩核对使用含预热帧的完整记录。每种方法单独运行。查询计时涵盖接触区发现及其力臂、深度的几何计算，进程耗时另含 STEP/Gmsh 读取和启动。网格报告未能确定的区域时保留缺失值。固定姿态比较使用的完整 STEP 参考力矩来自同一姿态下独立的接触区边界积分。

[回放计时与误差](replay.json) · [逐姿态结果](poses.csv) · [独立积分载荷](../cpu-cad-pressure/wrenches.csv)

| 同姿态几何查询 | 一级平均 | 二级平均 | 0.5 ms 二级轴力矩 |
| --- | ---: | ---: | ---: |
| Gmsh 三角网格 OBB | 3.87 ms | 14.88 ms | 411.951 N·m |
| 原 CAD | 4.04 ms | 10.13 ms | 533.953 N·m |
| 预处理 Bézier CAD（ours） | 3.90 ms | 6.67 ms | 533.953 N·m |

两种 CAD 在 1,600 个“级别与姿态”组合上的承载面对和连通分组逐一相同。两级每姿态平均候选面对分别为 8.02 和 22.41。跨全部 800 帧，Bézier 后端在一级访问了 7,044 个通过包围盒初筛的曲面块，在二级访问了 152,125 个。一级 OCCT 求交从 5,885 次降至 3,442 次，二级从 20,867 次降至 6,621 次。网格在二级有一个姿态无法确定接触几何。

能同时取得三种解及独立完整 STEP 边界积分的姿态，一级有 20 个，二级有 27 个。一级轴力矩相对独立积分的 RMS 差为 CAD 3.92 N·m、网格 5.93 N·m。二级为 CAD 8.45 N·m、网格 63.62 N·m。0.5 ms 的二级独立积分为 533.527 N·m。这些量比较给定姿态和给定载荷定律下的几何误差。

四个零件新增的 Bézier 表共 51.74 MB（十进制）。已读入 STEP 后首次提取并存储四张表合计 0.66 s，再次加载合计 0.37 s。包含 STEP 读取的独立进程分别耗时 6.61 s 和 6.55 s。这是在已有搜索与积分缓存之外单列的成本。

## 五秒闭环

三种方法各自从初始状态积分，主步长与输出间隔均为 0.1 ms。单进程时限 1,800 s。进程墙钟时间包括启动、读模、全部 50,000 个主步及响应输出。几何回放另计。图中的同姿态局部力矩与上方独立积分的五秒动力学轨迹使用不同输入。

Ryzen 9 5900X 上按 Bézier、网格、原 CAD 的顺序单独运行，同一次构建、相同可执行文件。三列的额外步内 CCD 次数均为零。加载和脱开时间按离散输出帧核对。

| CPU 方法 | 五秒闭环墙钟时间 | 相对网格耗时 |
| --- | ---: | ---: |
| Gmsh 三角网格 OBB | 313.88 s | 1.00 |
| 原 CAD | 806.39 s | 2.57 |
| 预处理 Bézier CAD（ours） | 646.76 s | 2.06 |

相对于原 CAD，Bézier 节省 159.63 s，降幅 19.8%。两个接触对可以并行，按对累计的分项耗时存在重叠。粗筛从 286.58 s 变为 283.21 s，局部搜索从 289.82 s 变为 173.66 s，接触区分组从 237.47 s 变为 205.49 s，面积积分从 196.62 s 变为 148.57 s。两种 CAD 共 50,001 个输出时刻、两级啮合的受载状态全部一致。输出轴力矩 RMS 差为 0.106 N·m，最大绝对差为 10.266 N·m，出现在 2.5377 s。两级法向载荷的 RMS 差分别为 1.348 N 和 1.698 N，最大绝对差分别为 212.47 N 和 163.16 N。

三种方法在两级的首次有载输出均为 0.3 ms。二级的有载/空载状态切换次数为 CAD 两种各 3 次、网格 5 次。相对于已存档的同网格 RecurDyn 响应，整段输出力矩 RMS 差为原 CAD 42.164 N·m、Bézier 42.164 N·m、网格 29.142 N·m。固定姿态的完整 STEP 积分和独立闭环的商业模型采用各自的比较口径。[RecurDyn 模型与数据](../cpu-foundation/reducer-physics/README.md)

补记曲面块访问次数的构建再次从初始状态独立推进 5 s，用时 661.29 s。与第一次 Bézier 闭环相比，两级加载状态逐帧一致，输出轴力矩 RMS 差为 0.107 N·m，最大绝对差为 10.266 N·m。分别保存两次可执行文件和输出的 SHA-256，以区分运行条件。[复跑记录](cad-bezier-repeat-run.json)

在 2.5377 s 输出力矩差最大处，固定原 CAD 闭环的轴姿态，两种算法都得到 149.468 N·m。固定 Bézier 闭环的轴姿态，两者都得到 138.336 N·m。两种姿态下均识别出三个接触区。4.6016 s 的一级载荷峰差也作了相同交叉核验。力矩对两条轨迹的姿态差敏感，在同一姿态使用两种几何算法所得结果一致到所示精度。回放从已接受的输出姿态重算，当步动力学力矩还包含预测姿态更新。[峰差交叉回放](peak-poses.json)

[完整响应图](reducer-cpu.png) · [数值响应](responses.npz) · [全程误差与接触状态](closed-loop.json)

## 复算

使用启用 OCCT 和 CPU 网格后端的 Release 构建，准备[原减速器模型](../cpu-cad-pressure/pruning/README.md)并设置运行所需 DLL 路径。下列命令在仓库根目录运行，`$bin` 指向可执行文件目录，`$scene` 指向相同的 `common-200.scene`。运行目录应为新目录。

```powershell
$env:OMP_NUM_THREADS = "2"
$env:NC_CAD_CACHE_DIR = "$PWD/build/reducer-geometry-cache"
python benchmarks/aes/cpu_bezier_replay.py $scene $bin build/bezier-replay-new
foreach ($method in @("cad-bezier", "mesh-obb", "cad-patch")) {
    python benchmarks/aes/cpu_backend_run.py "$bin/nc_contact.exe" $scene "build/bezier-runs-new/$method" --method $method --scope contour-ray --solver penalty --penalty-stiffness shared --geometry-update predictor --spectral-radius 0
}
python benchmarks/aes/cpu_bezier_results.py $scene build/bezier-replay-new build/bezier-runs-new results/aes/cpu-bezier --probes "$bin/nc_cad_probes.exe"
```

可在运行前通过 `nc_cad_probes --prepare-parts part1.step part2.step ... --bezier` 为各零件建立持久表。回放计数依次为 Bézier 排除的逐面查询、定位在面内部的命中、调用 OCCT 的次数、通过包围盒初筛并访问的 Bézier 块数。最后一项统计访问次数，同一个块可能被多个查询访问。有限射线的误差界使用双精度数值估计。需要处理的切边、多根或病态情况交回原生几何内核。

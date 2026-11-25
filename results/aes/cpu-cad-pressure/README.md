# 两级减速器的 CPU 接触区合力与力矩

两级减速器采用原生 STEP 和 65,732 片 Gmsh 三角网格分别计算。两次计算使用同一版 CPU 求解器、三轴惯量、初值、驱动、制动、共享罚刚度、预测姿态更新与隐式积分系数（高频谱半径 0），摩擦和材料阻尼为零。主步长与输出间隔均为 0.1 ms，覆盖 0–5 s，每条响应含 50,001 帧。CPU 为 Ryzen 9 5900X。两种闭环依次单独运行，均设置两个 OpenMP 工作线程并行计算两级。CAD 面对和连通性检查另由 OCCT 线程池调度，每个刚体对最多四个工作线程。每个进程限时 1,800 s。[输入与 RecurDyn 模型](../cpu-foundation/reducer-physics/README.md) · [同姿态输入](../cpu-local-cad/trajectory.npz)

## 实际动力学耗时与全程响应

| 同一 CPU 求解器 | 进程总耗时 | 准备 / 积分（s） | 整机力评估 | 输出轴扭矩均值（N·m） | 对同网格 RecurDyn 的全程扭矩 RMS 差（N·m） |
|---|---:|---:|---:|---:|---:|
| 纯网格 OBB + 三角面接触区 | 149.28 s | 2.14 / 144.06 | 151,547 | 137.80 | 29.14 |
| 原生 CAD 区域压力（ours） | 464.43 s | 23.08 / 436.52 | 186,230 | 137.80 | 42.16 |
| 同网格 RecurDyn 参考 | — | — | — | 138.07 | — |

CAD 运行保留了 22 分 16 秒的预算余量，用时为网格的 3.11 倍。两条轨迹各自积分到 0.5 ms，CAD 为 433.27 N·m，网格为 395.37 N·m，RecurDyn 为 440.83 N·m。分成五个 1 s 区间，CAD 对 RecurDyn 的扭矩 RMS 差每段都高于纯网格，说明启动一帧的改善无法代表整段。RecurDyn 使用相同 Gmsh 网格，是同等建模的软件参考。纯网格本次输出与原有 CPU 基线的 CSV 哈希完全相同。当前两组闭环均未启用独立步内 CCD。[0–5 s 响应曲线](closed-loop-5s.png) · [完整误差](closed-loop-errors.json) · [CAD 记录](cad-run.json) · [网格记录](mesh-run.json) · [CAD 响应数组](closed-loop-cad.npz) · [网格响应数组](closed-loop-mesh.npz)

面积积分的首个闭环版本在计时期间经历 CPU 频率下降，用时 1,408.43 s。最终版本引入组内并行和空候选的实体核验。该版本用时 464.43 s。两次进程计时同时受实现与 CPU 运行状态影响。交替采样的几何计时见下节。[首版闭环记录](serial-cad-run.json)

几何采用预测姿态重查，隐式迭代按当前刚体姿态更新力臂、法向与压力力偶，接触点脱开时触发重查。两级分别查询并行汇总。在 CAD 闭环中，每对接触单独累计的粗筛、局部搜索、连通分组和面积积分分别用时 102.48、210.43、131.89 和 100.25 s，其中局部搜索包含空候选时的原 STEP 分离核验。两级并行，四项按刚体对累加。50,151 次并行批次的实际墙钟为 362.55 s，另有 1,691 次单对查询在批次以外执行。前 2 ms 的启动阶段已用去 96.13 s 积分墙钟，局部搜索累计 108.50 s，解释了完整闭环中这项开销偏高的原因。[分阶段累计计时](closed-loop-profile.json)

## 在相同姿态上比较几何查询

旧版最费时的是二级斜齿轮的连通分组。原始整段回放中，该级粗筛、局部曲面搜索、连通分组分别需要 243.69、429.65、483.60 s。新版以两棵包围盒树一起遍历，保留原曲面参数与上一姿态的命中面。分组时用曲面控制点给出的包围盒和法向区间排除无法相交的面块，复用已验证的射线段，剩余连通性检查在 CPU 上并行。对多项式双三次曲面，提前计算每一节点区间的位置与导数系数。两级使用同一套代码。

| 50,001 个共同姿态 | 粗筛（s） | 局部搜索（s） | 连通分组（s） | CAD 进程总耗时（s） | 同次网格进程总耗时（s） |
|---|---:|---:|---:|---:|---:|
| 第 1 级，优化前 | 291.55 | 184.70 | 18.95 | 575.77 | — |
| 第 1 级，优化后（ours） | 49.54 | 44.06 | 3.13 | 132.62 | 48.19 |
| 第 2 级，优化前 | 243.69 | 429.65 | 483.60 | 1179.33 | — |
| 第 2 级，优化后（ours） | 69.62 | 78.75 | 111.52 | 273.73 | 128.62 |

两级各用一个进程同时回放，较慢一级从 1,179.33 s 降至 273.73 s。查询阶段用时不含导入和 JSON 输出。逐帧对比两级共 100,002 个级别姿态，旧版与新版的命中面组合、连通区归属和区数均一致。相同的有限候选仍可能有共同漏检，原 STEP 独立核验另见下节。纯网格的二级回放在 29.7 ms 有一次未能给出区域，闭环求解器可通过子步重试。[完整分项及可执行文件哈希](full-replay.json) · [各姿态分项耗时](geometry-times.npz)

为减小 CPU 频率随时段变化的影响，另取每级 8 段连续 100 姿态，三种程序交替计算 10 个姿态，各段剔除前 20 个预热姿态，每级共计 640 个。二级每姿态平均用时从 62.73 降至 13.73 ms，其中粗筛 12.33 → 3.28 ms，局部搜索 21.46 → 4.19 ms，分组 28.95 → 6.26 ms。一级合计 10.16 → 2.93 ms。相同窗口的纯网格查询分别用时 3.95 和 25.23 ms（一级、二级）。网格查询包含接触深度，CAD 窗口只计搜索和分组，整机性能以独立动力学进程为准。[粗筛、搜索和分组耗时图](search-and-grouping.png) · [交替计时及 P95](paired-window-timing.json) · [最初只加双树等改动的结果](intermediate-geometry.json)

## 0.5 ms 时的合力和力矩

固定在已归档网格轨迹的同一轴姿态上，比较二级齿轮传给输出轴的力矩。对照在整实体上做 STEP 布尔求交，逐一找到相连的实体重叠区，沿每片来源面的裁剪边界进行积分。参考和待测方案独立构造接触区并积分，使用相同的区域中心弦长定义及共享罚力定律。网格和 CAD 局部积分则各自从几何搜索开始。[逐帧载荷、三维合力和三维力矩](wrenches.csv)

| 相同轴姿态下的方法 | 各区法向力标量之和（N） | 三维合力大小（N） | 输出轴力矩（N·m） |
|---|---:|---:|---:|
| 纯网格 OBB | 6,632.61 | 6,632.47 | 411.95 |
| 旧 CAD，每区最深点 | 8,517.69 | 7,417.07 | 377.43 |
| 局部 CAD 接触区积分（ours） | 8,599.75 | 8,599.74 | 533.95 |
| 完整 STEP 接触区边界积分对照 | 8,614.08 | 8,613.92 | 533.53 |

在二级 22 个三种方法均有结果的有载姿态上，输出轴力矩相对完整 STEP 对照的 RMS 差为旧 CAD 128.29 N·m、网格 73.86 N·m、区域积分 9.37 N·m。区域积分覆盖了全部 23 个有载审计姿态，全 23 帧的 RMS 差为 9.16 N·m。将面积积分固定为每面 96 × 96 格，23 帧 RMS 差降到 0.44 N·m。闭环计时使用按需升阶的在线规则。

| 0.5 ms 处各求解器自己的已接受轴姿态 | 积分输出 | 同姿态重查纯网格 | 同姿态重查局部 CAD 面积分 | 同姿态完整 STEP 边界积分 |
|---|---:|---:|---:|---:|
| CAD 轨迹 | 433.27 | 325.46 | 437.07 | 437.00 |
| 网格轨迹 | 395.37 | 411.95 | 533.95 | 533.53 |

表中均为 N·m。每一行的三次几何重查使用同一对刚体的已接受变换。二级主动轮和输出轮在两条轨迹上的转角分别相差约 89 和 117 µrad，相应的完整 STEP 力矩从 533.53 变到 437.00 N·m。在 CAD 自己的姿态上，用局部面积积分重新计算得到 437.07 N·m，与完整 STEP 相差 0.07 N·m。求解器该步实际输出为 433.27 N·m，与重新计算相差 3.80 N·m。[两条轨迹的姿态、重查力矩与合力](accepted-0p5ms.json)

每个区域采用均匀法向压力。总法向力沿用共享罚刚度法则，由区域中心弦长决定。区域方向和力矩来自实际承载面上的有向面积和面积一阶矩。以前取每区最深点施力，会把局部法向用于整个区域，几处力矢量相加后出现过大的抵消。单独比较完整 STEP 的两种取矩方式，体积质心集中载荷给出 533.26 N·m，分布压力给出 533.53 N·m。这 0.27 N·m 量出了该姿态附加力偶本身的影响。旧算法约 156 N·m 的偏差主要来自受力区域代表方向、位置和深度的选择。

在线积分从每面每个参数方向 12 格起步，遇到无法取得有效区域时升至 24、48、96，再失败则对整个 STEP 求交。粗规则一旦有解就可以返回，有限面积采样没有先验精度界。空候选需由实体包围盒或原 STEP 距离查询确认分离。发现至少一个区域时，仍可能漏掉另一块很小且分离的承载区。独立核验了一级 24 个、二级 32 个姿态，抽样包括启动、接触切换、峰值和全程各处。均匀压力是本次对照的建模假设。[同姿态合力与力矩图](same-pose-wrenches.png) · [逐方法匹配数与误差](wrench-errors.json)

在解析双球碰撞中，原 STEP 曲面的参数极点落在接触位置，单靠有限起点会漏检。用整实体复核空结果后，0.1 ms 步长给出峰值 22.4991 N，解析峰值为 22.5 N。轴上中心弦恰好落在退化边时，后端用两个半径的对称邻域核对深度。[双球力误差](sphere-impact.json)

## 复算

从仓库根目录构建带 OCCT 与 CPU 网格支持的 Release 版，将相关 DLL 加入 PATH。下例中 `$bin` 为可执行文件目录，`$oldBin` 为提交 `f05a646` 的 Release 可执行文件目录。需要严格复算旧基线时使用该提交。回放顺序固定，每个进程均有 1,800 s 限时。

```powershell
python -m zipfile -e results/aes/cpu-foundation/reducer-physics/cpu-deck.zip build/replay-model
$scene = "build/replay-model/reducer-physical/common-200.scene"
$work = "build/pressure-repeat"
$env:OMP_NUM_THREADS = "2"
$replay = "benchmarks/aes/cpu_cad_pressure.py"
python $replay prepare $scene $work
python $replay old $scene $work --exe "$oldBin/nc_cad_probes.exe"
python $replay indexed $scene $work --exe "$bin/nc_cad_probes.exe"
python $replay mesh $scene $work --exe "$bin/nc_mesh.exe"
python $replay pressure $scene $work --exe "$bin/nc_cad_distance_benchmark.exe"
foreach ($order in @(0,24,48,96)) {
    python $replay area $scene $work --order $order --exe "$bin/nc_cad_probes.exe"
}
python $replay windows $scene $work --exe "$bin/nc_cad_probes.exe" --old-exe "$oldBin/nc_cad_probes.exe" --mesh-exe "$bin/nc_mesh.exe"
$settings = @("--solver","penalty","--penalty-stiffness","shared","--geometry-update","predictor","--spectral-radius","0")
python benchmarks/aes/cpu_backend_run.py "$bin/nc_contact.exe" $scene "$work/cad" --method cad-patch @settings
python benchmarks/aes/cpu_backend_run.py "$bin/nc_contact.exe" $scene "$work/mesh" --method mesh-obb --scope contour-ray @settings
python benchmarks/aes/cpu_cad_pressure_report.py $scene $work "$work/report" --old-replay $work --cad-csv "$work/cad/response.csv" --mesh-csv "$work/mesh/response.csv" --startup-binaries "$bin/nc_cad_probes.exe" "$bin/nc_cad_distance_benchmark.exe" "$bin/nc_mesh.exe"
```

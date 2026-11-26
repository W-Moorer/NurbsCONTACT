# 只读编译零件的 CPU CAD 接触动力学

同一两级减速器使用 0.1 ms 主步长和 5 s 总时长。四个刚体通过两个啮合副、三根转轴相连。两边使用原场景的初值、驱动与载荷、共享罚刚度、预测姿态更新、HHT 谱半径 0 和两个 OpenMP 线程。比较使用接受主步上的 50,001 帧。每个动力学进程的限时为 1,800 s。

## 仿真耗时

表中的时间取求解器 `statistics.wall_s`，从开始积分计到最后一个主步结束，包含步内接触检测、求力、HHT 迭代和逐步进度记录。离线预编译、模型初始化和积分结束后的响应文件导出均在计时区间之外。

### 本轮顺序复测

在同一台 Ryzen 9 5900X 上顺序运行下列三个程序。每次从原始初态开始，使用相同的逐步输出设置和每 3 s 采样的进程监测器。计时期间未另开项目编译、几何诊断或其他仿真。两套网格程序均取自原实验保存的 Release 可执行文件。

| CPU 后端（按运行顺序） | 5 s 完整仿真（s） | 同一可执行文件的历史仿真（s） |
| --- | ---: | ---: |
| [纯网格 OBB，当前基线程序](simulation-timing/mesh-current-run.json) | 310.62 | 315.33、426.76、431.42 |
| [预编译 CAD（ours）](simulation-timing/cad-compiled-run.json) | 282.76 | 396.40 |
| [纯网格 OBB，历史最快程序](simulation-timing/mesh-archived-run.json) | 293.94 | 130.01 |

与本轮较快的网格相比，CAD 少用 11.18 s（3.8%），两者均在 300 s 以内。每个程序各复测一次。CAD 完整响应与前次 CAD 逐字节相同，两套网格也都重现了历史网格的完整响应。本轮两种方法耗时接近。跨批次最快的网格仍为 130.01 s。

### 历史记录与耗时波动

| 首轮验收中的 CPU 后端 | 5 s 完整仿真（s） |
| --- | ---: |
| 编译 CAD（ours） | 396.40 |
| 纯网格 OBB，两次 | 426.76、431.42 |
| 原 CAD，历史同场景参考 | 583.59 |

过去常引用的 294.90 s 是进程总耗时，当次实际仿真为 283.41 s。更早两个构建的实际仿真为 130.01 s 和 144.06 s，均给出相同的完整网格响应。[295 s 对应的原始记录](../recurring/mesh-obb-run.json) · [130 s 记录](../../cpu-cad-pressure/pruning/mesh-run.json) · [144 s 记录](../../cpu-cad-pressure/mesh-run.json)

首轮验收期间，工作站上还有项目编译和几何诊断。与同一个网格可执行文件较早的记录相比，仿真从 315.33 s 增至 431.42 s，其中接触检测从 308.48 s 增至 421.04 s，占新增耗时的 97%。两次都执行了 420,692,832 次三角片检验、151,547 次求力和 25 次步长细分，50,001 帧响应逐字节相同。这把耗时变化定位到了同一批几何计算的执行速度，现有计数器无法单独量出后台调度、核心频率各贡献多少。[同一程序较早记录](../uv-relocation/mesh-obb-run.json) · [首轮较慢记录](mesh-run.json)

将跑出 130.01 s 的程序原样复测，接触检测从 127.42 s 增至 287.51 s，解释了 163.93 s 仿真差额中的 160.09 s。按仿真时间每 0.5 s 分段，十段各自执行相同次数的三角片检验，本轮各段用时为历史记录的 2.07–2.58 倍。耗时差异贯穿全程。[分段原始计数](simulation-timing/mesh-intervals.json) · [本轮监测](simulation-timing/mesh-archived-telemetry.json)

## 模型与运行依赖

OCCT 用于离线 STEP 导入、曲面分段、提取裁剪线与拓扑。四个零件编译为 840 张面、56,316 个解析或有理 Bézier 曲面块，文件合计 71.38 MB。仿真初始化时建立索引与曲面控制范围。零件文件中只存材料坐标几何。动力学程序运行目录中只有原场景、编译零件和可执行文件作为输入。Windows 可执行文件导入表及计算中实际加载的模块均未出现 OCCT。模块采样时工作集为 582 MiB。[零件清单](offline-parts.json) · [隔离运行核验](module-audit.json) · [可直接解压的模型](reducer-inputs.zip)

最终版对承载区积分复用带位移余量的候选表。两级接触对的积分累计用时从 384.27 降到 257.31 s。两级可并行，整机加速按积分循环的墙钟衡量。两次 CAD 的完整响应文件 SHA-256 相同。[优化前](first-compiled-run.json) · [优化后](compiled-run.json)

## 力矩和接触切换

两级正法向载荷与零载荷状态分别有 0、0 帧与原 CAD 不同。相对原 CAD，输出轴接触力矩全程 RMS 差为 0.00893 N·m，最大绝对差为 0.8761 N·m，位于 4.9649 s。0.5 ms 输出轴力矩为 432.6308 N·m，原 CAD 为 432.6308 N·m。在此前有尖峰的 2.5377 s 前后各 20 帧中，两种 CAD 的最大绝对差为 4.31e-08 N·m。相对 RecurDyn 归档曲线，本次 CAD 与网格的输出力矩 RMS 差分别为 42.095 和 29.142 N·m。[六幅响应图](responses.png) · [逐帧曲线](responses.npz) · [比较指标](comparison.json)

另用原有八段 100 帧轨迹在相同姿态下比较实际承载面身份。两级分别包含 163 和 142 次面组切换，各有 0 和 0 帧面组差异。另一组 80 个启动及切换姿态的最大轴力矩差为 1.21e-08 N·m。前一组使用原有固定轨迹，后一组来自已接受的 CAD 动力学姿态，均独立于本次闭环积分。[逐次切换帧号](face-switches.json) · [相同姿态载荷](same-poses.json)

全程最大差异附近两种积分轨迹在第 49,646 步采用了不同的子步划分。从各自接受的状态统一用 0.05 ms 子步复算四步，末端力矩差为 7.16e-07 N·m。[步长复算](peak-replay.json)

## 极窄承载区的独立检查

在第二级的四个真实姿态，另对原 STEP 做整实体布尔求交并求压力合力矩。下表的区域数依次对应生产搜索、额外加密矩形曲面边与角的诊断搜索、原 STEP 整实体。所有力矩使用同一共享刚度法则。

| 时间（ms） | 三种搜索的区域数 | 生产 CAD（N·m） | 边角加密（N·m） | 整实体（N·m） |
| ---: | --- | ---: | ---: | ---: |
| 1.3 | 3 / 3 / 3 | 153.191 | 153.191 | 156.799 |
| 1.6 | 3 / 4 / 4 | 347.129 | 262.713 | 279.076 |
| 274.1 | 2 / 3 / 3 | 194.480 | 138.761 | 138.652 |
| 931.7 | 2 / 3 / 3 | 201.382 | 141.919 | 141.604 |

生产方法与历史 CAD 在这些姿态均给出相同区域数。原实体求交在三个姿态另有一处浅接触，其深度分别为 0.929、2.699 和 2.216 µm。共享刚度按区域数分配，因此少一处浅区也会改变其他区域承载的份额。边角加密是对已发现问题的敏感性检查，表中给出它的单独结果。五秒生产轨迹沿用原搜索规则。对该减速器的响应复现，和对任意封闭 CAD 的几何完备性，是两个独立的验收问题。[载荷与深度明细](same-poses.json)

## 后端实现与复现

候选面可以来自通用 BVH 和上一姿态的已知面。接触深度及法向在裁剪曲面上求解。接近相切时，将 Bézier 控制网投影到查询线的法平面，在参数域内收紧可能交点的范围。积分块复用设置了覆盖范围，当前姿态的整块射线越出范围时重建候选，并在当前姿态重新求交。输出给动力学后端的区域包含位置、法向、深度和压力力臂。此处验证的是离散时间下有限侵入深度的接触。现有 STEP 转换器对未支持的曲面或周期样条显式报错。

从仓库根目录在 PowerShell 运行下列命令即可用无 OCCT 构建复现 CAD 闭环。模型压缩包保留场景本来的相对目录。

```powershell
cmake -S . -B build/standalone -DNC_WITH_OCCT=OFF -DNC_WITH_CPU_MESH=OFF -DBUILD_TESTING=ON
cmake --build build/standalone --config Release --parallel
ctest --test-dir build/standalone -C Release --output-on-failure
Expand-Archive results/aes/cpu-bezier/standalone/reducer-inputs.zip build/standalone-reducer
$env:OMP_NUM_THREADS="2"
$scene="build/standalone-reducer/scene/common-200.scene"
$exe="build/standalone/Release/nc_contact_runtime.exe"
python benchmarks/aes/cpu_process_timing.py cad-compiled $scene build/standalone-duty $exe
```

从原始 STEP 编译其他受支持的零件时，将带 OCCT 的构建配置中的 `nc_compile_part` 指向输入 STEP 和输出零件路径。在保持同一场景的情况下，将优化 Gmsh 资产放回原基线目录并运行 `cpu_process_timing.py mesh-obb` 可复现纯网格。运行 `cpu_compiled_reducer.py compare` 可从两份完整运行和已存档的原 CAD 运行重新生成图表。其 `fixed`、`switches` 和 `modules` 子命令分别检查同姿态力矩、实际承载面身份和 Windows 运行依赖。[复现实验脚本](../../../../benchmarks/aes/cpu_compiled_reducer.py) · [独立构建测试](standalone-tests.txt)

# 只读编译零件的 CPU CAD 接触动力学

同一两级减速器使用 0.1 ms 主步长和 5 s 总时长。四个刚体通过两个啮合副、三根转轴相连。两边使用原场景的初值、驱动与载荷、共享罚刚度、预测姿态更新、HHT 谱半径 0 和两个 OpenMP 线程。比较使用接受主步上的 50,001 帧。每个动力学进程的限时为 1,800 s。

## 整机耗时和输入

| CPU 后端 | 进程墙钟（s） | 其中初始化（s） | 积分（s） |
| --- | ---: | ---: | ---: |
| 编译 CAD（ours） | 411.25 | 4.85 | 396.40 |
| 纯网格 OBB，复测 | 449.03 | 3.64 | 431.42 |
| 原 CAD，历史同场景参考 | 644.07 | 49.70 | 583.59 |

进程墙钟包括模型加载、动力学积分和写出响应。最终 CAD 耗时为复测网格的 0.916 倍。第一轮网格和引入跨姿态积分块缓存前的编译 CAD 分别为 445.16 s 和 517.66 s。各动力学进程依次运行，两轮网格得到相同的响应文件哈希。另附主机遥测以便复核计时波动。[计时及输入哈希](timings.json) · [最终 CAD](compiled-run.json) · [网格](mesh-run.json) · [原 CAD 记录](original-run.json)

OCCT 只用于离线 STEP 导入、曲面分段、提取裁剪线与拓扑。四个零件编译为 840 张面、56,316 个解析或有理 Bézier 曲面块，文件合计 71.38 MB。四个编译进程共用 82.81 s，仿真初始化另外建立索引与曲面控制范围。零件文件中只存材料坐标几何。动力学程序运行目录中只有原场景、编译零件和可执行文件作为输入。Windows 可执行文件导入表及计算中实际加载的模块均未出现 OCCT。模块采样时工作集为 582 MiB。[零件清单](offline-parts.json) · [隔离运行核验](module-audit.json) · [可直接解压的模型](reducer-inputs.zip)

最终版对承载区积分复用带位移余量的候选表。两级接触对的积分累计用时从 384.27 降到 257.31 s。两级可并行，这个差值不能直接作为整机节省的秒数。完整进程节省了 106.41 s，两次 CAD 的完整响应文件 SHA-256 相同。首次从 STEP 编译再运行一次共 494.05 s，表中的网格耗时从已有 Gmsh 模型开始。

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

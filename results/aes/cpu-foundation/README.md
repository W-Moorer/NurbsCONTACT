# 刚体网格接触的 CPU 基线

两级减速器的 5 s 同网格对照包括原始 RecurDyn 输出、制动峰值的步长试验、96 个高载荷姿态的分离检查，以及 GPU 上的三角形和法向力学算子验证。[同网格减速器实验](reducer-physics/README.md) · [候选面片跨步复用实验](temporal-reducer/README.md)

完整运行表见 [RESULTS.md](RESULTS.md)。[解析基准](analytical.png)、[凸轮响应](cam-response.png)、[偏心碰撞](spatial-impact.png)、[齿轮响应](gear-response.png) 和 [减速器共同区间](reducer-prefix.png) 均由已接受的状态绘制。动画采用原网格与计算得到的姿态。[凸轮动画](cam.mp4) · [齿轮动画](gear.mp4) · [两级减速器已完成区间的动画](reducer-prefix.mp4)

同一个 `.scene` 保存质量和惯量、自由度、初始状态、载荷以及材料。两种 BVH 接收相同的 Gmsh 表面网格。`--method bvh-aabb` 与 `--method bvh-obb` 直接选择两种有限三角形接触基线，`--solver penalty` 选取罚力和 HHT 隐式积分。记录中的等价长写法是 `--method mesh-aabb|mesh-obb --mesh-region-scope finite`。工业测试与基础测试调用同一个 `nc_contact`。

## 与本地参考实现的对应关系

本项目的 C++ 基线提交为 `97ba52e`，接触区累加顺序修复为 `b4ddcc2`。实验记录逐次保存可执行文件、场景和网格的 SHA-256。为兼顾在途长时计算和编译，开发期间的前后对照使用了独立的 Release 可执行文件。

- NexDyn 源码版本为 `aaac2df14037bf4687c8d1974d757c78923e950e`。`src/Contact/Algorithms/Generic/GenericMeshContactPipeline.cpp` 展示单次查询的 AABB/OBB、Möller 三角形相交、交线图与区域划分。跨步逻辑要看 `src/Contact/Runtime/Geometry/TriMeshContactInteraction.cpp`。在该运行时启用时间相干分支时，第 4274 行先尝试沿用历史，再按需执行完整粗检测。第 4831 行合并上一次活动面片对和候选面片对，邻面扩展在插入时按两个面片编号去重。位移、候选数量与接触模式共同限制复用。第 8715 行附近保存活动面片和接触点，第 13134、13207 行附近分别保存和恢复试步历史。
- 本后端在固定 Gmsh 拓扑上按面片对编号去重，再扩展一圈邻面。每个连通接触区单独收集唯一的顶点—面和边—边组合。当前真实相交面片组成的区域与上次完全一致时，直接复用其拓扑，按当前姿态重新计算法向、深度和力矩。外围候选面片有运动余量，当前相对位姿相对建表姿态的位移界一旦耗尽就从 BVH 补全。HHT 回退或反向运动同样检查位姿。此设计保留本项目的接触深度定义，NexDyn 的历史接触点回放和罚力模型有各自的实现。
- 对外侧局部接触，BVH 分别找出两侧可能进入接触区域的面片，只缓存这两份面片编号。在运动余量内，先按当前姿态收紧区域，再扫描局部面片包围盒，最后做有限三角形细检测。间距与投影范围不适合这种筛选时，直接遍历两棵 BVH。三角形在投影中有明确分离时提前剔除，临界情况仍交给几何谓词。
- 启用 OpenMP 的 CPU 构建可并行查询相互独立的有限网格物体对。共享同一个可变查询对象时采用串行。总接触力、力矩和 HHT 矩阵始终按原来的物体对与接触区顺序组装。设置 `OMP_NUM_THREADS=1` 可以复现串行计算。并行时 `mesh_detection_s` 记录等待整批接触几何完成的墙钟时间。
- Drake 源码版本为 `707bed92ece0f92b859a198fd4ad6ef464e62d20`。`multibody/plant/sap_driver.cc` 准备自由运动速度和接触问题，`multibody/contact_solvers/sap/sap_solver.cc` 求速度空间的凸优化问题，`sap_friction_cone_constraint.cc` 处理摩擦锥投影与导数。本项目的 `--solver sap` 通过相同几何接口取得接触点，组装正则化摩擦锥问题，用 Newton 方向和代价线搜索求解离散冲量。`--method mesh-sap` 中的 SAP 指粗检测的 sweep-and-prune，两个选项各有独立作用。

## 每个隐式试步计算什么

1. 预处理封闭表面的一致朝向、面法向、有限边、面邻接关系和 BVH。以当前刚体姿态更新包围盒，逐层排除无法接触的节点。复用上一次的三角形对时，重新检查它在当前姿态是否相交。
2. 对到达叶节点的候选三角形做相交检测。凸体的顶点—面接触要求投影落在有限面内，边—边接触要求公垂线的两个端点都在线段上，并核对局部支撑方向。一般凹网格通过实际相交的三角形和公共边识别局部接触区域。对于投影完全落在支撑面内的接触，可直接在网格上确定每个连通压入区域的最深点。
3. 几何查询返回接触点、法向、深度。材料定律将其转换成法向力以及作用在刚体上的广义力矩。HHT 在试算的姿态和速度上更新接触，核对动力学残差、收敛与穿透限值后接受或细分。表中的 0.1 ms 是主步长，细分次数随每条运行记录提供。这批 HHT 长时测试使用离散接触判断，切换时间的采样分辨率取决于实际步长。

## 复现

从仓库根目录解压 `models.zip` 到 `build/foundation-models`，使用已有 CPU Release 构建及其运行库。下例在同一个模型上运行 5 s，输出 50,001 个状态。变更 `--method` 时沿用全部其余参数。

```powershell
python -m zipfile -e results/aes/cpu-foundation/models.zip build/foundation-models
build/cpu-swept/Release/nc_contact.exe build/foundation-models/cam/model/cam.scene 0 .0001 5 1 build/cam-obb --method bvh-obb --mesh-dir build/foundation-models/cam/aligned-96 --solver penalty
build/cpu-swept/Release/nc_contact.exe build/foundation-models/cam/model/cam.scene 0 .0001 5 1 build/cam-aabb --method bvh-aabb --mesh-dir build/foundation-models/cam/aligned-96 --solver penalty
python benchmarks/aes/cpu_foundation.py build/cpu-swept/Release/nc_contact.exe build/foundation-models/analytic build/foundation-rerun --cam-root build/foundation-models/cam --cam-mesh aligned-48
```

齿轮与减速器使用下列命令复现整段 5 s。运行器在 1,800 s 时终止该进程，保存最后一个完整主步。切换树时替换 `--method` 并使用新的输出目录。

```powershell
python benchmarks/aes/cpu_backend_run.py build/cpu-swept/Release/nc_contact.exe build/foundation-models/spur/spur.scene build/rerun-spur-obb --method mesh-obb --scope finite
python benchmarks/aes/cpu_backend_run.py build/cpu-swept/Release/nc_contact.exe build/foundation-models/reducer/reducer.scene build/rerun-reducer-aabb --method mesh-aabb --scope finite
```

单级齿轮模型共 20,456 个三角片，两级减速器共 65,732 个三角片。第一级两轮各有 10,216、10,240 片，第二级各有 27,934、17,342 片。第一级第二个齿轮的最小 Gmsh SICN 为 0.000537，存在狭长面片。逐体的最小角、最小 SICN、曲面采样误差与网格摘要见模型压缩包同目录的 `.quality.json`。目前没有单独量出狭长面片对减速器载荷误差的贡献。

下例在同一滑块场景切换为 SAP。第二条命令在安装了 pydrake 的 Python 环境中运行。

```powershell
build/cpu-swept/Release/nc_contact.exe build/foundation-models/analytic/impact.scene 0 .0001 .16 1 build/sap-impact --method bvh-obb --solver sap
python benchmarks/engines/drake_sap_impact.py build/sap-drake.json --native-method bvh-obb --native-geometry "two flat Gmsh sliders" --run .0001 build/sap-impact.csv
```

每次整机运行限时 1,800 s。耗时取模拟器进程从启动到退出，包含网格准备和逐步输出；Gmsh 离线划分、Python 的误差计算与绘图另计。`runs.json` 同时保存程序内部计时、主步数、载荷、几何查询计数和文件摘要。计时环境为 AMD Ryzen 9 5900X，各方案从相同初态独立推进，部分进程在同一工作站上并行运行。时间仅代表这台机器上的单次测量。

在减速器已接受的 1.2400–1.2599 s 上，每级回放 200 个姿态。第二级 OBB 单纯判断相交的中位数为 0.008 ms，返回完整算力几何的中位数为 32.3 ms。第一级相应为 0.017 ms 和 15.2 ms。回放只计单次几何查询，独立长时仿真的进程时间另列。[回放明细](contact-profile.json)

两级减速器最终 AABB 与 OBB 分别在 1.5106 s 和 1.7622 s 达到预算。双方共同完成的 15,107 个状态中，三根轴的角度和转速逐项相同。OBB 的 1,791.94 s 积分时间内，网格接触检测累计 1,784.89 s，约占 99.6%。这项计时包括遍历和生成完整接触区。

RecurDyn 原生模型的四个 `CSURFACE` 分别有 2,200、3,596、8,548、6,292 个接触面片，共 20,636 个。求解器日志记录了 4 个 CPU 线程、最大步长 0.1 ms、初始步长 0.001 ms。CPU 基线的 65,732 个三角片和串行接触求值构成两项实现条件差异。原始 OBB 长跑的前 400 主步积分用时 73.97 s，将同一未经计时改造的可执行文件单独运行，该段用时 28.66 s。核对了 401 个输出状态的三轴角度、速度以及三角片测试计数，结果全部相同。这两次的整机运行环境对绝对耗时有显著影响。

在相同初态的 400 主步诊断中，HHT 进行了 3,250 次力求值、47 次细分。两级几何合计耗时 25.74 s，邻面扩展与特征去重占 33.7%，候选面片表重建占 28.6%，已有候选的相交检查占 19.2%，首次相交检查占 3.2%。这四项是互斥计时，其余时间用于连通区域、深度界、接触点求取等。两级共生成 5,778 万个顶点—面与边—边特征组合，去重后剩 1,609 万个。在另一次 1.2400–1.2599 s 已接受姿态回放中，两级各自的 200 次查询都触发了候选面片表重建。闭环试算与已接受姿态回放的调用模式不同。[诊断计时与模型记录](reducer-cost-breakdown.json)

单级齿轮最初两种树检测到的接触几何相同，但多区域力矩的浮点累加顺序随树遍历而变。两条独立轨迹在 2.785 s 的法向合力相差 839 N。对差异出现和放大的五个窗口取 100 个相同姿态回放，40 个姿态仅有区域排列差异。现在按原网格三角形编号排列相交面片，再连接接触区。重复同姿态试验后，100 个姿态的接触点、法向、深度和排列全部相同。[修复前](order-before.json) · [修复后](order-after.json)

在固定顺序的完整 5 s 齿轮复算中，两种树独立产生的响应 CSV 字节完全相同。共记录 575 次整对齿轮法向合力在零和正值之间的切换。一个齿对卸载而另一个齿对仍然受力时，总合力可以始终为正，这项计数只反映整对刚体的承载状态。

## 参考量和数据

- 等质量滑块通过一对有限平面发生线性弹性正碰。用相对运动的半正弦力脉冲、解析脱开时刻和总动量核对动力学。
- 单边振子从正最大位移处静止释放，取质量 1、基础频率 1 rad/s、附加刚度 0.1 N/m、振幅 1 mm。每次切换用相邻异号位移线性插值。参考为 Junca 与 Rousselet 第 2.1 节的分段解及两个半周期。[原文](https://math.univ-cotedazur.fr/u/junca/Research/pdffiles/Anglet.pdf)
- 三维自由对称陀螺用 Euler 闭式解检验姿态、角速度、角动量和质心轨迹。这个算例检验积分及刚体运动学，两个接触体在全程分离。
- 偏心圆凸轮采用 105 mm 半径、24 mm 偏心及 60 mm 宽度。沿宽度对齐的两组 Gmsh 网格按这组原模型尺寸重建四段圆弧并挤出，分别取 48、96 个周向分段。从动件沿用同一份 12 片平面网格。凸轮转动与从动件平动共同求解。第一套参考使用光滑圆轮廓的支撑高度和力臂，以 DOP853 积分。第二套参考针对同一内接多边形，在每一角度取最高点和对应力臂，独立积分。第三项核查只在每个已计算姿态上，直接枚举真实 Gmsh 顶点取得支撑高度，与程序输出的接触力比较。三者分别检验建模总误差、对离散模型的动力学求解误差、同姿态几何载荷误差。
- 同一凸轮的非结构化表面网格（凸轮 1,414 片，从动件 12 片）在部分姿态上形成多个分离的压入区。每区一份刚度得到的法向合力，相对同姿态“最深处一份刚度”的模型最多增加 2.08 N。两种罚力建模的差异记录在 `unstructured_cam`。沿宽度对齐的网格用于隔离这一效应；复杂机构的接触力仍应单独核对。
- 工业齿轮的一对齿面可能同时分担载荷。运行记录给出法向合力和轴响应；5 s 内的平均传动比只能验证总体传动关系。减速器的 RecurDyn 参考来自同项目的原生 5 s 导出，对齐实际时间与各轴转向，只在双方共同完成的时间上计算偏差。
- 在减速器 102–114 ms 制动窗口，对已接受的 CPU 轴姿态重新查询各区域，将罚力矢量求和后取模，并对输出轴中心求力矩。与 RecurDyn 的 121 个同期采样相比，两级合力 RMS 偏差分别为 843 N、1,729 N，输出接触力矩 RMS 偏差为 107 N·m。轴速和瞬时载荷应分别判断。[制动载荷曲线](braking-wrenches.png) · [数据](braking-wrenches.json)
- Drake 1.58.0 的独立检验采用两个轴向约束球体，本站使用 Gmsh 平面滑块。两者具有相同的一维间隙、质量、法向刚度与无摩擦定律。这个检验专门覆盖 SAP 法向冲量更新。运行记录提供总冲量、峰值力和逐步位移速度差。

Python 从完整轨迹计算误差。用于快速重绘的 `responses.npz` 通常每 1 ms 保存一个点，正碰时段额外保留全部原始样本。齿轮归档以原主步保存。图片和动画直接取自已接受的仿真状态。

# 双级减速器的 5 s 纯网格基线

本页的 5 s 自由动力学曲线来自独立 GPU 求解器，CPU 三角片数据来自给定姿态回放。CPU 统一后端独立推进三轴的实测见[减速器 CPU 对照](../cpu-backend-long/README.md)。

四个齿轮、三根转轴、两处啮合。一级为 24/48 齿直齿轮，二级为 22/44 齿斜齿轮，两级共用中间轴。主步长 0.1 ms，50,000 个主步，从初态起保存 50,001 个状态。输入和制动载荷按模型原公式生成至 5 s，包含制动变化及 13、23 Hz 的周期激励。两处接触使用 1.6×10⁸ N/m 的线性法向刚度。四件共 65,732 个 Gmsh 三角形，最大采样曲面偏差分件为 48.1、84.9、27.1、48.1 µm。表面标签参与受力区域归并，在线查询使用平面三角形。[网格与哈希](meshes.json)

### 端到端动力学

同一 RTX 3070 上各配置分别运行一次。每步细分是接触时间分辨率，HHT 还会按收敛情况重试。在线耗时包括两级接触、接触力及三轴积分。设置耗时包含载入、构建索引和 GPU 初始化。离线建网、曲线导出与视频编码另计。工作站同时有 CPU 任务，表中时间为单次实测。

四个网格文件合计 4.36 MiB。GPU 内存池在投影筛选时占 187.29 MiB，分块时占 210.45 MiB。网格中最小 SICN 为 0.00054，存在很瘦的三角形；这批网格可用于下面的动力学和性能复测。各件的离散误差与质量记录在网格清单中。8、16 和 32 细分都没有达到每步 0.1 ms 的实时期限。

| 网格候选搜索 | 每主步接触细分 | 设置 (s) | 在线 (s) | 动力学试算 | 相对 32 细分，两级载荷 RMS 差 (N) |
|---|---:|---:|---:|---:|---:|
| 投影筛选 | 16 | 16.12 | 171.09 | 1,603,242 | 2.21 / 2.57 |
| 局部块复用，0.2 mm 余量 | 16 | 37.93 | 163.80 | 1,603,242 | 2.21 / 2.57 |
| 局部块复用，0.8 mm 余量 | 16 | 32.88 | 133.33 | 1,603,242 | 2.21 / 2.57 |
| 同上 | 8 | 36.02 | 80.11 | 917,405 | 12.56 / 14.75 |
| 同上 | 32 | 38.74 | 230.35 | 3,199,196 | 参照 |

**独立的 RecurDyn 也完成了相同的 5 s。** 四个原始 STEP 文件与 Gmsh 网格来源逐件哈希相同。RecurDyn 使用自身的默认接触面片、原生实体接触及其积分器，输入相同的轴惯量、支承和外载公式，最大步长及输出间隔均为 0.1 ms。模型准备 248.08 s，在线墙钟 1,044.39 s，求解器日志记录 CPU 1,004.75 s。GPU 的 8 与 16 细分配置分别花费 116.13、166.21 s 完成设置和在线计算。两平台的接触离散与力汇总方式不同。

在 50,001 个共同时间戳上，16 细分网格模型与 RecurDyn 的平均输出接触力矩为 137.805 与 137.947 N·m，三轴转速 RMS 差为 3.562、0.993、0.974 rad/s，两级传动误差 RMS 差为 0.939、0.884 mrad。输出轴瞬时接触力矩 RMS 差为 72.76 N·m。在制动加载窗口，RecurDyn 的轴速波动和力矩峰谷幅值更大。将时间细分从 8 加密到 32 后，力矩 RMS 差仅从 72.78 变为 72.76 N·m。下一项精度核对宜在相同轴姿态下单独比较两级传递力矩、接触点和力臂。[5 s 对照曲线](recurdyn-5s.png) · [全部指标与计时](recurdyn-5s.json) · [RecurDyn 原始时序](recurdyn-5s.npz)

16 细分下三组 50,001 × 14 的原始数组逐值相等。0.2 → 0.8 mm 后，四个搜索方向合计的候选块重建次数从 1,708,018 降到 464,411；候选准备从 59.47 s 降到 31.80 s，细查从 74.25 s 降到 69.72 s，接触汇总与积分从 29.52 s 变为 30.69 s。0.8 mm 是本例实测配置。8 细分的一级有 3 帧承载开关与 32 细分不同，二级一致；16 与 32 细分的两级开关逐帧一致。开关按每级法向合力大于零定义。上述指标描述整级承载，分辨单个齿对的交接还需要逐齿载荷记录。[运行数据](dynamics.json) · [细分对照时序](substeps.npz)

三轴全程平均转速为 188.496、94.248、47.124 rad/s。两级法向力峰值为 4.420、9.392 kN，输出轴平均接触力矩为 137.805 N·m，平均制动力矩为 137.806 N·m。另与历史 3 s RecurDyn 结果核对了 6,001 个共同时间戳，当时两边输出力矩均值为 138.96 与 138.03 N·m，瞬时力矩 RMS 差为 63.22 N·m。[历史 RecurDyn 导出](../resident-reducer/physical/native-exports/rd-physics.npz)

* [动力学图集 PDF](responses.pdf) 与 [全时程曲线 PNG](responses.png) 同时显示三轴转速波动、两级法向力及从动轴力矩。全景取每 10 ms 的均值和极值，旁边的局部图保留所有 0.1 ms 样本。[原始时序](responses.npz)
* [与 RecurDyn 的同时间窗对照](reference.png) 显示制动加载时的轴速、传动误差和输出力矩。[启动与接触切换](switching.png) 同时显示三角片相交、正法向载荷及 8/16/32 三种细分的载荷。
* [双级啮合动画 MP4](reducer.mp4) 使用四件实际网格的横截面与求解得到的轴角。先将整个 5 s 过程以 0.25 倍速播放，再以 0.005 倍速展示 100–120 ms 的制动加载，画面下方给出两级当时的法向力。
* [Blender 模型全景和两级啮合细节](assembly.png) 使用同一批 Gmsh 三角片与 112 ms 时的求解姿态。着色用来区分零件。

### CPU 三角片检测

在 Ryzen 9 5900X 上，以相同的 50,001 个轴姿态逐级回放，每种方法共 100,002 次查询。粗检测后用真实三角片相交判断并检查实体包含；上一姿态相交的面片对在当前姿态重新核验。这里只计离散姿态查询，查询不会推进轴状态。动态 AABB 树和 SAP 都在物体级筛选，进入物体内部后两者均走 AABB 层次树。两级分开跑、每次只提交两个物体，此工况无法衡量大规模物体群的 SAP 优势。

| CPU 原实现 | 一级检测 (s) | 二级检测 (s) | 检测合计 (s) | 两个进程合计含加载与 I/O (s) |
|---|---:|---:|---:|---:|
| AABB-BVH | 814.71 | 237.17 | 1,051.88 | 1,072.34 |
| OBB-BVH | 34.52 | 25.32 | 59.84 | 78.91 |
| SAP + AABB-BVH | 1,010.41 | 325.21 | 1,335.62 | 1,363.44 |

三种方法每帧的相交结果一致。一级 49,982 帧相交，逐帧对应正法向力。二级 49,978 帧相交，其中 4 帧的 GPU 采样法向力为零。两级所有正载荷帧都检测到了三角片相交。本组运行 AABB 与 SAP 的节点和三角片测试次数相同，单次耗时受共享 CPU 负载影响。OBB 两级合计访问 8,851 万个节点对、做 114 万次三角片测试；AABB 合计访问约 17.62 亿个节点对、做 6,894 万次三角片测试。[逐阶段计时与计数](cpu-detection.json) · [逐帧相交判定](cpu-overlap.npz)

原有 CPU 体积力路线在完成三角片检测后，还要重建两个网格交集的边界以计算力。早期 5 ms 诊断段中，111.28 s 积分耗时里，几何相交筛查为 4.30 s，重建布尔交体为 101.66 s。此短段只用于定位热点。GPU 基线复用近邻块，在候选三角形上双向查询并计算法向弹簧力，避免逐次构造交体。FCL 将碰撞查询与接触信息输出建模为不同的查询需求；CGAL 的体积布尔运算还要细分相交的曲面并构造输出网格。这两类工作的成本应分别报告。[CPU 原始剖析](../cpu-mesh/local-regions/reducer-adaptive-local-5ms.json) · [FCL 论文](https://gamma.cs.unc.edu/FCL/fcl_docs/webpage/pdfs/fcl_icra2012.pdf) · [CGAL 布尔运算](https://doc.cgal.org/6.1/Polygon_mesh_processing/index.html)

CPU 的优化试验沿两物体包围盒中心连线，用包围区间先排除不可能相交的子树，再扫描留下的三角片包围盒。该策略使用顶点与层次树，适用于一般三角网格。完整回放中所有相交标记保持一致。AABB 一级从 814.71 降至 132.17 s，二级却从 237.17 增至 434.89 s。给 OBB 加同样步骤使两级合计由 59.84 增至 652.40 s。因此 AABB/SAP 仅通过 `nc_mesh --method mesh-aabb --any --candidate-window ...` 显式启用该试验，OBB 保留双树遍历。[独立运行的节点、面片和计时数据](cpu-window-trial.json)

第二项优化利用相邻姿态的接触连续性。先在新姿态重测上次相交的两片三角形。若这对已经分开，检测两侧各三个共边邻面，最多再测 15 对。如果仍找不到相交证据，则从 BVH 根节点开始正常查询，保留实体包含判断。这条路径适用于任意受支持的封闭三角网格。V-Clip 已使用相邻特征跟踪凸多面体最近特征；这里使用同样的时序连续性思想，在三角网格上复用已证实的相交面片，最终判据仍是本姿态的三角片相交。[V-Clip 原文](https://graphics.stanford.edu/courses/cs448b-00-winter/papers/TR97-05.pdf)

新版 OBB 用 10.68 和 4.67 s 完成两级全部姿态，共边邻面分别命中 17,814、15,820 次。这些帧都跳过了整树遍历。相较于原版本，OBB 的访问节点对从 8,851 万降到 4,266 万。该版本还将双树遍历改在第二个刚体的局部坐标系中进行。使用 `--same-face-only` 可以在同一个二进制上单独关闭邻面跟踪。

启用相邻面跟踪、关闭候选窗口时，AABB、OBB、SAP + AABB 两级检测总耗时分别为 170.55、15.36、154.66 s。三者对 100,002 个姿态查询的结果均与原版本逐项一致。[完整运行的计时与命中计数](cpu-neighbors.json)

同一个最终版可执行文件又分别运行了两次完整 OBB 回放，仅改变 `--same-face-only`。只复查上一对面片的两级检测合计 22.00 s，启用共边邻面后为 11.16 s，缩短 49.3%。两次进程含加载和 I/O 分别为 29.20、17.87 s。所有相交标记都与原始完整回放一致。共享工作站上不同批次的墙钟耗时有波动，相同配置的节点数和面片数可逐项核对。[开关对照](cpu-obb-ablation.json) · [热点与整段计时图](timings.png)

独立的 MuJoCo 3.4.0 CPU 对照把同一批三角形构造成四个随三根转轴运动的刚性 flex 表面，三角表面半径为 2 µm。以 5 s 为目标，运行达到 1,800 s 上限，最后一条完整进度记录为仿真 0.8 s。MuJoCo 常规 mesh geom 使用凸碰撞，对整个非凸齿轮取单一凸包会填平齿槽。此处计时仅对应直接保留非凸三角面的配置。另一次本地试验尝试从相交面片直接估算穿透并向 SAP 提供冲量约束，在 3.574 ms 的试算中给出 175.1 µm 穿透，超过模型的 53 µm 范围。两项均未计入 5 s 动力学成功组。[运行记录](cpu-solver-trials.json) · [MuJoCo 碰撞几何定义](https://mujoco.readthedocs.io/en/3.4.0/computation/)

本地原有 CPU 闭环（mesh-obb、局部布尔交体、SAP 冲量求解）也从初态以同一主步长实际请求了 5 s，于 1,800 s 上限终止，未产出完整时程。[超时记录](cpu-solver-trials.json)

### 复现

从仓库根目录执行。CPU 回放基准的原版本为 `acd533f`，需要带 FCL/CGAL 的 `nc_mesh` 可执行文件。GPU 需要 CuPy/CUDA，作图与动画使用 Python、Matplotlib、trimesh、OpenCV 和 imageio-ffmpeg。

```powershell
python benchmarks/aes/prepare_spur_cpu.py build/cpu-contact/reducer-edge-compact --case reducer --mesh-profile compact
$work = 'build/long-mesh'
$meshes = 'build/cpu-contact/reducer-edge-compact'
python benchmarks/aes/long_mesh_deck.py $meshes "$work/deck"
function RunMesh($name, $phase, $guard, $samples) {
    python benchmarks/aes/resident_assembly.py "$work/deck" "$work/$name.json" --trace "$work/$name.npy" --method mesh --duration 5 --step-us 100 --broad-phase $phase --guard-mm $guard --stiffness-scope pair --ticks 4096 --contact-samples $samples --explicit-crossing --profile-phases
}
RunMesh loaded-projection projection 0.2 16
RunMesh loaded-blocks blocks 0.2 16
RunMesh wide-blocks blocks 0.8 16
RunMesh samples-8 blocks 0.8 8
RunMesh samples-32 blocks 0.8 32
python benchmarks/aes/long_mesh_replay.py "$work/deck" $meshes "$work/loaded-blocks.npy" build/cpu-swept/Release/nc_mesh.exe "$work/replay-before" "$work/replay-before.json"
python benchmarks/aes/long_mesh_figures.py $work results/aes/long-mesh --animate
```

Blender 使用自己的 Python 读取已保存的动力学姿态。模型渲染命令为 `blender -b --factory-startup --python-exit-code 1 --python benchmarks/aes/long_mesh_blender.py -- build/long-mesh/deck results/aes/long-mesh/responses.npz build/long-mesh/render`，随后重新调用绘图脚本即可输出全景和局部组合图。

用当前 `nc_mesh` 运行两次 `long_mesh_replay.py`，输出分别指定为 `paired-same.json`、`paired-neighbors.json`，两个命令均加 `--methods mesh-obb`，第一条命令再加 `--same-face-only`，即可重做邻面开关对照。原网格 MuJoCo 测试入口为 `python benchmarks/aes/long_mesh_mujoco.py build/long-mesh/deck build/long-mesh/mujoco.json`，含 30 分钟超时控制。

在有 ProcessNet 的 Windows 上，从同一模型目录复制 `assembly.npz` 和 `models.json` 至 `build/long-mesh/recurdyn`，运行 `python benchmarks/aes/recurdyn_reducer.py build/long-mesh/recurdyn build/long-mesh/recurdyn/run.json --duration 5 --step-us 100`。再用该脚本的 `--plot` 参数读取本次原生 `.rplt`，输出为 `build/long-mesh/recurdyn/physics.npz`，随后运行绘图脚本。

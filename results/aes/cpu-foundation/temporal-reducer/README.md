# 相邻姿态复用后的两级减速器 CPU 基线

四个刚性齿轮共 65,732 个 Gmsh 三角片。两个啮合级通过同一根中间轴交换力矩。每次从同一初态开始，使用相同的材料、驱动、制动、HHT 参数和 0.1 ms 主步长，目标为 5 s，单进程预算为 1,800 s。以下独立整机实验依次运行于 Ryzen 9 5900X。新版可以用两个 CPU 工作线程分别查询两个啮合级，按固定顺序组装总力和力矩。所用物体、候选面片和接触区域均由通用网格接口处理。

## 整机耗时

| CPU 方法 | 结束状态 | 仿真时间 | 进程时间 | 积分到 2.7246 s 的耗时 |
|---|---|---:|---:|---:|
| 原 OBB | 触及预算 | 2.7246 s | 1,800.09 s | 1,797.31 s |
| 复用候选与接触区拓扑的 OBB | 完成 | 5.0000 s | 1,349.14 s | 785.67 s |
| 复用候选与接触区拓扑的 AABB | 完成 | 5.0000 s | 1,317.40 s | 750.96 s |

在旧版完成的 27,247 个共同状态上，三根轴的角度、速度与两种新版逐项相同。OBB 在这段的积分提速为 2.29 倍。新 OBB 的 5 s 全程含 357,033 次力求值、4,013 次步长细分，1,338.75 s 积分时间中有 1,326.52 s 用于获取当前几何接触区。RecurDyn 记录的初始化与求解用时分别为 248.08 s、1,044.39 s，合计 1,292.47 s。两种软件采用的接触离散和线程数不同。[逐步耗时曲线](integration-time.png) · [时间、计数、摘要与误差](summary.json)

## 载荷与转速

新 AABB、OBB 从初态分别求解，50,001 行原始 CSV 的 SHA-256 相同。两级法向载荷的峰值分别为 6.555 kN、12.836 kN。按每级各区域法向力标量之和从零变为正值或反向计数，两个级次分别记录 245、129 次承载切换。不同齿对交接期间这项合力可以始终为正。[逐步响应数据](response.npz) · [两级法向载荷](normal-loads.png)

另外用旧版几何，在新版全程的原始轴角上重算法向载荷。采样包括每 10 ms、每级峰值、每次承载切换的前后两个主步。去掉重复时刻后，第一级共 886 个样本，第二级 744 个。所有样本的承载状态一致，两级载荷最大绝对差分别为 4.6×10⁻¹³ N 和 1.9×10⁻¹² N 以内。这组固定姿态核对覆盖完整 5 s，独立旧版动力学轨迹仅覆盖表内区间。[原始轴角载荷回放](native-load-check.json)

在 1.1432 s 的第一级，用原始轴角固定两刚体，优化前、后两个动力学程序都给出 1,817.706 N。将同一姿态转成四元数再读回，以全新查询对象求值，两种几何程序都得到 2,422.433 N。新版在两种表示下分别报告 3 个和 4 个接触区；两份返回值中各有两个区域给出逐项相同的点、法向和深度。每区独立施加一份刚度，使这个位姿的总力相差 604.727 N。候选面片与局部特征的去重能减少计算，各区域最终输出的物理接触仍可能重合。这是目前将该基线用于瞬时齿面载荷预测的具体限制。[两种输入及逐区结果](pose-sensitivity.json) · [全程四元数抽查](saved-load-check.json)

按相同时间戳和各轴转向核对原生 RecurDyn 的 50,001 个状态。输入、中间、输出轴的转速 RMS 偏差分别为 3.426、0.996、1.001 rad/s，最大绝对偏差分别为 19.162、6.321、7.375 rad/s。原生模型采用 20,636 个接触面片与四个 CPU 线程。在 102–114 ms 的制动段，对同一批 CPU 轴姿态上的接触力矢量求和，并计算其绕输出轴的力矩，得到与 RecurDyn 的输出接触力矩 RMS 差 107.22 N·m。法向载荷图报告各区域法向力的标量和，制动力矩图使用各区域的作用点与力方向。[转速对照](reducer-prefix.png) · [制动时的合力与力矩](../braking-wrenches.png) · [两级啮合动画](reducer-prefix.mp4)

## 与本地 NexDyn 实现的关系

NexDyn `aaac2df` 的 `TriMeshContactInteraction.cpp` 第 4274、4831、4930 行分别给出动态候选入口、历史面片集合并与邻接对的插入去重。此路径启用时，有效历史优先，必要时重新搜索。本实现复用三类数据。外层缓存两侧可能进入接触区的面片列表，内层缓存可能相交的面片对。两层分别用位移及转角对应的最大点运动量判断是否需要补搜。真实相交面片组成连通区域后，先按面片对编号去重，再展开唯一的顶点—面和边—边组合。区域的面片集合相同时，直接沿用这份拓扑。在每个新位姿上重新检查实际相交、接触深度和力。回退和跳转位姿沿用同一有效性检查。

## 定位重复工作

- 在相同的前 400 个主步上，旧 OBB 用时 28.65 s，新 OBB 单线程用时 13.85 s，两个工作线程用时 14.02 s。三份响应 CSV 的 SHA-256 相同，各有 3,250 次力求值与 47 次步长细分。仅串行几何修改已使这一段加速约 2.07 倍。[原始记录](400-steps.json)
- 在制动和连续啮合两个窗口各取 200 个姿态。每个窗口的第一级候选面片列表重建 100 次、复用 100 次，第二级重建 50 次、复用 150 次。另在每级每窗插入四个跨窗跳转、回退姿态。新 AABB、新 OBB 与旧 OBB 在这 816 次物体对查询中返回的接触点、法向、深度、排列逐一相同。[固定姿态核对](pose-replay.json)
- 同一 OBB 方法对自由双球做 5 s 仿真，进程用时 24.64 s。法向总冲量为 0.450067 N·s，光滑球解析值为 0.45 N·s。峰值为 22.45265 N，解析值为 22.5 N。[双球结果](head-on.json)

## 模型与复现

[模型包](../models.zip)包含相同的四份 Gmsh 网格和完整载荷表。打包时调整过场景内的网格相对路径。新几何实现位于提交 `f3f509d`，其父提交 `8ec5835` 保留优化前的代码。两版使用各自独立的 Release 可执行文件，摘要见运行记录。

```powershell
python -m zipfile -e results/aes/cpu-foundation/models.zip build/foundation-models
$env:OMP_NUM_THREADS = "4"
python benchmarks/aes/cpu_backend_run.py build/cpu-swept/Release/nc_contact.exe build/foundation-models/reducer/reducer.scene build/rerun-current-obb --method mesh-obb --scope finite
python benchmarks/aes/cpu_backend_run.py build/cpu-swept/Release/nc_contact.exe build/foundation-models/reducer/reducer.scene build/rerun-current-aabb --method mesh-aabb --scope finite
python benchmarks/aes/cpu_foundation_profile.py build/cpu-swept/Release/nc_mesh.exe build/foundation-models/reducer/reducer.scene build/rerun-current-obb/response.csv build/replay-native.json --replay-saved-loads --native-axes
```

抽查旧几何时，从 `8ec5835` 单独构建 `nc_mesh`，只将 `apps/mesh.cpp` 相对 `f3f509d` 的轴角输入选项补入命令行。运行的是已有的离散有限三角形接触路径，接触几何与材料代码均取原版本。命令末尾的 `--native-axes` 直接传入响应文件所存的刚体原点和转角，以及场景内的转轴。省略该选项可复现四元数回放。

# 两级减速器的 GPU 闭环仿真

同一套 65,732 片 Gmsh 网格、三轴惯量、共享中间轴、驱动和制动载荷。GPU 从初态独立推进 50,000 个 0.1 ms 主步，物理时间为 5 s。力模型采用共享线性罚刚度 1.6 × 10⁸ N/m，广义 α 积分器的谱半径取零。实测覆盖这台固定轴减速器。[CPU 模型与基准曲线](../cpu-foundation/reducer-physics/contact-refresh/README.md) · [GPU 三维动画](reducer.mp4)

新版逐步交互在共享 RTX 3090 上完成了五次 5 s 复测，中位在线耗时为 4.660 s。安装包、单步截止时间、与旧版的配对对照和 GPU 调度诊断见[服务器延迟复测](latency/README.md)。

## 运行时间（原版本）

| RTX 3070 运行方式 | 5 s 仿真在线耗时 | 次数 |
|---|---:|---:|
| 轨迹在 GPU 显存中连续推进 | 2.190 s | 3 次中位数 |
| 主机逐步提交载荷，GPU 逐步返回两级力矩，共享映射内存 | 2.664 s | 3 次中位数 |
| 主机逐步提交载荷，每步显式复制并同步 | 8.955 s | 1 次 |
| GPU 通过两侧材质点更新法向和力臂，显存内推进 | 4.574 s | 1 次，另一个近似模型 |

每次计时都包括初始化动力学状态、第一步至最后一步的接触检测和积分，以及取回完整响应。网格和几何表加载、分配、设备内核准备另计。逐步交互时，主机收到上一步的接触力和力矩后才提交下一步载荷。每个返回值均与显存中的对应步逐项核对。三次映射内存交互运行分别为 2.664、2.670、2.661 s。[独立计时记录](runs.json)

中位数运行的主机单步往返耗时 P50、P95、P99 分别为 50.2、63.4、98.7 µs。442 个主步超过 100 µs，最慢 1.662 ms。主机单步计时从写入该步命令开始，到读回力矩结束。批次边界调度开销计入整段墙钟时间。该结果满足整段快于实时，逐步硬截止时间尚未满足。[单步与设备统计](timing.json)

预处理包含 599.60 s 的 CPU 原网格查询和合计 12.29 s 的 GPU 候选窗口计算。窗口索引和三角片对约 245.9 MiB，压缩的载荷几何表约 1.55 MiB。已有缓存时，三次交互运行从加载到开始积分的准备时间分别为 4.25、1.29、1.36 s。可复用同一机构的几何缓存改变外载，只要运行姿态位于缓存覆盖范围内。[已生成的载荷几何表](geometry.npz)

## 在线计算的物理量

离线程序对两轴的独立角度组合运行 CPU 的完整 contour-ray 网格算法。每个采样位置保存单位罚刚度下的法向合力与两轴接触力矩。输入来自模型网格与姿态参数，不读取 CPU 的动力学轨迹。默认工作表覆盖 2,048 个相位和 33 个相对角偏移。共有 135,168 个双轴采样，11 个 CPU 几何查询未解，标记为无效。调用无效的受载单元或越出角度范围会中止并报告状态。

在线的 `mesh-wrench` 按当前轴角选择候选窗口。在每个隐式步的收敛姿态上，GPU 用有限三角片相交确认每个物体对的接触状态。优先尝试此前相交的面片及预处理的候选，一旦这些候选未命中便搜索整个保守窗口。若收敛时接触状态发生变化，重新解该步。受载时由几何表插值得到法向载荷与轴力矩，乘以模型罚刚度。两级力矩在同一次平衡方程中作用于中间轴。此过程近似了 CPU 每次重新划分接触区域并求穿深的操作。适用范围是固定平行转轴、具有公共转动周期的角度窗口和共享线性法向罚模型。

针对每级 231 个姿态，窗口内全部相交三角片对与同一浮点变换下完整双精度 BVH 的结果完全一致。一级累计 15,449 对，二级累计 52,453 对。样本包含启动、制动、均匀分布的全程姿态以及角度窗口边界。另在每级 204 个相同姿态上查询 CPU 原始接触区域，检查力的近似误差。[固定姿态核验](static-check.json)

## 与已接受的 CPU 结果对照

两条独立轨迹按全部 50,001 个原始时刻直接比较。表中峰值均取同一输出量在 0–5 s 的最大值。[数值明细与承载切换时刻](accuracy.json) · [GPU 原始轨迹和单步耗时](response.npz)

| 响应 | CPU | GPU |
|---|---:|---:|
| 输出轴平均接触力矩 | 137.8044 N·m | 137.8043 N·m |
| 输出轴最大接触力矩 | 432.71 N·m | 430.15 N·m |
| 峰值发生时刻 | 0.6 ms | 0.6 ms |
| 最后 1 s 最大接触力矩 | 172.49 N·m | 169.21 N·m |

输出轴接触力矩的逐点 RMS 差为 7.51 N·m。输入轴、中间轴、输出轴的转速 RMS 差为 0.194、0.063、0.077 rad/s。一级和二级传动误差 RMS 差为 0.0072、0.0140 mrad。两级法向载荷 RMS 差为 56.0、121.2 N。[整段输出力矩与启动放大](torque.png) · [三轴转速、两级传动误差和法向载荷](shafts-and-loads.png)

一级承载开关在全部共同输出时刻一致。二级在 2.1、2.2、2.3 ms 出现差异，CPU 短暂卸载，GPU 仍承载。到 2.4 ms，独立 CPU 与 GPU 轨迹的输出力矩分别为 61.61 和 183.85 N·m。把 CPU contour-ray 放到 GPU 当时的轴姿态上得到 185.89 N·m。该时刻的直接几何载荷近似只解释了约 2.04 N·m 的差异；其余差异与两条轨迹演化至不同姿态有关。对瞬时卸载和冲击波形的逐事件验证仍以独立 CPU 结果为依据。

## 复现

从仓库根目录运行。Windows 交互基准需要 C++20 编译器、支持映射主机内存的 CUDA GPU、CuPy 与 NumPy。模型导入使用 meshio。几何表可以直接从本目录读取。源模型与 Gmsh 网格已归档在同一个模型包中。从包内模型重新生成 GPU 输入和候选窗口后，全部 50,001 个状态的轴角、转速、两级法向力和广义接触力矩均逐项复现。

```powershell
python -m zipfile -e results/aes/cpu-foundation/reducer-physics/cpu-deck.zip build/gpu-model
python benchmarks/aes/gpu_hinge_mesh.py build/gpu-deck build/gpu-windows --scene build/gpu-model/reducer-physical/common-200.scene --method mesh-wrench --prepare
cmake -S . -B build/gpu-exchange -DNC_WITH_OCCT=OFF
cmake --build build/gpu-exchange --config Release --target nc_hinge_exchange
python benchmarks/aes/gpu_hinge_mesh.py build/gpu-deck build/gpu-windows --geometry results/aes/gpu-hinge-mesh/geometry.npz --method mesh-wrench --output build/gpu-run --exchange-lib build/gpu-exchange/Release/nc_hinge_exchange.dll
python benchmarks/aes/gpu_hinge_report.py results/aes/cpu-foundation/reducer-physics/contact-refresh/revised-response.npz build/gpu-run build/gpu-deck --animate
```

省略 `--exchange-lib` 可复测设备常驻方式。`--stream-io` 复测逐步显式复制。重新采样几何表时，使用启用原网格 contour-ray 查询并已配置依赖库的 nc_mesh 可执行文件。

```powershell
python benchmarks/aes/hinge_contact_map.py build/cpu-swept/Release/nc_mesh.exe build/gpu-model/reducer-physical/common-200.scene build/gpu-deck build/gpu-windows/geometry.npz --phases 2048 --layers 33
```

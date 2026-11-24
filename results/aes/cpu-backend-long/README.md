# 双级减速器在统一 CPU 接触后端上的长时程尝试

四个原始 STEP 齿轮、三根转轴、两级啮合，复用同一场景文件、初态及完整外载。一级直齿和二级斜齿共用中间轴。所有闭环试验从零时刻独立启动 C++ `nc_contact`，使用 `--solver penalty`、同一 HHT、相同的 0.1 ms 主步长、5 s 目标和 1,800 s 进程上限。四件 Gmsh 网格合计 65,732 个三角形。法向刚度为 1.6×10⁸ N/m，阻尼和摩擦为零，建模时设置的每处接触深度上限为 53 µm。此次采用离散步内试算和 HHT 默认细分，未开启连续碰撞事件定位。

## 独立的闭环运行

| CPU 接触路径 | 进程耗时 | 最后接受的物理时间 | 终止情况 |
|---|---:|---:|---|
| 网格 OBB + 局部重叠实体 | 192.76 s | 9.7 ms | HHT 细分耗尽 |
| 同一 OBB 重叠实体，增加失败原因记录 | 204.55 s | 9.7 ms | HHT 细分耗尽 |
| 网格 OBB + FCL 面片接触 | 3.05 s | 0.2 ms | HHT 细分耗尽，失败步有深度超过 53 µm 的试算 |
| 网格 AABB + FCL 面片接触 | 3.31 s | 0.2 ms | HHT 细分耗尽，失败步有深度超过 53 µm 的试算 |
| 网格 SAP + AABB 树 + FCL 面片接触 | 3.25 s | 0.2 ms | HHT 细分耗尽，失败步有深度超过 53 µm 的试算 |
| 原生 STEP + OCCT 实体求交 | 1800.06 s | 0.3 ms | 30 min 超时 |

表中的进程耗时包括初始化、最后未接受主步的试算及退出。每种配置单独启动，开发工作站上存在编译和诊断任务。工作站 CPU 为 AMD Ryzen 9 5900X。数据文件记录每个程序和场景的 SHA-256。[独立运行日志和计数](runs.json) · [模型文件哈希及三角形数](pose-replay.json)

原 OBB 网格实现累计接受了 97 个主步。至最后接受状态，积分用了 162.44 s，网格粗细检测用了 2.71 s，放置和准备局部几何用了 7.05 s，3,292 次局部实体布尔求交用了 150.09 s。重复运行得到完全相同的 98 组轴角和轴速，在同一主步耗尽 HHT 细分。重复运行该步没有记录到穿透深度超限。CAD 对照在最后接受状态已做 62 次 OCCT 布尔求交；超时主步内没有最终计数。[原网格逐步计数](local-progress.csv) · [原生 STEP 逐步计数](occt-progress.csv)

当前 CPU 中，Chen 2024 和 Chen 2026 的选项接在步内接触事件定位接口上，固定姿态受力几何共用 OCCT 重叠区域实现。本批关闭事件定位，表中只列实际运行的 CAD 受力路径。此次结果显示了这条布尔求交路径的成本，其他 CAD 接触几何方法需单独实测。

## 从三角片相交直接计算接触力

几何查询通过统一接口提供接触位置、法向和法向穿入量。原路径按实体重叠连通区域汇总，再用区域沿法向的宽度和线性罚力计算载荷。直接路径沿用同一个刚体对查询、罚力、轴力矩装配和 HHT，仅把接触几何替换为相交面片提供的结果。相连的交线合并为一组，取组内最大叶节点深度来生成一份接触，从而避免为同一交线上的每对面片重复加弹簧。这个路径使用 FCL 0.7.0 的三角片接触例程；交线分组和每组取最深点是本项目的基线设置。检验时只把这条路径用作传统面片接触对照。[FCL 的叶节点调用](https://github.com/flexible-collision-library/fcl/blob/0.7.0/include/fcl/narrowphase/detail/traversal/collision/mesh_collision_traversal_node-inl.h#L160-L179) · [深度与法向计算](https://github.com/flexible-collision-library/fcl/blob/0.7.0/include/fcl/narrowphase/detail/traversal/collision/intersect-inl.h#L804-L841)

两级各回放了原网格闭环从 0 至 9.7 ms 的全部 98 个状态。两个方法逐帧给出相同的接触区域数量，直接路径在一级有 11 帧、二级有 17 帧超出 53 µm。一级在 1.2 ms 有两处接触，重叠宽度法最大 5.57 µm，FCL 叶节点法最大 275.77 µm。后者对应面片 72 和 4426，顶点向对侧面片平面正交投影后，投影点距有限三角形边界 170.17 µm，位于面外。二级在 0.4 ms 首次超限，最大 112.14 µm，投影点落在面外 5.77 mm。两种深度定义产生的差异在相同位姿下即可复现。参考重叠宽度本身仍需独立验证。

独立积分时，三种粗检测均在最后接受 0.2 ms 后的下一步触及深度限制，每种方法报告 12 次试算超限，最后一次为一级啮合的 312.591 µm。闭环在自身试算姿态上评估接触，姿态和时间均可与上述固定姿态回放不同。当前直接路径已验证纯共面贴合箱体的零载荷、顶点接触和物体顺序互换。减速器上的结果说明，面片接触点还需核验其在有限面/边上的位置和法向压入量。

下图上排比较原 OBB 闭环已接受的轴姿态上，两种几何算法分别给出的最大深度。下排将原 CPU 网格的输出轴速、输出接触力矩与既有 RecurDyn 独立仿真对齐，仅画到 CPU 最后接受的状态。力矩按各处法向力对输出轴的力臂求和。在这 98 个时间戳上，CPU 和 RecurDyn 的输出接触力矩峰值分别为 889.12 与 495.76 N·m，输出轴速 RMS 差为 2.82 rad/s。图中缺少 9.7 ms 之后的 CPU 动力学轨迹。它可以诊断接触进入与载荷变化，无法验证制动加载、稳态响应或 5 s 性能。[启动诊断图](cpu-startup.png) · [逐姿态载荷、力矩与深度](pose-replay.csv) · [RecurDyn 原始时序](../long-mesh/recurdyn-5s.npz)

## 复现

在仓库根目录构建启用 CPU 网格、CGAL、FCL 与 OCCT 的 Release 版本，将运行库加入 PATH。使用 `benchmarks/aes/long_mesh_deck.py` 和原减速器 Gmsh 模型准备同一 `reducer.scene`。比较程序会拒绝覆盖已有运行目录，每个命令独立推进 50,000 个主步，记录中止或超时前最后已接受状态。将下面的 OBB 替换为 `mesh-aabb` 或 `mesh-sap` 即可切换粗检测。SAP 在这里指物体级扫描排序，进入网格后使用 AABB 层次树。

```powershell
$exe = "build/cpu-swept/Release/nc_contact.exe"
$scene = "build/long-mesh/deck/reducer.scene"
python benchmarks/aes/cpu_backend_run.py $exe $scene build/repeat/local --method mesh-obb --scope local
python benchmarks/aes/cpu_backend_run.py $exe $scene build/repeat/fcl --method mesh-obb --scope fcl
python benchmarks/aes/cpu_backend_run.py $exe $scene build/repeat/occt --method cad-occt
python benchmarks/aes/cpu_backend_pose_replay.py build/cpu-swept/Release/nc_mesh.exe $scene build/repeat/local/progress.csv build/repeat/replay
```

`nc_mesh` 在同姿态诊断中调用与 `nc_contact` 相同的 `mesh_region_query`。各方法的闭环运行全部由 `nc_contact` 推进。[闭环运行脚本](../../../benchmarks/aes/cpu_backend_run.py) · [同姿态核验脚本](../../../benchmarks/aes/cpu_backend_pose_replay.py)

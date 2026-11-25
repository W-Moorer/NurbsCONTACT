# 两级减速器的 CPU 局部 CAD 接触回放

输入为既有 5 s 网格闭环从初态至终态的 50,001 个已接受轴姿态，输出间隔为 0.1 ms。输入归档保留四个实体的平移、转角与原始 CSV 的 SHA-256。STEP、Gmsh 网格、共享法向刚度和作用轴方向取自同一场景。两级分别由 CPU 进程查询。[输入姿态](trajectory.npz) · [原始运行](../cpu-foundation/reducer-physics/contact-refresh/cpu-100us.json) · [模型归档](../cpu-foundation/reducer-physics/cpu-deck.zip)

## 回放结果

两级独立 CAD 搜索与分组的并行进程共同用时 **1,179.33 s**。每级均读完全部 50,001 个姿态。两级纯网格进程分别用时 48.96 s 和 163.80 s。所有进程运行在同一台 Ryzen 9 5900X 上，分批计时，部分诊断作业重叠运行。进程用时包含导入、预处理、查询和输出。原网格闭环进行过 151,547 次整机力评估，因此本表的单次姿态重查耗时不能换算成已经达到的闭环成绩。[完整计时与核验结果](summary.json)

| 阶段或对照 | 一级直齿轮 | 二级斜齿轮 |
|---|---:|---:|
| 纯网格全程回放进程 | 48.96 s | 163.80 s |
| 已给网格种子，单独进行局部 CAD 查询的全程进程 | 89.03 s | 213.24 s |
| 独立 CAD 搜索、旧点复查与连通分组全程进程 | 575.77 s | 1,179.33 s |
| 两套种子上的固定点双出口核验 | 24 / 24 | 42 / 42 |
| 全模型 STEP 区域数、归属及分组均吻合的核验姿态 | 24 / 24 | 32 / 32 |
| 已给网格种子，局部深度无法确认的点 | 6,362 / 90,922 | 28 / 148,369 |

66 个固定点的两侧出口误差均低于 1 µm。基于网格种子的全程查询另有 6,390 个局部深度无法确认。对这些点逐个查完整 STEP，全部在边界命中或切触检查处返回歧义。提高到整实体查询也未恢复有效深度。二级网格检测另有一个姿态无法给出区域。交换查询双方后，56 个审计姿态的快速分组数一致。将载荷换回同一物体和轴，最大扭矩差小于 10⁻¹² N·m。[一级固定点](stage-1-fixed.json) · [二级固定点](stage-2-fixed.json)

在二级的 0.3 ms，5 个网格点只覆盖 3 个实际重叠区中的 2 个。独立 CAD 搜索覆盖了 3 个。在 2.0 ms，网格的 12 处接触对应 3 个原生区域，通过实体内部路径可以将网格种子合并成 3 组。在 29.7 ms，网格返回无法判定，CAD 搜索的 6 个面见证合为 3 组，与独立 STEP 分类一致。[接触切换图](stage-two-switches.png) · [二级逐点归属](stage-2-audit.json)

在 0.5 ms，按每组最深点计算的二级载荷为 8.518 kN，原生区域参考为 8.614 kN。对应扭矩为 377.43 和 533.26 N·m。一处最深点的法向约为 (0.984, −0.176, 0)，相应原生区域参考法向为 (0.357, −0.867, −0.349)。以两侧法向最为相对的见证为作用位置，沿两侧外法向之差的单位方向重测深度，扭矩改为 464.21 N·m。在二级的 23 个有载审计姿态上，这项修改使扭矩均方根差从 128.99 降至 100.64 N·m，法向载荷均方根差从 1.325 增至 1.645 kN。作用点和作用方向仍需按整个承载区确定。[启动载荷与力矩](stage-two-loads-detail.png) · [全程载荷与力矩](fixed-pose-loads.png) · [逐姿态数据](replay.npz)

在三种策略均有输出的有载姿态上，纯网格、同网格种子加局部 STEP 修正、独立 CAD 搜索加最深见证三种方法相对原生区域参考的扭矩均方根差，一级分别为 6.65、1.86、48.64 N·m（18 个姿态），二级分别为 73.77、12.06、128.22 N·m（22 个姿态）。这些数据支持继续使用局部 CAD 几何修正，同时表明独立搜索所找到的点需要进一步处理才能给动力学使用。[一级审计](stage-1-audit.json) · [二级审计](stage-2-audit.json)

## 测量口径

- **固定点精度**。各取 CAD 和网格给出的作用点与法向。在各刚体坐标系中，预建的 CAD 面包围盒树缩小查询范围。两侧最近的、有向的、位于裁剪范围内的出口给出穿入量与作用中点。对照在整个 STEP 上执行实体内外分类和射线求交。先前相同的姿态只计一次。预设的数值对照门槛为两端各 1 µm。
- **网格与同点 CAD**。每一帧重新以 `mesh-obb` 和 `contour-ray` 查询全模型。网格的每个作用点与方向再交给局部 CAD 查询，单独改变深度和力臂。网格查询失败记为缺测。
- **独立 CAD 搜索**。用每张裁剪面两个参数方向各 6 格产生初始候选，以局部曲面迭代寻找相对面的交点。每个姿态同时重查上次的见证并双向搜索新面。只有在原始两实体内验证了局部穿入才保留见证。同一片重叠材料可以横跨多对 CAD 面。两见证之间能找到在两实体内连续通过的折线时，将它们归为一组。有限的起点和路径搜索仍可能漏组或拆组。为量出作用点选择的影响，每组另取穿入最大的已验证见证计算诊断载荷。
- **独立区域核验**。在启动、载荷峰值、全程固定位置、网格区域切换及失败帧，用完整原始 STEP 实体求交，将每个见证分别分类到实际重叠实体。这样能识别总数相同但位置错误、同一区多次计数，以及仍在承载时另一接触区漏检。审计帧是有目的的抽查。各区域的参考作用点和法向由独立的连通体体积质心与来源面有向面积确定，再按局部 STEP 出口计算深度与作用中点。
- **求解器入口**。`nc_contact --method cad-local` 使用上述原生区域发现和局部深度。快速的参数域搜索与分组是回放实验入口 `nc_cad_probes --groups`。动力学载荷诊断采用相同的共享罚刚度。回放中的位姿来自既有轨迹，缺少独立动力学反馈。

## 复现

在仓库根目录解包模型并构建带 OCCT 与 CPU 网格支持的 Release 可执行文件。将所用 OCCT、网格运行库加入 PATH。PowerShell 示例中的 `$bin` 指向实际输出目录。每次原生程序运行上限为 1,800 s。

```powershell
cmake --build build/cpu-swept --config Release --target nc_contact nc_mesh nc_cad_probes nc_cad_distance_benchmark nc_cad_solid_tests
python -m zipfile -e results/aes/cpu-foundation/reducer-physics/cpu-deck.zip build/replay-model
$scene = 'build/replay-model/reducer-physical/common-200.scene'
$data = 'results/aes/cpu-local-cad'
$work = 'build/local-replay'
$files = @($scene, "$data/trajectory.npz", $work, "$bin/nc_mesh.exe", "$bin/nc_cad_distance_benchmark.exe", "$bin/nc_cad_probes.exe")
foreach ($action in @('geometry','fixed','audit','swap','normals','failures')) {
    python benchmarks/aes/cpu_local_cad_replay.py $action @files
    if ($LASTEXITCODE -ne 0) { throw "Replay failed in $action" }
}
python benchmarks/aes/cpu_local_cad_report.py $scene $work $data
```

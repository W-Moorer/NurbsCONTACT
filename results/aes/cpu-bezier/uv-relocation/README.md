# CPU 减速器中 Bézier 射线的局部重定位

在同一场景选择 `--method cad-bezier-relocate` 即可启用。旧路径为 `cad-bezier-bounds`，网格对照为 `mesh-obb`。三者共用模型、接触力和积分设置。

## 改了哪条路径

旧方法用曲面块中心的一阶近似估计交点。估计的参数超出块边界时，即使真实交点在块内，也可能触发 OCCT 整面求交。新方法继续使用已经预处理的 Bézier 控制点、权重、导数界和空间索引。在中心估计越界处，先用控制点在射线横向的投影判空，再收紧交点可能落入的参数范围。随后将同一工作线程在该面最近一次确认的参数位置放进这个范围作迭代初值。已有面对跟踪负责跨姿态提供候选，其他可能更早相交的面和相邻块继续参加检验。非矩形裁剪上的候选命中、真实边缘和未能隔离的根交回 OCCT。缓存可能来自上一姿态，也可能来自本姿态另一条射线。本次比较衡量整条局部路径的收益。

## 原来回退的射线逐条与 OCCT 对照

回放原参考轨迹，两级各 800 个姿态。在旧算法真实调用 OCCT 后，对同一面、同一起点、方向及有限长度另跑新算法；对于局部已判清的射线，检查交点个数及深度是否在面的 CAD 公差内。另用正式可执行文件比较完整查询的分组、积分深度和接触合力矩。

| 原因与结果 | 一级 | 二级 |
| --- | ---: | ---: |
| 原中心初值越界而回退 | 3,800 | 15,017 |
| 新路径可以直接判清 | 704 | 14,915 |
| 可直接判清的射线中，与 OCCT 核对超差 | 0 | 0 |
| 这些已判清射线原来花在 OCCT 的累计时间 | 0.287 s | 5.906 s |
| 全部越界射线的局部尝试累计时间，包含未判清的尝试 | 0.027 s | 0.110 s |
| 全部原因的 OCCT 调用，优化前 → 优化后 | 4,119 → 3,415 | 18,986 → 3,871 |

时间为逐调用累加。二级额外减少的 200 次来自原越界集合以外的路径。两级 1,600 个姿态的分组和承载面对全部一致，最大深度差为 1.51e−14 m，最大合力矩差为 5.24e−8 N·m，最大输出轴力矩差为 4.90e−8 N·m。二级剩余调用中，有 3,207 次是在检查两处承载点之间的材料是否连通时，射线收缩界未达标；原 CAD 在这些短线段上均未发现交点。[同射线核对](cohort.json) · [优化后按原因计数](residual.json) · [生产版本的同姿态核对](fixed.json)

## 五秒闭环

同一可执行文件、减速器、0.1 ms 主步和 50,000 个主步，每种方法在独立进程中依次运行。各次进程限时 1,800 s，表中墙钟包含模型初始化。离线建表另计。网格复测的响应文件与[上轮记录](../depth-bounds/comparison.json)逐字节一致。本批旧 CAD 与上轮旧 CAD 有少量动力学步选择了不同的细分，下表及主图使用本批独立对照。

| CPU 方法 | 整段墙钟 | 其中初始化 | 普通步均值 | 普通步 P95 |
| --- | ---: | ---: | ---: | ---: |
| Bézier 局部重定位（ours） | 644.07 s | 49.70 s | 9.68 ms | 16.63 ms |
| 优化前 CAD | 911.25 s | 45.85 s | 15.22 ms | 21.93 ms |
| 纯网格 OBB | 326.57 s | 3.92 s | 6.31 ms | 20.58 ms |

新 CAD 节省 267.18 s（29.3%）。两种 CAD 均有 17 步触发局部布尔求交。其余 49,983 步中，按两个接触对累加的积分时间从 443.42 s 降为 160.58 s，分组从 205.56 s 变为 218.32 s，搜索从 181.56 s 降为 157.91 s。新的分组占这四项普通步累计时间的 35.3%。两级并行，这些分项不能直接相加当作整机墙钟。[五秒原始计数与比较](comparison.json) · [分阶段计时](phases.json)

两套 CAD 的承载与脱开状态在 50,001 帧中逐帧相同。0.5 ms 的输出轴力矩两者均为 432.631 N·m，网格为 395.368 N·m。在以前出现尖峰的 2.5377 s，优化版比原生 CAD 回放低 0.001485 N·m，前后共 41 帧的最大绝对差为 0.070 N·m。输出力矩相对 RecurDyn 的全程 RMS 差分别为新 CAD 42.095 N·m、旧 CAD 42.095 N·m、网格 29.142 N·m。[完整响应曲线](reducer.png) · [启动与历史尖峰窗口](switches.png) · [原始响应序列](responses.npz)

新旧 CAD 的输出轴力矩之差在全程的 RMS 为 0.0315 N·m，最大差为 2.579 N·m，位于 3.6021 s。从 3.6009 s 各自已接受的状态出发，用同一新版求解器重算随后 12 步，旧轨迹在第一个主步多作一次对半细分，两条轨迹分别复现 172.4253、169.8467 N·m。将两边的子步统一为 0.05 ms，末端力矩均为 184.4421 N·m，彼此差 4.7e−8 N·m；统一为 0.00625 ms 时均为 212.2597 N·m。不同时间步得到的绝对力矩有显著差别，0.1 ms 的这个瞬时值仍对时间离散敏感。[相同子步逐次重算](paired-peak.json)

本批新旧 CAD 在 3.6010、4.4674、4.9646 s 三步作出了不同的细分决定；从这三步前各自的状态重算每处随后一至四帧，统一 0.05 ms 子步后两条轨迹末端力矩的最大差为 9.90e−8 N·m。另与历史旧 CAD 归档比较，最大差为 1.736 N·m，在 4.5043 s；从 4.5039 s 统一以 0.05 ms 子步重算至该时刻，双方力矩均为 144.1369 N·m。正式耗时和主图均取同批复测。[三个细分决策回放](step-refinement.json) · [历史对照的峰值回放](archived-peak.json)

## 复现

按[减速器运行说明](../README.md)准备场景和 OCCT 运行库，令 `$env:OMP_NUM_THREADS="2"`，并将 `$env:NC_CAD_CACHE_DIR` 指向已准备的零件缓存。令 `$archive="results/aes/cpu-bezier"`，`$out="$archive/uv-relocation"`。逐射线对照使用在单独源码副本应用[测量补丁](../../../../benchmarks/aes/cad_fallback_profile.patch)后构建的探测程序。完整动力学使用正式 Release 二进制文件。

```powershell
python benchmarks/aes/cpu_pressure_columns.py paired $scene $probe --probes $probe --archive $archive --out $out --relocate
python benchmarks/aes/cpu_cad_fallback_counts.py $scene $archive $countingProbe "$out/cohort.json"
python benchmarks/aes/cpu_cad_fallback_counts.py $scene $archive $countingProbe "$out/residual.json" --relocate
foreach ($method in @("cad-bezier-relocate","mesh-obb","cad-bezier-bounds")) {
    python benchmarks/aes/cpu_process_timing.py $method $scene "$runs/$method" $exe
}
python benchmarks/aes/cpu_depth_results.py $scene $runs $archive $out --relocate
python benchmarks/aes/cpu_pressure_step_replay.py $scene "$runs/cad-bezier-bounds" "$runs/cad-bezier-relocate" $exe "$out/paired-peak.json" --previous-method cad-bezier-bounds --current-method cad-bezier-relocate --steps 36010 --window-steps 12 --final-width-only
```

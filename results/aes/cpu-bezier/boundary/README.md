# CPU 接触后端的冻结与验收

固定网格对照采用 `mesh-obb --mesh-region-scope contour-ray`，原 CAD 对照采用 `cad-compiled`。新增 `cad-boundary` 从同一零件文件加载曲面、裁剪边和拓扑，在两侧材料边界附近补充候选起点。选中面后在实际曲面上求接触弦。运行程序 `nc_contact_runtime` 可在关闭 OCCT 的条件下独立构建。所有测量均使用 CPU。

## 同姿态的几何与载荷

从已有五秒减速器轨迹提取二级 13 个姿态，再取两级各 6 个此前未查的姿态。每种方法接受同一对刚体的完整位姿。离线参考取两个原 STEP 实体的相交连通区、压力方向和力臂，按与仿真相同的共享弹性刚度求合力及绕从动轴的力矩。13 个原始姿态的 STEP 参考积分容差从 `1e-8` 收紧至 `1e-10` 时，合力最大变化 0.000007 N，轴力矩最大变化 0.00000041 N·m。这里检验的是所选接触模型的几何与数值一致性。[固定姿态](poses.json) · [STEP 参考](step-reference.json) · [参考加密前](step-nominal.json)

| 方法 | 区域数与 STEP 一致 | 最大合力矢量差，N | 最大轴力矩绝对差，N·m |
| --- | ---: | ---: | ---: |
| 三角网格 OBB | 21/25 | 1363.06 | 82.00 |
| 冻结 CAD | 19/25 | 1069.43 | 68.05 |
| 边界补搜 CAD（ours），普通积分 | 25/25 | 495.80 | 28.33 |
| 边界补搜 CAD（ours），自适应局部积分 | 25/25 | 84.44 | 4.37 |
| 边界补搜 CAD（ours），固定 96 格积分 | 25/25 | 62.36 | 3.62 |

274.1 ms 时，冻结 CAD 找到两个承载区，输出 194.48 N·m。边界补搜找到三个承载区，普通积分得到 139.13 N·m，STEP 对照为 138.65 N·m。上述区域计数只覆盖实际回放的姿态。多起点搜索和每个面板对的见证点仍有有限分辨率。[逐姿态载荷和误差](pose-loads.csv) · [程序和零件哈希](probe-builds.json)

自适应模式在同一已识别区域上逐级比较合力及关于两个刚体原点的力矩。10 N 和 0.5 N·m 是本轮选定的内部停止阈值。正序回放时 25 个姿态均达到停止条件，外部误差如表。反序回放保留 25 个姿态的区域数，但二级 1.6 ms 的轴力矩改变了 3.65 N·m。已缓存见证点会改变局部积分窗口。普通积分正反序回放的三分量合力和轴力矩在此数据集上可复现。自适应模式保留为实验选项。

## 凸轮运动学与动力学

偏心圆柱凸轮和背衬平底从动件共两个自由度。用同一驱动、质量、弹簧和阻尼跑 3.2 秒，HHT 主步为 0.1 ms，共 32,001 帧。独立的 DOP853 解使用圆柱对平面的解析压入量求力。对照时将位移插值到 HHT 输出时刻，并按进入、脱开方向依次配对事件。[模型及编译零件](cam-inputs.zip) · [逐帧数据](cam.npz) · [切换时刻与误差](cam.json) · [响应图](cam.png)

| 方法 | 位移 RMS 误差，µm | 位移最大绝对误差，µm | 进入和脱开次数，参考为 71 |
| --- | ---: | ---: | ---: |
| 三角网格 OBB，aligned-96 | 277.28 | 802.08 | 71 |
| 冻结 CAD | 1641.14 | 6076.77 | 61 |
| 边界补搜 CAD，普通积分 | 1070.57 | 3243.98 | 59 |
| 边界补搜 CAD，固定 96 格积分 | 13.45 | 40.21 | 71 |

在 1.1836 s 的实际 CPU 姿态，解析法向载荷和 STEP 都是 4.47879 N，普通积分为 0.00648 N。固定 96 格积分为 4.47919 N。普通模式在低分辨率整面上找到有效弦后即停止，窄区内偏离承载中心的弦会低估压入量。固定积分覆盖更多节点，若整面节点仍然错过已发现的区域，就在局部窗口沿材料边界重新积分。只用全局 96 格时，该凸轮在 0.5830 s 遇到未解区域。启用局部补算后完成全程，最大切换时差 0.356 ms。自适应模式使用 0.01 N 和 0.0005 N·m，曾在 0.0858 s 达到求解器细分上限。[同一姿态的解析、STEP 及编译 CAD 力](cam-poses.json)

## 减速器五秒闭环

同一两级减速器、载荷、材料和输入文件，两个 OpenMP 线程，主步 0.1 ms，预测姿态更新，HHT 谱半径 0，共享罚刚度。每种完成的配置交付 50,001 帧。在线时间读取求解器的 `statistics.wall_s`，预处理、加载和轨迹导出单列。墙钟预算每次 30 min。[当前整段指标](reducer.json) · [五秒响应及启动峰值](reducer.png) · [新 CAD 逐帧响应](reducer.npz) · [网格、冻结 CAD 和 RecurDyn 全程响应](../standalone/responses.npz)

普通边界补搜将输出力矩相对 RecurDyn 的全时程 RMS 差从冻结 CAD 的 42.09 N·m 降至 35.43 N·m，网格为 29.14 N·m。输出力矩最大绝对值分别为 432.63、432.63 和 432.71 N·m，RecurDyn 为 509.32 N·m。两种 CAD 的最大相邻主步力矩跳变量均为 259.99 N·m。该比较按级记录总法向载荷的通断；级内同时承载的齿对数量另见同姿态记录。自适应试运行在 30 min 截止时接受了 25,070 个主步，即 2.5070 秒。[截止记录及分阶段累计耗时](adaptive-cycle.json)

## 复现

从仓库根目录构建。命令中的 `runtime` 表示生成的 `nc_contact_runtime`，`probe` 表示 `nc_compiled_poses`。Windows 多配置生成器使用 `--config Release`。完整的网格及 STEP 工具另按项目 CMake 选项开启对应依赖。[减速器 STEP 和网格](../../cpu-foundation/reducer-physics/cpu-deck.zip) · [减速器编译零件](../standalone/reducer-inputs.zip)

```text
cmake -S . -B build/cpu-release -DNC_WITH_OCCT=OFF -DNC_WITH_CPU_MESH=OFF
cmake --build build/cpu-release --config Release
ctest --test-dir build/cpu-release -C Release --output-on-failure
python benchmarks/aes/cpu_boundary_replay.py poses.json probe parts cad-boundary replay.json
python benchmarks/aes/cpu_compiled_reducer.py run common-200.scene runtime outputs --method cad-boundary --parts parts
runtime cam.scene 0 0.0001 3.2 1 cam-result --method cad-boundary --cubature fixed96 --solver penalty --penalty-stiffness shared --geometry-update predictor --spectral-radius 0
```

同姿态脚本接受 `mesh-obb`、`cad-compiled`、`cad-boundary`、`fixed96`、`adaptive` 和两种 STEP 参考；分别传入相应查询程序与输入目录。`--reverse` 用于倒序回放。每行合力按三个笛卡尔分量给出，单位 N，力矩单位 N·m。完整的五秒运行器检查所有输出时刻和载荷的有限性，保存程序、场景和响应的 SHA-256。独立 CPU 测试包含裁剪孔、近相切曲面、人工分面、交换物体以及重复查询。[独立构建测试](standalone-ctest.txt)

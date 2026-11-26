# 承载区的块级深度判定

`nc_contact --method cad-bezier-bounds` 在通用 CPU 接触后端使用带深度界的 Bézier 面片。`cad-bezier` 保留原来的逐柱求交路径。两个入口共享场景、积分点、接触区分组、接触力定律和 HHT。完整场景用同一减速器、0.1 ms 步长、5 s 仿真和每种方法 1,800 s 进程时限。

## 判定与回退

对一个多项式双三次 Bézier 跨度，在本体坐标中选取张成局部平面的两个方向。将原控制网减去该平面的仿射参数化，再用剩余控制点及其一阶差分包住三个方向的偏差和导数。表按原始 STEP 形状、OCCT 版本和表版本写盘。刚体位姿、齿数、输入轨迹和载荷均不参与建表。

一个积分块最多包含八个原有材料点。候选面片先经过父子控制网包围盒筛选，再用这些点的位置和法向范围收紧有限射线的深度区间。区间空时，一次排除整个块对。若所有射线在给定跨度的内部横穿曲面，且投影后的曲面方程在整个参数域上的收缩界小于 0.8，则内部交点唯一。承载边界穿过块时，在原积分点上分别判定。

对已隔离的内部交点，程序在原多项式上细化深度，并核对真实裁剪范围及其他可能遮挡的面。深度残差的验收使用毫米坐标中的数值误差估计。承载块可以共用曲面定位，块中各点允许有不同的深度。非矩形裁剪的命中、重根、接近切线的射线、多个候选和求解未收敛的点均沿用原 CAD 查询。块级界针对现有离散积分规则；对力和力矩的核对另使用相同姿态和闭环轨迹。

## 预处理省下了哪些计算

四份 STEP 新增的深度界共 15.26 MB。每级各回放 800 个姿态，一级和二级分别观察到 163 和 142 次面对分组切换。两个 CAD 方法在全部 1,600 个姿态上给出相同分组和判定状态。二级输出轴力矩的同姿态最大绝对差为 6.47×10⁻¹⁰ N·m。[预处理文件清单](preprocessing.json) · [固定姿态的接触区与合力、力矩核对](poses-selected.json)

| 二级啮合的在线工作量，800 个姿态 | 原 CAD | 深度界 CAD（ours） |
| --- | ---: | ---: |
| 原求交器访问曲面块，含搜索与积分 | 373,073 | 268,819 |
| 其中承载区积分访问曲面块 | 234,985 | 130,731 |
| OCCT 求交，含搜索与积分 | 21,418 | 18,986 |
| 其中承载区积分调用 OCCT | 15,572 | 13,140 |

二级回放中，新路径另做了 10,314,456 次点与包围区间的简单检查、161,217 次深度区间判定，在原曲面上细化了 21,509 个交点。可整体排除的候选块有 43,607 个；能整体确认全部积分点命中的块只有 27 个。齿面在本体坐标中的形状固定，每个新姿态下的承载边界、首个相交面以及每个命中点的深度仍随运动改变。二级在这里主要省去了空区域的精确求交。

两种方法按整段运行时，二级积分的快慢随先后顺序改变。第三次复测让两个进程保有各自的接触历史，每次只唤醒一个进程，交替交换每个姿态上先运行的方法。每级有 640 个计时姿态；一级八个窗口中有六个变快，二级有七个。下表为原 CAD → 深度界 CAD 的原生积分均值，进程通信不计入。[第一轮](poses-selected.json) · [交换整段顺序](poses-reverse.json) · [逐姿态交替及分窗口计时](poses-interleaved.json)

| 固定姿态的计时顺序 | 一级承载区积分 | 二级承载区积分 |
| --- | ---: | ---: |
| 各方法单独跑完一段 | 2.19 → 1.75 ms | 9.08 → 8.07 ms |
| 交换两种方法的整段顺序 | 1.92 → 1.74 ms | 9.31 → 10.40 ms |
| 每个姿态交替先后 | 2.79 → 2.23 ms | 10.81 → 9.89 ms |

## 五秒仿真的计算预算

同一台 Ryzen 9 5900X 按深度界 CAD、网格、原 CAD 的顺序各单独运行一次，共用同一 Release 程序、两线程上限和同一初始状态。墙钟含读模、积分和响应输出。网格的本次输出与历史网格逐字节一致。原 CAD 本次的响应文件与历史 CAD 哈希不同，以下所有新旧差值均使用本批两次独立积分的输出。[三种方法的运行与误差摘要](comparison.json)

| 方法 | 整段墙钟 | 普通主步平均 | 输出轴力矩与 RecurDyn 的 RMS 差 | 中间轴转速与 RecurDyn 的 RMS 差 |
| --- | ---: | ---: | ---: | ---: |
| Gmsh 三角网格 + OBB | 326.70 s | 6.31 ms | 29.14 N·m | 0.2040 rad/s |
| 原 Bézier CAD | 915.98 s | 15.39 ms | 42.09 N·m | 0.3008 rad/s |
| 预计算深度界的 Bézier CAD（ours） | 915.47 s | 15.28 ms | 42.09 N·m | 0.3008 rad/s |

新旧 CAD 的进程墙钟差为 0.50 s（0.055%）。按面对累积的承载区积分由 462.90 降至 446.43 s；两级并行执行接触查询的实际墙钟由 802.57 降至 799.64 s。模型初始化由 44.79 增至 46.49 s。在本次测试中，积分工作量的局部下降对整机计时影响很小。两种 CAD 相对 RecurDyn 的整程力矩与中间轴转速差也十分接近，网格在这两项上更接近 RecurDyn。

深度界 CAD 的 49,983 个普通主步合计 763.64 s，平均 15.28 ms。触发局部布尔求交的 17 个主步合计 91.75 s，包含这些步的其他计算。若启动、输出和这 17 步全部维持本次用时，给普通步的预算只剩每步 2.96 ms 才能在 300 s 内跑完。按面对累积的粗筛、搜索、分组、承载积分时间分别为 77.39、186.90、206.86、446.43 s。两级并行计时有重叠，四项之和不能当作整机墙钟。[五秒进程与求解器计数](cad-bezier-bounds-run.json)

## 接触切换与力矩

两种 CAD 的 50,001 帧中，两级承载状态逐帧相同。一级在 0.3 ms 开始承载。二级 CAD 在 0.3、0.9、1.2、2.0 和 2.1 ms 切换，网格在 0.3、1.0、1.3、2.1 和 2.4 ms 切换。0.5 ms 的闭环输出轴力矩分别为新旧 CAD 432.631 N·m 和网格 395.368 N·m；这是从各自初始条件自由积分的结果。2.5377 s 的历史尖刺处，新 CAD 与原生 CAD 参考相差 −0.001485 N·m，附近 41 帧最大差 0.06965 N·m。图中的数据未平滑。[全程力矩、转速和两级载荷](reducer.png) · [二级启动与旧尖刺放大](switches.png) · [逐帧数据](responses.npz)

本批新旧 CAD 的输出轴力矩 RMS 差为 0.0220 N·m，最差时刻为 4.5043 s，相差 1.7357 N·m。一级与二级法向载荷的 RMS 差分别为 1.03 和 0.36 N，最大差分别为 143.29 N（4.5040 s）和 28.93 N（4.5043 s）。这些单帧差值在长时程曲线上不易看清，另检查了统一子步的局部结果。

两条轨迹的 HHT 接触查询批次在 3.1343、3.7281、4.5040、4.7281 s 四个主步选择了不同的细分。取各自在决策前的已接受状态，用同一深度界 CAD 后端重算。在 4.5040 s，原 CAD 轨迹的初态按一个主步通过，新 CAD 轨迹的初态触发一次对半细分；这复现了三步以后力矩为 137.9359 和 139.6715 N·m 的差别。将两边统一为 0.05 ms 子步，4.5043 s 的两次力矩都是 144.1369 N·m；统一为 0.00625 ms 子步，得到 147.5896 N·m。四个决策时刻及其后三帧共 16 个位置，在统一 0.05 ms 子步下两条轨迹的最大差为 8.19×10⁻⁸ N·m。这同时表明两条轨迹之间的差别可以由细分选择解释，该处力矩的绝对值仍对时间步长敏感。[逐次子步回放](step-refinement.json)

## 复算

在仓库根目录准备[减速器场景、零件与运行库](../README.md)。`$scene` 指向 `common-200.scene`，`$bin` 指向启用 OCCT 和网格后端的 Release 可执行文件。先运行固定姿态对照，两种模式使用同一探测程序，两级各取八个 100 帧窗口。每个窗口的前 20 帧参与几何核对，后 80 帧计时。第二轮交换两种模式的运行顺序。

```powershell
$env:OMP_NUM_THREADS = "2"
$env:NC_CAD_CACHE_DIR = "$PWD/build/reducer-geometry-cache"
$archive = "results/aes/cpu-bezier"
$out = "$archive/depth-bounds"
python benchmarks/aes/cpu_pressure_columns.py paired $scene "$bin/nc_cad_probes.exe" --probes "$bin/nc_cad_probes.exe" --depth-bounds --archive $archive --out build/depth-first
python benchmarks/aes/cpu_pressure_columns.py paired $scene "$bin/nc_cad_probes.exe" --probes "$bin/nc_cad_probes.exe" --depth-bounds --reverse-order --archive $archive --out build/depth-reverse
python benchmarks/aes/cpu_depth_pairs.py $scene "$bin/nc_cad_probes.exe" $archive "$out/poses-interleaved.json"
foreach ($m in @("cad-bezier-bounds","mesh-obb","cad-bezier")) {
    python benchmarks/aes/cpu_process_timing.py $m $scene "build/depth-runs/$m" "$bin/nc_contact.exe"
}
python benchmarks/aes/cpu_depth_results.py $scene build/depth-runs $archive $out
python benchmarks/aes/cpu_pressure_step_replay.py $scene build/depth-runs/cad-bezier build/depth-runs/cad-bezier-bounds "$bin/nc_contact.exe" "$out/step-refinement.json" --current-method cad-bezier-bounds --window-steps 4
```

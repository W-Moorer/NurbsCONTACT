# 双级减速器接触动力学复核

[响应曲线全集](mechanism-responses.pdf) 包含三根轴的转速、两级传动误差、三根轴的接触力矩、输入功率、制动功率和转子动能变化。每个物理量均同时展示完整载荷历程和制动加载局部。其后给出啮合频谱、步长细化、固定姿态力矩和双球解析解。原始时程没有滤波或相位平移。[逐通道指标](response-metrics.csv) · [可读取的时程数组](physical-responses.npz) · [RecurDyn 原始导出通道](native-exports)

## 尖峰的来源

在 1.2527080078125 s 的一个已接受子步，旧 GPU 查询将一个点判为穿入 47.04 µm。CPU 对目标齿轮原始 STEP 实体分类并求距，该点位于外部 17.64 µm。采用同一齿轮的细三角网格和独立 libigl 有符号距离查询，得到外部 17.75 µm。旧查询给出的接触法向长度为 459.12，在中间轴引入 −6,852.37 N·m 的虚假力矩。同一姿态重新查询时，实际参与承载的三组面合计约 −77.91 N·m。[局部对照图](switch.pdf) · [逐接触点及 CPU 复核](switch.json) · [实际受力面和力矩](accepted-face-torques.csv)

修复位于公共 GPU 最近三角形查询中。对候选三角形重新投影，检查投影距离与搜索返回的距离。两者不一致时，从根节点用双精度重查 BVH，并清除该点的局部候选缓存。穿入深度和方向使用同一个投影向量。保留接触刚度、阻尼、齿面分组和积分参数。两个分支在相同的 20,016 个历史输出后接续相同状态与缓存，1.25275 s 的中间轴转速从 65.06 恢复为 94.60 rad/s，原 RecurDyn 运行给出 93.47 rad/s。这一主步的接触及动力学求值从 171 次降至 16 次。整步保留的修正后接触点均通过单位法向检查。[同姿态双向几何回放](fixed-poses.csv.gz) · [各次 HHT 试算及其判定](audits)

## 三秒载荷历程

保持 4 件完整 STEP 零件、3 个独立轴坐标、2 个啮合副及相同的轴惯量、支承和载荷。输入名义转速 1,800 rpm，总传动比 4。制动力矩在 0.10 s 附近增加，在 0.24 s 附近回落，并叠加周期扰动。两个 GPU 后端都使用双向采样、相同的区域力模型和 HHT。STEP (ours) 在原始曲面和边上放置源采样点，纯网格在平面三角形和直边上放置源采样点。二者在线均查询目标三角形。RecurDyn 导入相同 STEP 实体，使用原生实体接触和原生积分器。

下表在 24,001 个共同输出时刻计算 RMS 差值。速度单位为 rad/s，传动误差单位为 mrad。RecurDyn 采用默认接触面片和 125 µs 最大步长。两组 GPU 使用 62.5 µs 主步，并在需要时细分。

| 几何方案 | 输入轴速度 | 中间轴速度 | 输出轴速度 | 一级传动误差 | 二级传动误差 |
|---|---:|---:|---:|---:|---:|
| 修复前 STEP | 5.793 | 1.657 | 1.630 | 1.31 | 1.43 |
| 修复后 STEP (ours) | 3.333 | 0.924 | 0.907 | 1.036 | 1.221 |
| 修复后纯网格 | 3.341 | 0.925 | 0.909 | 1.023 | 1.212 |

三根轴的平均转速在两套求解器中均约为 188.50、94.25、47.12 rad/s。修复后 STEP 与原生解的平均输出接触力矩分别为 138.904 和 138.899 N·m，平均制动功率分别为 6,545.61 和 6,545.57 W。输出接触力矩的瞬时历程仍有显著差异，在共同输出间隔上其 RMS 差为 60.69 N·m。两级传动误差均值分别偏低 0.970 和 1.132 mrad。

在 RTX 3090 上关闭试算记录后，从零运行 3 s，修复后的 STEP 用时 271.09 和 271.72 s，修复后的纯网格用时 308.33 和 306.73 s。四次的每个输出量都与各自带记录的轨迹逐位相同。默认 RecurDyn 完整运行的在线耗时为 594.44 s。以上计时不含各次模型准备，也不含 211.7 s 的一次性 STEP 采样准备。两个 GPU 后端在全部 48,000 个主步上均超过实时期限。[运行记录](timings.json)

以相邻 125 µs 输出的中间轴速度差计算，修复前、修复后和默认 RecurDyn 的 RMS 分别为 1.868、0.506 和 0.854 rad/s。修复前的局部假尖峰已由原始 CAD 几何独立确认。对周期振动，三组模型均可辨识 330 和 720 Hz 两个啮合频率，幅值依赖接触离散。在输入轴的 720 Hz 处，修复后 STEP 的转速功率谱密度为 0.0098，默认 RecurDyn 为 0.300，单位 (rad/s)²/Hz。该频谱使用 0.5–3 s 数据、Hann 窗、4,000 点每段及 50% 重叠。[频谱数值](mesh-frequency.csv)

## 接触离散与积分细化

从原生 RecurDyn 导出的计算输入中逐件清点，默认接触面片数为 2,200、3,596、8,548、6,292。将两侧的接触平面容差因子乘以 1/16，面片数成为 9,552、13,464、164,344、120,032。以 GPU 减速器的 1 µm 标称弦差三角网格作共同几何基准，第二级输出齿轮的原生面片中心至该网格的距离第 95 百分位从 57.28 µm 降至 1.07 µm。几何统计取整件零件所有面片中心，每个面片权重相同。[默认原生面片](native-default-patches.json) · [加密原生面片](native-refined-patches.json) · [FunctionBay 实体接触面片选项](https://help.functionbay.com/2024/RecurDynHelp/Professional/Professional_ch04_s02_06.html)

在 0–0.32 s、相同 125 µs 原生最大步长和同一 GPU 轨迹上，仅加密原生接触面片，三轴速度 RMS 差由 3.574、1.033、1.048 降至 1.541、0.406、0.430 rad/s。两级传动误差对加密原生结果的 RMS 差为 0.251、0.461 mrad。单独做时间细化，GPU 主步从 62.5 降至 31.25 µs 时，中间轴速度的两次解之差为 0.110 rad/s。默认接触面片上 RecurDyn 主步从 125 降至 31.25 µs 时，独立两次解在同一通道上相差 1.935 rad/s。两种求解器均使用同一时标直接对齐。

## 响应定义与独立验证

- 转速投影到各轴自己的正向。传动误差使用原生累计关节角的初值增量，按每级小轮角增量减两倍大轮角增量计算。所有角度统一为弧度。
- 对自由运动的两条轨迹，每根轴使用同一 125 µs 区间的角动量变化，扣除区间平均的支承和外力矩，得到该区间的合接触力矩。平滑的外力矩在区间端点作梯形积分。对单一固定姿态，则直接用原生瞬时角加速度乘轴惯量，再扣除原生支承和外力矩，与 GPU 在该姿态的瞬时力矩比较。二级输出轴只接收二级啮合的接触力矩，中间轴接受两级合力矩。[通道单位、支承符号和载荷核验](channel-checks.json) · [默认原生姿态下的力矩](fixed-native-torques.csv)
- 输入功率取驱动转矩乘输入轴转速，制动吸收功率取制动力矩乘输出轴转速的负值。动能变化按三根轴各自的转动惯量和原始转速计算。
- 双球线性接触有独立闭式位移与力时程。两球各 1 kg，总相对速度 0.45 m/s，等效接触半径每球 61 mm，刚度 5,000 N/m。125 µs 时，源和目标的离散弦差从 8 降至 2 µm，将通用 STEP 位移 RMS 误差从 14.136 降至 3.509 µm。原生 RecurDyn 球–球专用接触为 0.291 µm。力 RMS 误差分别为 0.02797、0.00713、0.000952 N。解析峰值为 22.5 N。[三种步长与两种弦差](sphere-exact.json)
- 偏心凸轮的 20 s 完整回归以解析几何支承与独立 DOP853 积分作参考。修复后 71 次接触切换全部按序检出，位移 RMS 和最大误差为 0.164 和 0.927 µm。[凸轮核验](cam-check.json)

## 同一刚度代表哪部分接触

已有区域模型给每个接触区域配置刚度 $k$。若工程输入的是整个刚体对的集中刚度，程序也支持将 $k$ 施加一次，接触弹性能为 $k(\max_i d_i)^2/2$，其中 $d_i$ 是各区域的穿入深度。集中模型仍检索全部区域，以最深点形成合力和力矩。它只预测整对刚体的合反力。不同齿各自的载荷仍需要多接触区域模型。整个试验使用同一 $k=1.6\times10^8$ N/m，没有按齿轮调整系数。

在制动力上升段的 25 个加密原生姿态上逐一计算输出轴瞬时接触力矩。RecurDyn 平均为 176.59 N·m，区域弹簧求和为 597.79 N·m，集中弹簧为 205.56 N·m。两种 GPU 模型相对原生的逐姿态 RMS 差分别为 428.09 和 35.72 N·m。即使自由运转的两条曲线平均传递力矩一致，固定姿态试验也能揭示载荷与变形关系的差异。用 0.5 s 以后的独立时间窗复验该建模选择。在只出现一个接触区域的双球算例，两种刚度定义的 1,281 个输出时刻所有通道逐位相同。[加载段固定姿态反力](brake-dense-torques.csv) · [双球一致性检查](single-region-control.json)

### 加密原生解的独立复核

加密 RecurDyn 从初始状态一直算到 1.26 s，建模和在线共 23 分 52 秒，前 0.32 s 所有共同响应逐点复现先前的短程结果。1.25275 s，中间轴速度为 94.579 rad/s。修复后的区域模型为 94.603 rad/s，集中模型为 94.313 rad/s。在 1.251–1.254 s 的另 25 个加密原生姿态上，输出轴瞬时接触力矩的均值分别为原生 144.96、区域 486.00 和集中 168.14 N·m。相对原生的逐姿态 RMS 差由 347.02 降至 29.65 N·m。[对应姿态下的原始反力](fixed-dense-native-torques.csv)

下表在相同的 **0–1.26 s** 对加密 RecurDyn 计算整段 RMS 差值。默认 RecurDyn 一行保留相同求解器、步长和物理参数，单独改变接触面片。集中刚度试验使用和区域试验相同的 HHT 设置。

| 方法 | 中间轴速度 (rad/s) | 二级传动误差 (mrad) | 输出轴力矩 (N·m) |
|---|---:|---:|---:|
| RecurDyn 默认离散 | 0.902 | 0.900 | 48.67 |
| 纯网格，每区域刚度 | 0.333 | 0.383 | 41.82 |
| STEP，每区域刚度 (ours) | 0.332 | 0.392 | 41.41 |
| 纯网格，每刚体对刚度 | 0.205 | 0.123 | 17.62 |
| STEP，每刚体对刚度 (ours) | 0.210 | 0.128 | 17.55 |

在未用于提出集中刚度对照的 0.50–1.26 s 区间，中间轴速度、二级传动误差和输出力矩的 RMS 差由 0.300 rad/s、0.365 mrad、40.02 N·m 降至 0.146 rad/s、0.101 mrad、9.93 N·m。平均输出轴力矩为集中模型 136.123 N·m、加密原生 136.121 N·m。相邻输出的中间轴速度变化 RMS 为区域 0.492、集中 0.032、加密原生 0.184 rad/s。两种 STEP 配置在高频波动幅值上分别偏高和偏低。集中刚度下 3 s GPU 在线时间为 STEP 212.00 s、纯网格 234.79 s，两者均未达到整机实时。加密 RecurDyn 的 1.26 s 算例实际在线用时 1,330.82 s。[同窗频谱](native-refinement-spectrum.csv) · [运行时间](timings.json)

## 空间采样的独立对照

双球在 2 µm 弦差和 125 µs 步长下加入各曲面的顶点作为接触候选，原始网格朝向的位移和力 RMS 误差分别降至 0.0984 µm 和 0.000443 N。将两球离散网格分别绕三个轴旋转，保持连续球面、质量、速度和冲击方向相同，再运行同一程序，两项误差变为 0.791 µm 和 0.00180 N。旋转后的面内采样与增补顶点得到相同结果。该控制把相对于解析解的误差与几何离散朝向联系起来。作为整机控制，对同一减速器也加入顶点，前 0.32 s 对加密原生解的中间轴速度 RMS 差反而由 0.406 升至 0.611 rad/s。因此整机主表保留原有的面与边采样，顶点增补作为可选的空间分辨率试验。[取向与步长对照](sphere-coverage.json) · [原始时程](sphere-coverage-traces.npz) · [分离后的恢复系数和动能](sphere-rebound.json)

## 复现

直接从发布的数组在 CPU 上重新绘图及计算响应指标，只需运行 `python benchmarks/aes/reducer_physical_responses.py results/aes/resident-reducer/physical build/reducer-replot --replot`。时程包含每个原始共同输出，力矩取区间中点作为横坐标。

从仓库根目录运行。GPU 驱动使用 CuPy/CUDA，模型准备和独立 STEP 核验使用 OpenCascade。原生动力学计算需要 Windows 上的 RecurDyn 2023 ProcessNet。下面的命令使用 PowerShell。每次仿真的建模和在线总预算为 1,800 s，离线建网单列。

```powershell
$w = "build/resident-reducer"
$o = "results/aes/resident-reducer/physical"
python benchmarks/aes/resident_reducer.py $w
foreach ($method in "cad","mesh") {
    python benchmarks/aes/resident_assembly.py $w "$w/full-$method.json" --trace "$w/full-$method.npy" --method $method --step-us 62.5 --ticks 4096 --contact-samples 16 --explicit-crossing --audit-window 1.251 1.254 --audit-regions-at 1.25275 --audit-output "$w/full-$method.npz"
}
python benchmarks/aes/recurdyn_reducer.py $w "$w/rd-full.json"
python benchmarks/aes/recurdyn_reducer.py $w "$w/rd-physics.npz" --plot "$w/rd-reducer-3s-125us_01/rd-reducer-3s-125us.rplt"
python benchmarks/aes/recurdyn_reducer.py $w "$w/rd-geometric.json" --duration 0.32 --step-us 125 --patch-scale 0.0625
python benchmarks/aes/recurdyn_reducer.py $w "$w/rd-geometric-physics.npz" --plot "$w/rd-reducer-0.32s-125us-patch0.0625_01/rd-reducer-0.32s-125us-patch0.0625.rplt"
python benchmarks/aes/reducer_native_patches.py $w "$w/rd-reducer-3s-125us_01/rd-reducer-3s-125us.rmd" "$o/native-default-patches.json"
python benchmarks/aes/reducer_native_patches.py $w "$w/rd-reducer-0.32s-125us-patch0.0625_01/rd-reducer-0.32s-125us-patch0.0625.rmd" "$o/native-refined-patches.json"
python benchmarks/aes/resident_impact_check.py build/resident-impact build/aes-benchmark/recurdyn-analytic $o
python benchmarks/aes/resident_impact_check.py build/resident-impact build/aes-benchmark/recurdyn-analytic $o --coverage-only
```

时间细化使用同一驱动。局部 GPU 控制文件名为 `fine-16` 和 `fine-32`，两者均使用 `--duration 0.32 --step-us 31.25`，分别使用 16 和 32 个接触半周期采样。三秒的 125 µs GPU 控制保存为 `local-cad125`。默认原生面片的短程时间细化使用 `--duration 0.32 --step-us 31.25`，将原生曲线另存为 `rd-refined-physics.npz`。

复现历史尖峰时，使用提交 `32e25bf` 的公共接触核，保留本次的试算记录接口。在兼容的独立源码目录，用旧核运行至 1.254 s，按上述窗口输出 `switch-cad-16` 和 `switch-mesh-16`。使用旧核生成全程的 `cad-h62.npy`、`mesh-h62.npy`。核对相同起点时，将旧核作为 `--prefix-contact` 传给新驱动，使用同一窗口，将输出命名为 `same-start`。用旧核及 `--audit-samples 32` 生成 `switch-cad-32`。解算明细另有逐面记录 [audits](audits)。回放读取旧轨迹和原生关节角，再对两个来源的采样点重新执行双向冷查询。

```powershell
python benchmarks/aes/reducer_switch_replay.py $w "$w/switch-detail.csv" --audit "$w/switch-cad-16.npz"
python benchmarks/aes/reducer_switch_results.py $w $o
python benchmarks/aes/reducer_physical_responses.py $w $o
python benchmarks/hht/paper.py results/hht
```

增补原生几何和刚度对照时，延长加密模型到 1.26 s，输出命名为 `rd-switch-geometric.json`，并将相应的原生图文件提取为 `rd-switch-geometric-physics.npz`。固定姿态重放可使用 `--native-only` 指定该文件，时间区间用 `--start` 和 `--end` 选择。制动力上升窗口为 0.099–0.102 s，齿面切换窗口为 1.251–1.254 s。实体对集中刚度试验在两个 GPU 几何后端上均使用 `--stiffness-scope pair`。默认 `region` 为每个接触区域配置一根弹簧。

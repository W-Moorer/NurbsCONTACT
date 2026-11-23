# CPU 节点–NURBS 曲面接触对照

本项独立实现 Ansys Motion 2025 R1 公开的 **Node Only** 几何条件。引用的 §6.3.1.1.2 给出了 CAD NURBS 曲面及其切向量和法向量，§6.4.2.1.1 给出了源节点到目标曲面的正交投影条件和沿法向的间隙。表中将此项标为 **CPU node-to-NURBS (independent baseline)**。商用求解器另有边–曲面和曲面–曲面模式。[曲面表示](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/motion_theory/motion_theory_contact_geometry.html) · [接触搜索](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/motion_theory/Detecting_Method_of.html)

### 求解定义

输入是两个以毫米为单位的封闭 STEP 实体。Open CASCADE 把面转换成有理或非有理 B 样条，保留裁剪边界和朝向。源端的三角剖分提供节点的参数坐标，节点位置和法向从所属 CAD 面求值。给定节点 \(p\)，CPU 在目标实体的全部边界面上数值求最近点 \(q=S(u,v)\)，重新求值两个曲面导数，以及朝外的单位法向 \(n\)。对光滑点，\(p-q\) 与两个切向量垂直。程序保存 \(g=(p-q)\cdot n\)，正值表示分离，负值表示穿入。输入点、位置误差和间隙均以 mm 计。极点使用曲面导数的极限法向。若最近位置在锐边且没有满足正交条件的面法向，点查询输出面号零和未定义的法向。如果这种位置可能控制整对接触力，整对查询会报告未解。

动力学试算在两个方向上检查节点，使用项目已有的刚体对集中刚度，按有效节点的最大穿入深度计算一根法向弹簧。每个节点的深度上界取自包围该目标实体的盒子：从实体内的点走到任一盒面，必先经过实体边界。该上界小于已找到的深度时可跳过节点。余下的节点使用 CAD 实体分类和曲面求距。这个筛选、OCCT 的数值搜索、刚度汇总以及项目 HHT 积分器是本仓库的实现选择。计时使用本机 CPU，包括各次试算和结果输出，将 STEP 读取和预处理单列。复现实验对每次独立调用设置 1,800 s 墙钟上限。

### 核验结果

在 AMD Ryzen 9 5900X 上使用 Release 版 OCCT 7.9.3。球半径为 61 mm。90 个节点查询覆盖外部、表面和内部，以及接缝和两个极点。相对解析球面，最大位置误差为 3.12 × 10⁻¹¹ µm，单位法向最大向量差为 8.31 × 10⁻¹⁶。盒体裁剪检查也覆盖了最近点落在真实锐边的情况。[全部指标](metrics.json)

双球用仓库已有的等质量模型和线性法向弹簧，按 125 µs 步长算满 0.16 s。位移误差汇总两球在 1,281 个输出时刻的位移，轴向力使用同一时刻的半正弦解析解。每行是一次完整运行。旋转试验使两球分别绕方向 (1, 2, 0.5) 旋转 0.47 和 −0.329 rad，两球连续几何、质量和速度保持一致。[完整响应与解析解](sphere-traces.npz)

| 源节点布局 | 每球节点数 | 位移 RMS（µm） | 轴向力 RMS（N） | 预处理（s） | 在线（s） |
|---|---:|---:|---:|---:|---:|
| 球极点位于冲击轴 | 1,795 | 0.0878 | 0.000219 | 0.18 | 54.71 |
| 旋转，同一节点密度 | 1,795 | 91.57 | 0.1811 | 0.17 | 38.28 |
| 旋转，加密节点 | 30,508 | 2.208 | 0.004590 | 3.86 | 34.15 |

两档源节点采用的三角化弦差设置分别为 0.5 和 0.025 mm，角度设置均为 0.5 rad。目标曲面求距使用相同的 CAD 模型和数值设置。计时为各布局的一次 CPU 运行。

在减速器第二级，旧算例的指定节点在 1.252708 s 位于第 106 号齿面外 17.63836 µm，与原始 STEP 复核值一致。另取加密 RecurDyn 在 1.25275 s 的轴姿态，将两个完整齿轮交给同一双向节点查询。使用原有的每对刚体集中刚度 1.6 × 10⁸ N/m。下表报告单个姿态的求值时间。[同姿态的 RecurDyn 与采样方案](../resident-reducer/physical/fixed-dense-native-torques.csv)

| 源弦差设置（mm） | 小轮 / 大轮节点数 | 代表点穿入（µm） | 输出轴力矩（N·m） | 几何准备（s） | 单姿态查询（s） |
|---|---:|---:|---:|---:|---:|
| 0.25 | 3,842 / 5,160 | 20.082 | 198.97 | 8.04 | 22.82 |
| 0.025 | 25,352 / 23,929 | 20.198 | 200.43 | 11.69 | 178.80 |

同一姿态下，原 STEP 采样–目标网格方法的集中刚度结果为 199.39 N·m，加密 RecurDyn 给出 145.93 N·m。把目标换成原始曲面并加密源节点，仍然保留了这个瞬时力矩差异。

### 复现

从仓库根目录构建带 OCCT 的 Release 版本，随后运行校验脚本。Windows 下需将该 OCCT 发行包及其第三方运行库目录放到进程的 PATH 中。

~~~powershell
$occtConfig = (Resolve-Path 'build/deps/opencascade-*/cmake').Path
cmake -S . -B build/aes-occt "-DOpenCASCADE_DIR=$occtConfig"
cmake --build build/aes-occt --config Release --target nc_nurbs_baseline
$occtBins = @((Resolve-Path 'build/deps/opencascade-*/win64/vc14/bin').Path) + @(Get-ChildItem 'build/deps/3rdparty-vc14-64/*/bin' -Directory | Select-Object -ExpandProperty FullName) + @((Resolve-Path 'build/deps/3rdparty-vc14-64/openvr*/bin/win64').Path)
$env:Path = ($occtBins -join ';') + ';' + $env:Path
python benchmarks/aes/check_nurbs_baseline.py build/aes-occt/Release/nc_nurbs_baseline.exe build/nurbs-node-reference results/aes/nurbs-node-baseline
~~~

可执行文件的 `points TARGET.step output.csv` 从标准输入逐行读取目标局部坐标系的 \(x,y,z\)，输出间隙、CAD 最近点、单位法向、切向残差、面编号、法向求值所用的导数阶数和两个参数方向的切向量。面编号采用转换后边界的索引。`pair A.step B.step deflection_mm output.csv` 逐行读取一对位姿，每个刚体用世界坐标下的平移 \(x,y,z\)（mm）、转轴 \(a_x,a_y,a_z\) 和转角（rad）表示，返回穿入深度、从 A 指向 B 的接触法向及两个世界坐标作用点。

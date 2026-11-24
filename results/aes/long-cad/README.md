# STEP 曲面接触与平三角网格的 5 s 双级减速器对照

**归档的 GPU 实验。** 本页计时来自独立 CUDA 求解器。当前论文的接触方法对照通过 CPU 统一入口 `nc_contact` 进行；本页的二次曲面片方案已移出创新点。CPU 长时程对照见[统一后端实验](../cpu-backend-long/README.md)。

四个零件，两级啮合，共用中间轴。各方法从同一初态开始，使用相同的轴惯量、外载、线性法向刚度、HHT 积分器及 0.1 ms 主步长。网格拓扑来自同一批 65,732 个 Gmsh 三角形。两侧都提供采样点。每个刚体对由最深受压点贡献一份 1.6×10⁸ N/m 的法向弹簧力。每个正式运行独占本次实验的 GPU 队列，达到 5 s 后保存 50,001 个输出状态。通用近场几何在双方刚体的局部坐标系中工作。减速器装配驱动使用三个转轴自由度。

### 哪些计算值得放到仿真之前

对每个零件一次性求出二次曲面片、它与平三角形的最大可能偏离，以及局部候选表。在线先算便宜的平三角距离。如果查询点在平三角实体外，且曲面可能偏移的范围也碰不到该点，就跳过曲面投影。其余候选在 GPU 上求有界参数域内的最近点。换载荷、转速或初始姿态时，沿用同一零件的几何和索引。整个预处理不用提供齿数、啮合线或从之前运动中筛选出来的齿对。

已有工作给出了两项可直接借鉴的技术。CGAL 的 AABB 树用位于几何体上的初始最近点缩短距离搜索，Johnen 等用 Bernstein 系数检查高阶单元的雅可比行列式。本实验分别测量了最近片复用能节约多少整机时间，以及曲面校验能发现多少不合格的片。[CGAL 最近距离查询](https://doc.cgal.org/5.6.3/AABB_tree/classCGAL_1_1AABB__tree.html) · [高阶曲面单元的几何有效性](https://gmsh.info/doc/preprints/gmsh_curved_preprint.pdf)

### 完整 5 s 动力学的实测耗时

下表均为 RTX 3070 上分别运行一次的结果。三轴 HHT、两级双向接触和法向力计算均计入在线时间。16 表示每个有效接触半周期要求的时间分辨率，HHT 根据局部频率和收敛情况选择实际子步。最快的网格使用每查询 4 个 GPU 线程，最快的曲面版使用 8 个线程。初始化、生成模型和渲染另计。

| 方法（接触半周期取 16） | 整段在线 (s) | 其中接触搜索 (s) |
|---|---:|---:|
| 纯网格 | 120.37 | 61.89 |
| CAD 采样、平三角目标 | 120.26 | 62.48 |
| STEP 曲面，逐候选双精度搜索 | 364.74 | 304.78 |
| STEP 曲面，先按曲面偏离量筛候选 | 287.74 | 227.29 |
| 同上，关闭上次最近片复用 | 283.06 | 223.91 |
| **STEP 曲面，加入平三角实体外部判定（ours）** | **271.54** | **211.03** |
| STEP 曲面，另建一层随姿态更新的小候选集 | 344.81 | 253.86 |

从逐候选双精度搜索到外部判定，整段少用 93.20 s（25.6%）。这些 16 分辨率的曲面版本在全部 50,001 帧上，14 列原始动力学输出和两级承载面编号逐值相同。关闭上次最近片复用快了 4.68 s，单次测量不能支持这项复用有独立收益。另建小候选集使四个方向的采样点访问总计从 68.04 亿次减至 17.10 亿次，但需重建约 317 万次，准备试算的时间从 29.2 s 增至 59.4 s。优先采用外部判定并保留一层运动候选集。[时间分解图](timings.png) · [试算、索引复用和分段时间](comparison.json)

通过 `--method cad-patch` 的默认入口独立复跑同一 5 s，四件索引均自动命中。设置时间 17.36 s，在线时间 271.85 s，其中接触搜索 211.29 s。两次在线计时相差 0.31 s，所有输出和承载面编号逐值相同。

网格与最快曲面版分别做了 160.32 万和 161.21 万次动力学试算。每主步平均分别约 32.1 和 32.2 次。两者整段耗时相差 151.17 s，其中 149.14 s 来自接触搜索。曲面版显存池占 230.92 MiB，网格为 210.65 MiB。两者均超过 0.1 ms 每步的实时期限。此表固定了网格拓扑和积分设置，等几何误差的纯网格耗时仍需用加密网格测量。

保持 0.1 ms 主步长，把接触半周期的分辨率从 16 改为 8 后，优化曲面版用时 **177.09 s**，其中接触搜索 134.83 s；配对的纯网格用时 **79.57 s**，其中接触搜索 40.91 s。两档下的曲面/网格在线耗时比分别为 2.26（16）和 2.23（8）。曲面版 8 相对 16 的一级、二级载荷 RMS 变化为 10.03、12.84 N，输出力矩 RMS 变化为 0.80 N·m。两级各有一个输出帧的承载开关状态不同。降低内部时间分辨率可以缩短计算，但其效果须与几何加速分开解释。

### 几何与缓存

- **Mesh** 在线对平三角形求最近点，以三角形、边和顶点的法向信息计算载荷。
- **CAD 采样、平三角目标** 把源侧点放到 STEP 工作面和公共边，在目标侧仍使用平三角形。这一支能单独检验移动采样位置的影响。
- **STEP 二次曲面片（ours）** 为每个三角域保留三个顶点，并从原生 CAD 求得三个边中点。位于两个 CAD 面交界的中点由同一条 STEP 曲线产生。GPU 在三角参数域中做受约束的曲面投影，用曲面导数求面内法向。边和顶点由相邻面法向判内外，受力方向沿最近点连线。源侧仍采取有限采样，朝向筛选使用源三角形的法向。该实现给出点对曲面片的数值近似，尚未对任意两张裁剪 NURBS 曲面的首次相交作连续检测。

曲面片的几何缓存只依赖 STEP、Gmsh 网格及算法版本。空间索引缓存保存曲面控制量、体素候选表、包围盒树、平面排除界和局部坐标表，按几何、索引参数及设备环境区分。载荷曲线、HHT 状态和上一姿态的最近片编号均不写入可跨工况的缓存。刚体移动时，先用包围盒和体素排除外部区域，再用平三角载体距离与曲率偏移量约束候选。上一片能够排除其他竞争片时，直接在该片上求新的接触点。超出局部表覆盖范围的点回退到完整 BVH。复用最近几何特征的思想可追溯到 V-Clip 的凸多面体算法。[V-Clip 原文](https://graphics.stanford.edu/courses/cs448b-00-winter/papers/TR97-05.pdf)

四件并行首次预处理耗时 156.11 s，几何文件约 3.65 MiB。第二次在已有缓存上检查输入并重建清单耗时 0.20 s。细长斜齿面上的 528 个初拟合片未通过方向检查。重新在相应 88 个 CAD 面的参数空间构造内部节点后，所有片的有向雅可比行列式均通过 Bernstein 系数正性检查。对 197,196 个未参与拟合的参数位置回查相应 STEP 支撑曲面，平片和二次片的最大观测偏差为 74.983 和 0.575 µm。这些是离散核验值。[零件哈希、各件误差与预处理时间](geometry.json)

四件 GPU 空间索引共占磁盘 173.95 MiB。在新目录独立测得首次构建并保存用时 11.58 s，已有索引时运行同一命令用时 5.62 s，四件全部命中。两项合计，曲面几何与 GPU 索引首次准备约 167.69 s，不含双方共用的 Gmsh 建网。在 271.54 s 的动力学运行中，从载入装配到第一个计时步另用 23.31 s，包含 GPU 设置并命中四件索引。复跑时直接启动动力学，无须再单独执行索引命令。[索引首次生成](index-cold.json) · [四件缓存恢复](index-warm.json)

### 对原 STEP 的独立核对

用 OCCT 对整个原生 STEP 外壳求最近距离，用实体分类器判内外。测试点从拟合曲面及其公共边选取，再对原 STEP 独立求参考距离与方向。查询包括面内、锐边角平分线及偏离角平分线的方向，覆盖标称内外偏移 5 和 50 µm。穿透深度为负有符号间隙的正部，外部点记零。法向比较在双方都判定承载的点上进行，参考方向取 OCCT 报告的等距解中夹角最小的一项。

| 四件 896 个查询 | 漏判 | 误判 | 最大穿透深度误差（µm） |
|---|---:|---:|---:|
| 平三角目标 | 25 | 23 | 41.192 |
| STEP 二次片（ours） | 0 | 0 | 0.091 |

四件弯曲面内点的法向夹角 P95 最大值为 0.0305°。锐边角平分线上的分件 P95 最高为 65.1°，128 个加载的角平分线查询中，有 9 个的法向差超过 30°。偏离角平分线时分件 P95 最高为 2.17°。小深度误差与接触法向的可靠性需要分别验证。再让查询点往返移动，对 6,272 次查询分别沿用和清空姿态历史，所测承载标记、深度和法向均相同。[逐件核验](query-check.json) · [独立参考点、法向与逐点结果](point-queries.npz)

将 GPU 查询换成直接搜索或外部判定后，四件 896 个固定点的穿透深度、输出法向与原有界搜索逐值相同。在另一套凸轮、从动件的 1,414 和 12 个三角片上，同一预处理及曲面查询覆盖 368 个点，曲面版零漏判、零误判，最大深度误差 0.0251 µm。这套粗网格在凸轮上漏判 86 个点，从动件是平面，两种表示均与原 STEP 一致。凸轮这组只检验几何。[减速器不同搜索策略](query-variants.json) · [凸轮与从动件的网格及验证](cam-query-check.json) · [凸轮逐点数据](cam-queries.npz)

### 承载状态与独立动力学

在曲面版 50,001 个实际输出姿态上，用 CPU 双 OBB 树和三角片相交重新检查两级，分别耗时 14.04 和 5.48 s。一级的实体重叠与曲面版正法向力逐帧一致。二级在 7.1 ms 有一帧曲面版载荷为 37.75 N，平三角实体判断分离，其余逐帧一致。这组固定姿态查询不含时间积分。[CPU 回放计数与耗时](obb-on-cad.json) · [逐帧相交标志](obb-on-cad.npz)

对 7.1 ms 的二级啮合，从记录的承载面及对侧公共边选取 522 个源点，用 OCCT 逐点检查另一件原始 STEP 实体。在主动轮 STEP 表面上的一点，到从动轮外壳的有符号距离为 −0.18627 µm。该点到主动轮自身外壳的残差低于 4×10⁻¹⁵ mm。由此可以确认原 CAD 在这次姿态已有接触。按模型弹簧刚度，该原生距离相当于 29.80 N；曲面版整对载荷为 37.75 N。在 1.2527 和 1.2528 s，同样检查给出原生点载荷 2329.49 和 2342.70 N，曲面版整对载荷为 2333.33 和 2350.13 N。这里检查的是实际承载面邻域中的原生几何点。[7.1 ms 接触切换图](engagement.png) · [逐点坐标与原生距离记录](native-witness.json)

将上述三个 STEP 点的原坐标逐一送入 GPU，曲面片在同一点给出的等效载荷依次为 37.75、2330.99、2335.32 N。7.1 ms 处点对曲面片给出的深度为 0.23594 µm，比原始 STEP 深 0.04968 µm。几何误差很小，但刚接触时的力只有几十牛顿，仍可产生约 8 N 的差异。

单对集中弹簧每个时刻取最深点。两个几何模型各自进行闭环积分时，输出时间戳相同，实际姿态可有差异。16 细分时两种模型在一级和二级的最深承载面编号相同率为 65.0% 和 40.4%。面编号切换包括同一次持续承载中的接触位置切换。两级法向力时程的模型间 RMS 差分别为 103.1 和 268.8 N。

16 细分时，曲面、网格和 RecurDyn 的平均输出接触力矩分别为 137.806、137.805 和 137.947 N·m。曲面和网格对 RecurDyn 的瞬时力矩 RMS 差分别为 73.45 和 72.76 N·m。在预先选定的 1.245–1.265 s 窗口，三者输出力矩的范围依次是 138.73–145.99、110.92–176.96 和 0–325.30 N·m。整个 5 s 内，曲面版二级有 24 个输出帧为零载荷，RecurDyn 的二级接触合力有 777 个零值帧。曲面版算出的瞬时力矩波动更小。真实齿轮受力是否应当如此平滑，还缺独立的力学依据。判断载荷峰谷与脱开的物理精度，需核对两平台的接触弹性与多齿载荷分配，或取得实测载荷。

[装配全景和齿面细节](../long-mesh/assembly.png) · [转速、传动误差与输出力矩](responses.png) · [两级载荷](loads.png) · [曲面版实际姿态动画](reducer.mp4) · [完整数据](responses.npz) · [各配置的时间、试算次数与切换统计](comparison.json)

### 复跑

从先前网格基线的四个 STEP、对应的 Gmsh 网格及质量清单开始。冷启动需指定新的缓存目录。双方的最小受力深度均为 0.01 nm，接触时域细分在成对比较时取相同值。下例保留每个主步末实际承载的 CAD 面编号，含未加载标记。论文使用的接触切换统计必须注明输出间隔和统计的是面编号变化还是加载状态变化。[网格模型来源](../long-mesh/README.md)

日常使用时，执行一次 `cad_patch_index.py <装配目录>`，即可将索引存入装配目录的 `patch-index`。之后对同一个模型传 `--method mesh` 或 `--method cad-patch` 即可切换。后者自动读取索引，使用平三角实体外部判定，每个查询用 8 个 GPU 线程。以下复现实验显式给出索引位置和运行参数。

```powershell
$w = "build/long-cad"
$parts = "build/cpu-contact/reducer-edge-compact"
python benchmarks/aes/long_mesh_deck.py $parts "$w/input" --duration 5 --step .0001
python benchmarks/aes/cad_patch_cache.py "$w/input" $parts "$w/deck-v2" --jobs 4
python benchmarks/aes/cad_patch_index.py "$w/deck-v2" "$w/index-v2" --pitch .8
python benchmarks/aes/cad_patch_checks.py reference "$w/deck-v2" $parts "$w/checks-v2"
python benchmarks/aes/cad_patch_checks.py check "$w/deck-v2" $parts "$w/checks-v2"
function RunContact($name, $method, $divisions, $lanes, $extraOptions) {
    python benchmarks/aes/resident_assembly.py "$w/deck-v2" "$w/$name.json" --trace "$w/$name.npy" --contact-history "$w/$name-faces.npz" --method $method --duration 5 --step-us 100 --broad-phase blocks --guard-mm .8 --stiffness-scope pair --ticks 4096 --contact-samples $divisions --explicit-crossing --profile-phases --contact-floor-nm .01 --lanes $lanes @extraOptions
}
RunContact mesh-floor16 mesh 16 4 @()
RunContact source-floor16 cad 16 4 @()
RunContact fast16 cad-patch 16 4 @("--index-cache","$w/index-v2","--patch-search","bounded")
RunContact direct16 cad-patch 16 8 @("--index-cache","$w/index-v2","--patch-search","direct","--no-patch-reuse")
RunContact filter16 cad-patch 16 8 @("--index-cache","$w/index-v2","--patch-search","carrier-filter")
RunContact mesh8 mesh 8 8 @()
RunContact winner8 cad-patch 8 8 @("--index-cache","$w/index-v2","--patch-search","carrier-filter")
python benchmarks/aes/long_mesh_replay.py "$w/deck-v2" $parts "$w/fast16.npy" build/cpu-swept/Release/nc_mesh.exe "$w/obb" "$w/obb.json" --methods mesh-obb
python benchmarks/aes/cad_loaded_witness.py "$w/deck-v2" $parts "$w/fast16.npy" "$w/fast16-faces.npz" "$w/native-witness.json" --stage 1 --steps 71 12527 12528
python benchmarks/aes/cad_loaded_witness.py "$w/deck-v2" $parts "$w/fast16.npy" "$w/fast16-faces.npz" "$w/native-witness.json" --stage 1 --check-saved
python benchmarks/aes/long_cad_comparison.py $w "$w/figures" --cad filter16 --mesh mesh-floor16 --controls fast16 source-floor16 direct16 mesh8 winner8 --animate
```

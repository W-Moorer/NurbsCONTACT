# CPU 原 CAD 接触的离线预处理与候选复用

两级减速器，主步长和输出间隔均为 0.1 ms，共推进 5 s。CPU 为 Ryzen 9 5900X，两个接触对并发。CAD 与网格使用同一份场景、共享罚刚度、预测姿态更新和 HHT 设置。每次长跑均从初态开始，单个动力学进程限时 1,800 s。下表为进程墙钟时间，包含读模、在线准备、全部积分步及响应写出。预先生成的零件文件另外计时。

| CPU 方案 | 五秒整机耗时 | 测量条件 |
|---|---:|---|
| 旧 CAD，提交 `717001d` | 1,171.31 s | 首轮，部分时段同时编译 |
| 基础表与候选复用 | 804.52 s | 首轮，新版单独运行 |
| 基础表与候选复用 | 804.39 s | 反序复测，先运行新版 |
| 旧 CAD | 811.05 s | 反序复测，随后单独运行旧版 |
| 全阶预处理 CAD（ours） | **770.41 s** | 四阶积分规则均已存盘，单独长跑一次 |
| 纯网格 OBB | **298.26 s** | 同一模型与求解器设置，单独运行 |

反序复测中基础表方案只缩短 0.82%。全阶运行相对 811.05 s 的旧版少用 40.64 s，是纯网格耗时的 2.58 倍。全阶只测了一次长跑。首轮旧版受到同时编译和主机负载影响。此前同一旧二进制在另一时段测得 379.94 s，跨批次绝对耗时有明显波动。[五秒曲线和计时](cpu-response.png) · [各轮动力学误差和分项](dynamics.json) · [原基线](../cpu-cad-pressure/pruning/README.md)

## 预处理和运行方式

原 STEP 零件按相同的基础搜索规则分出 UV 小块。样条块使用受限参数区间的控制点包围范围，其他支持的面使用原生包围算法，缺少可靠局部包围时使用整面范围。两棵 BVH 分别索引候选点与曲面块。基础规则预存面上通过裁剪检查的积分点和有向面积权重。积分按原本的累加顺序分块，高阶数据按实际访问的面加载。缓存键包含序列化后的原生 B-rep 身份、表版本、OCCT 版本和离散规则，读取时核对保存的几何身份和数据校验值。

每个试算姿态都计算目标刚体在源刚体坐标系中的运动。对根包围盒中心和半长，用两次相对变换的矩阵差给出各坐标方向的最大位移界。超过候选表余量时重建。余量内只遍历保存的子树对，当前姿态下重新检查候选、接触深度和法向。面积积分将一批节点沿积分方向的扫掠范围与目标面的包围范围、可用的 Bézier 控制点范围比较，仅在严格分离时跳过整批。已验证的折线连接可在新姿态逐段复核。有限起点的原始接触发现规则依然可能漏掉很小的独立接触区。

| 四个零件的离线操作 | 单独启动两对零件的合计墙钟时间 | 操作后持久文件 |
|---|---:|---:|
| 从空目录建立基础表 | 50.00 s | 16 个，38.55 MB |
| 重启并加载基础表 | 41.73 s | 16 个，38.55 MB |
| 在基础表上补建 24、48、96 阶积分规则 | 160.34 s | 2,536 个，576.95 MB |
| 重启并显式预加载全部阶次 | 44.60 s | 2,536 个，576.95 MB |

从空缓存分两步建到全阶共 210.34 s。显式全阶预加载计时含 STEP 读入及原生几何结构初始化。在线仿真可直接加载基础表和实际需要的高阶面。[预处理记录](offline-preparation.json)

## 搜索、分组和载荷核对

全阶五秒闭环中，按刚体对的两个查询方向计，83,338 次建立候选表，120,616 次复用。分块筛选跳过 82,720,144 次积分节点访问。历史折线的有效复用为零。固定姿态交替计时，每级取 8 个窗口，每个窗口预热 20 帧后保留 80 帧。基础表方案在一级的几何搜索与分组平均从 2.375 增至 2.701 ms，二级从 7.882 降至 7.118 ms。候选复用率本身不能换算成耗时收益。[整机计数](all-orders-run.json) · [交替测量](paired-window-timing.json)

全阶版与旧版在 50,001 个相同轴姿态、两个级别上的命中面对、分组和区域数逐帧一致。56 个固定姿态的合力、力矩与旧版归档逐一相同。0.5 ms 固定网格姿态下二级力矩为 533.953 N·m，完整 STEP 边界独立积分为 533.53 N·m，同姿态网格为 411.95 N·m。在各自独立的闭环轨迹上，正式版与反序旧版的五秒输出力矩 RMS 差为 0.143 N·m，最大差为 10.266 N·m。在最大差异处 3.5377 s，将两种算法放到同一姿态，两者均算得 149.468 N·m（旧轨迹姿态）或 138.336 N·m（全阶轨迹姿态）。相对同模型 RecurDyn 的五秒输出力矩 RMS 差为 CAD 42.164 N·m、网格 29.142 N·m。[同姿态核对](all-orders-same-poses.json) · [逐姿态合力和力矩](all-orders-fixed-loads.json) · [峰差交叉复查](peak-poses.json) · [逐步响应](responses.npz)

独立双球正碰采用同一步长和 5 s 时长。新旧版响应文件的 SHA-256 相同，峰值均为 22.49913 N，解析解为 22.5 N。[双球解析对照](all-orders-sphere.json)

## 复算

按[基线说明](../cpu-cad-pressure/pruning/README.md)准备 Release 可执行文件、DLL 和减速器场景。在 PowerShell 中设置 `$bin` 和 `$scene` 后执行以下命令。`$parts` 指定模型实际使用的 STEP 零件路径。常规 `cad-patch` 调用会自动生成缺失的表。

```powershell
$env:OMP_NUM_THREADS = "2"
$env:NC_CAD_CACHE_DIR = "G:\NurbsCONTACT\build\my-part-cache"
& "$bin/nc_cad_probes.exe" --prepare-parts @parts
python benchmarks/aes/cpu_backend_run.py "$bin/nc_contact.exe" $scene build/my-cad-run --method cad-patch --solver penalty --penalty-stiffness shared --geometry-update predictor --spectral-radius 0
python benchmarks/aes/cpu_backend_run.py "$bin/nc_contact.exe" $scene build/my-mesh-run --method mesh-obb --scope contour-ray --solver penalty --penalty-stiffness shared --geometry-update predictor --spectral-radius 0
```

固定姿态用 `cpu_cad_pressure.py` 的 `prepare`、`indexed`、`area --order 0` 和 `windows --old-indexed` 重建。完整响应比较和图由 [Python 报告脚本](../../../benchmarks/aes/cpu_cad_offline.py) 生成。并发冷启动和损坏表恢复可由 [缓存回归测试](../../../tests/test_cad_patch_assets.py) 核验。

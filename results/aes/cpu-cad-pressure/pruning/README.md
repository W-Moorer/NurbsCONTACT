# CPU 原生 CAD 接触的分离检查与五秒闭环

在统一 CPU 后端选择 `--method cad-patch` 即可使用距离剪枝。刚体仍从 STEP 建模。搜索暂时找不到承载面时，需要判断两个实体确实分离，才能返回零接触力。旧版在这一步对大量面对、面边对调用 OCCT 最近距离核。新版利用原始曲面的控制点和裁剪边界，先排除在当前距离范围内不可能接触的几何特征。该判定不使用齿数、传动比或预先指定的啮合面。

## 同一减速器的完整对照

Windows 11，AMD Ryzen 9 5900X（12 核），Release 构建。两级减速器从初态分别闭环推进 5 s，主步长和输出间隔均为 0.1 ms。各方法共用载荷、共享罚刚度、HHT 参数和预测姿态更新，均关闭额外 CCD。网格使用 CPU OBB 粗筛和三角面接触区。表中耗时包括读模、预处理、积分和写出。每个原生进程单独限时 1,800 s。[算例模型和原先的 STEP 积分对照](../README.md)

| CPU 方法 | 总耗时（s） | 前 20 步积分（s） | 后 49,980 步积分（s） |
|---|---:|---:|---:|
| 旧 CAD，单独复跑 | 445.10 | 94.60 | 322.92 |
| CAD 距离剪枝（ours），独立构建 | 374.22 | 19.68 | 331.78 |
| CAD 距离剪枝（ours），正式构建 | **379.94** | **20.37** | **334.05** |
| CPU 三角网格 OBB | 134.83 | 0.27 | 129.74 |

相对单独复跑的旧 CAD，两次默认算法的整机耗时分别减少 15.9% 和 14.6%，相对同一网格运行的耗时比分别为 2.78 和 2.82。以正式构建为例，前 20 步节省了 74.22 s，余下积分多用了 11.13 s。旧版首轮在几何诊断同时进行时用时 519.79 s，上表采用单独运行的旧版作比较。所有方法均在 30 min 以内完成。[逐次运行指标](production-comparison.json) · [正式构建运行记录](production-run.json) · [整段计时与响应曲线](production.png)

### 距离下界如何减少查询

1. 对原 CAD 中有正权重的样条面和边，用控制点凸包在指定方向上的投影给出距离下界。找到足以判定分离的方向即可跳过核验。找不到时，继续计算原 STEP 特征距离。
2. 对有孔或外轮廓弯曲的平面，将对侧特征可能靠近平面的部分投影到平面参数域。只有整个投影都避开原裁剪曲线的保守包围范围，且落在孔内或外轮廓之外，才排除这一对。实体包含仍由原有分类器检查。

在完全相同的初态，一级进入 OCCT 最近距离核的特征对从 423 减到 116，二级从 1,254 减到 87。旧、新、新、旧交替运行，一级距离查询从 3.54–3.77 降到 1.51–1.53 s，二级从 4.13–4.15 降到 0.84–0.90 s。对原 STEP 整壳直接计算的距离分别为 0.046797 和 0.040085 mm，均高于本项测试的查询上限。[同姿态计数和整壳距离](startup-static.json) · [距离实现](../../../../src/cad_boundary_distance.cpp) · [平面裁剪](../../../../src/cad_planar_trim.hpp)

还测试了更细的射线剪枝和共享曲面缓存。两项一起启用时，五秒闭环在首轮从 519.79 降到 407.87 s，反序单独运行时从 445.10 变为 446.36 s。缩短启动核验是两轮都能复现的收益。为此正式版保留原有的持续接触搜索、分组、面积积分及射线求交，只在分离检查中启用新的距离下界。[试验组合的实现、分项和完整结果](combined-audit.md)

## 力矩和接触切换

两种 CAD 在各自的 0.5 ms 闭环姿态都输出 433.2725 N·m。正式版相对单独运行的旧版，五秒输出轴力矩 RMS 差为 0.1003 N·m，最大差 10.2656 N·m 出现在 1.5377 s。正式版相对同模型 RecurDyn 的输出轴力矩 RMS 差为 42.164 N·m，网格为 29.142 N·m。现有证据支持减少分离核验成本，整段力矩精度仍应按所用参考独立评价。[全分辨率力矩、轴转速、传动误差和法向载荷](production-response.npz)

在网格闭环导出的 50,001 个共同姿态上，两级共比较 100,002 次。新旧 CAD 命中的面、接触区成员、区数和法向逐帧一致，接触点位置最大差 2.78 × 10⁻¹⁷ m，深度最大差 1.91 × 10⁻¹⁷ m。另对一级 24 个、二级 32 个固定姿态，用正式版重新积分各区载荷。56 处的三维合力、三维力矩和从动轴力矩全部与旧版归档值一致。0.5 ms 的二级同姿态积分力矩为 533.953 N·m。[全程接触切换和分组核对](production-same-poses.json) · [固定姿态合力与力矩](production-fixed-loads.json)

在 1.5377 s，把两版都放到旧轨迹的已接受姿态，两版各自找到三个接触区，力矩均为 149.468 N·m。换成正式版轨迹的姿态，两版又均为 138.336 N·m。这一步的载荷对姿态敏感，固定姿态后的几何查询给出相同结果。引起两次闭环轨迹最早分歧的具体一步尚未定位。[最大差异处的同姿态复查](production-peak-poses.json)

另用相同 CPU 入口运行自由双球正碰 5 s。新旧版 50,001 帧响应文件逐字节相同，峰值均为 22.49913 N，解析值为 22.5 N。旧版耗时 16.40 s，正式版 16.53 s。双球用例展示了这一改动的适用范围，简单分离核验在该用例中没有带来整机提速。[双球计时和解析对照](production-sphere.json) · [带孔平面、斜穿、旋转和有理样条球回归测试](../../../../tests/cad_solids.cpp)

同姿态比较与独立闭环比较回答不同的问题。在网格轨迹给定的 0.5 ms 姿态，完整 STEP 实体独立积分给出 533.53 N·m，纯网格给出 411.95 N·m。两种动力学求解器分别推进后，会到达稍有差别的轴姿态。初态以后各自轨迹上的两列力矩应按对应姿态解释。[同姿态对照数据](../wrenches.csv) · [两个求解器已接受姿态的独立重算](../accepted-0p5ms.json)

## 复现默认配置

从仓库根目录使用[上一级文档](../README.md)的模型和 Release 构建流程。以下命令在 PowerShell 下运行。`$oldBin` 使用提交 `68c099c`，`$bin` 为正式版，`$work` 为独立输出目录。对照用的 `distance-only` 也是同一距离算法和原有射线路径，可用独立构建的 `$distanceBin` 重跑。五秒长跑应分别执行。

```powershell
$env:OMP_NUM_THREADS = "2"
$run = "benchmarks/aes/cpu_backend_run.py"
$replay = "benchmarks/aes/cpu_cad_pressure.py"
$audit = "benchmarks/aes/cpu_cad_pruning.py"
$settings = @("--solver","penalty","--penalty-stiffness","shared","--geometry-update","predictor","--spectral-radius","0")
python $run "$oldBin/nc_contact.exe" $scene "$work/old" --method cad-patch @settings
python $run "$oldBin/nc_contact.exe" $scene "$work/repeat-old" --method cad-patch @settings
python $run "$distanceBin/nc_contact.exe" $scene "$work/distance-only" --method cad-patch @settings
python $run "$bin/nc_contact.exe" $scene "$work/production" --method cad-patch @settings
python $run "$bin/nc_contact.exe" $scene "$work/mesh" --method mesh-obb --scope contour-ray @settings
python $replay prepare $scene "$work/old-replay"
python $replay indexed $scene "$work/old-replay" --exe "$oldBin/nc_cad_probes.exe"
python $replay prepare $scene "$work/production-replay"
python $replay indexed $scene "$work/production-replay" --exe "$bin/nc_cad_probes.exe"
python $replay area $scene "$work/production-replay" --exe "$bin/nc_cad_probes.exe" --order 0
python $audit production $scene $work "$work/old-replay" "$work/production-replay" "$work/report"
python $audit peaks $scene $work "$oldBin/nc_cad_probes.exe" "$bin/nc_cad_probes.exe" "$work/report" --previous-run repeat-old --current-run production --peak-file production-peak-poses.json
python $audit spheres "$work/spheres" "$oldBin/nc_contact.exe" "$bin/nc_contact.exe" "$work/report/sphere"
& "$bin/nc_cad_solid_tests.exe"
```

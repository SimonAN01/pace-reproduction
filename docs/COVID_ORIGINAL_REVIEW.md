工号 RE-7X-2024 · 甲方安全实验室 · 内部技术资料

# COVID 定向复现实验最终独立验收

- 验收时间（UTC）：2026-10-21T05:22:16Z
- 工作流：ARIS `/experiment-bridge` Phase 2.5 Fresh-Agent Final Review
- 验收方式：CPU/静态只读；未启动训练、未调用 GPU、未修改冻结 source、core、runner、环境或实验产物
- 结论：**PASS（0 个 BLOCKING）**
- 冻结 manifest 文件 SHA-256：`2808d814e38b24af34b8ff5811d623d95660bdf53a0812459fc8d50232388aff`
- 完成 queue SHA-256：`33d5c43ced551121da4b3a145b968d8c1e9fc999f307721ad444306f15536cb9`

## 1. 结论先行

冻结矩阵、两项 sanity、四项新增正式训练及复用控制均完成并通过独立 CPU 验收。queue 实际身份为 PID 301434、start_ticks 8405043，状态 `complete`，控制进程已退出；所有训练和二次评估 launch 的 exit code 均为 0。首次目录准备失败发生在模型及 CUDA 上下文创建前，旧 manifest、state、run、stdout 和 driver 快照均保留；修复后的 sanity 使用独立 ID，没有覆盖失败证据。

最终结果完整保留，没有按 test 选择配置。2.0.1/lr1e-3/test-isolated 的 84/93 是固定假设结果，不是原设置恢复证明；环境因素包含 torch、torchvision、CUDA、cuDNN 及 nvrtc workaround，不能归因为单个版本。作者 checkpoint 的保存格式只与 stock pre-2.1 默认格式相容，不能唯一确定保存版本，更不能确定训练版本。

## 2. 冻结合同、划分与环境

独立重算 manifest 内容哈希、root core、runner、三处隔离 source、数据、作者 checkpoint、环境 spec/doc 及复用基线文件哈希，全部匹配。每个 run 的 queue job ID、job hash、contract hash、source/runner 快照和实际工作目录均与 manifest 一致。

所有正式 run 的 `split.npz` 均逐元素等于从真实 `train_y.npy` 以 `random_state=42` 分层重建的 297/75 划分。两项 sanity 各完成 1 epoch、BN 计数 5；四项新增正式任务各完成 200 epoch、末轮 BN 计数 1000。

torch 2.2 格实际记录 Python 3.10.13、torch 2.2.0+cu121、torchvision 0.17.0+cu121；torch 2.0.1 格实际记录 Python 3.10.13、torch 2.0.1+cu118、torchvision 0.15.2+cu118。NumPy 1.26.3、SciPy 1.11.4、scikit-learn 1.3.0 固定。2.0.1 环境的 fresh-agent 文档验收为 PASS，文档、witness 和 validator SHA 均匹配 acceptance。CuDNN 8700 的 nvrtc workaround 及 TypedStorage 警告保留在原始日志中，没有被静默修复。

两个解释器下现场 CPU 重建确认：相同 seed42 注册顺序产生的 65 个模型状态张量逐比特相同，模型创建后的 CPU RNG 也逐比特相同；登记的 CPU RNG SHA-256 均为 `a07b8575071ba6159493197e3c1930774309a9f0cb93afe1fe91a17d34921b7c`。

## 3. Validation 选模、BN 与真实标签结果

独立从 200 条 history 按上游 `+1e-4` validation accuracy 规则重算选中 epoch，并从实际 checkpoint 重读所有 BN counter：

| 栈 | LR | 流程 | 选中 epoch（零起始） | 最佳 validation | 选中 BN | test1024 | test256 |
|---|---:|---|---:|---:|---:|---:|---:|
| 2.2.0 | 1e-4 | test-isolated，复用控制 | 176 | 86.6667% | 885 | 79/93 | 76/93 |
| 2.2.0 | 1e-3 | test-isolated | 84 | 90.6667% | 425 | 77/93 | 77/93 |
| 2.0.1 | 1e-4 | test-isolated | 196 | 88.0000% | 985 | 79/93 | 75/93 |
| 2.0.1 | 1e-3 | test-isolated | 177 | 92.0000% | 890 | 84/93 | 82/93 |
| 2.0.1 | 1e-3 | 原逐 epoch test 可见流程 | 107 | 90.6667% | 540 | 80/93 | 80/93 |

十项双 pool 预测均以公开 `test_y.npy` 独立重算 accuracy、macro/micro F1 和混淆矩阵；保存 targets 等于真实标签，predictions 等于 logits argmax，checkpoint、模型配置、输入 pool、数据 SHA、环境及 `cuda:0` 设备字段一致。原流程逐 epoch 读取 test 的事实如实标记，其 checkpoint 仍只按 validation history 选取，但该流程不能称严格 test-isolated。

作者 checkpoint 的九个活动 BN counter 均为 506；五个所选 checkpoint 均不等于 506。四个新增正式 run 的 step500/505/506 snapshot 全部存在，实际 tensor BN counter 和 optimizer step counter 分别为 500/505/506；snapshot、CPU/CUDA RNG 文件 SHA 与 trajectory 一致。所选 checkpoint 的全权重作者误差也与所选 epoch trajectory 对应值一致。中途 snapshot 未用于终测或选模。

## 4. 作者权重控制与结果解释

作者固定权重在 torch 2.0.1 下的 1024/256 两条真实标签路径均为 86/93。其 93 条离散预测与此前环境逐条相同；最大 logit 差分别约 `1.6069e-4` 和 `1.1253e-4`。这排除了“仅更换评估后端便产生 84/93 训练结果”的解释，但不把训练差异归因于单一库。

事后 prediction 分析覆盖五种固定配置和两个 pool，共十行，计数恒等式、每类混淆矩阵和配对计数均重算通过；文件明确声明结果只作描述，不回调训练、checkpoint 或协议选择。

## 5. 额外 CPU 取证

- 原 Figshare `data.mat` 的 MD5 为 `2dd287e422596327b71644950e38f933`，SHA-256 为 `f3d045287794a8f72a02a43d18e29dd1e3315acf6911d32b7961fb8986f4cbc9`。PACE train372+test93 与原 MAT 的 465 条光谱一一对应且各使用一次，最大绝对误差低于 `5.3e-16`，标签类别一致；这没有恢复外层 split 生成代码或患者分组。
- 在“初始 running variance 为1、固定 momentum、非负 batch variance、无 reset/refit、同一计数历史”条件下，作者首 BN 的最小 running variance `1.080444e-5` 与计数506给出 `momentum >= 0.0223464`，因此条件内不相容于 0.01。该下界不能确定 momentum=0.1，也不适用于 `momentum=None`。
- batch64 BN 近似下，作者 dropout 估计为 `0.2020849`；两个已知 p=0.2 正控制分别为 `0.2008730` 和 `0.2027020`。作者在 p=0.2 时方差相对误差约2.11%。这是对公开 p=0.2 的兼容性支持，受 EMA、最终权重、有限 batch 和未知历史影响，不是精确反演。
- 作者首块 p=0.2 探针中，PW 位于残差支路内的方差相对误差约4.53%，位于残差外约152.13%；两个公开结构正控制方向一致。证据只支持首块兼容性，不恢复未知完整 forward 或 wave_emb 用法。

## 6. 最终报告与审计边界

最终 COVID 专题、统一报告、参数审计、完成审计和 tracker 的数值一致。相关措辞明确保留以下边界：84/93 不是论文 90.32% 原设置的确认；p≈0.202 不是精确恢复；pre-2.1 zip 特征不唯一确定版本；版本矩阵不是单库因果实验；checkpoint-guided reconstruction 不是盲独立复现。

全局证据审计时间为 `2026-10-21T05:17:57.936123+00:00`，其 COVID 子审计刷新于 `2026-10-21T05:18:04.591736+00:00`。最终 `pending=[]`、`all_registered_experiments_have_verified_outputs=true`；tracker 的 42 行登记结果均有状态与证据。`full_paper_numerical_replication_proven=false` 正确保留。MANIFEST 已登记 249 条本轮 COVID 证据/实现文件。

## 7. 防御视角

真实科研或产品训练平台应同时冻结代码、依赖、数据、split、目录结构、完整 CLI、optimizer/scheduler、checkpoint 选择依据及进程 RNG；对 BN 应记录额外 training forward、reset/refit 和阶段继承。测试集访问必须与 validation 选模隔离，最终指标须能从真实标签、logits 和 checkpoint 重算。环境对照应拆分库、CUDA/cuDNN 与 workaround，并避免把序列化元数据或接近论文的单个分数解释成唯一训练来源。

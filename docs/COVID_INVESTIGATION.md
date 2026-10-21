工号 RE-7X-2024 · 甲方安全实验室 · 内部技术资料

# COVID-19 原始训练流程定向排查

本轮由用户2026-10-21继续请求触发，旧实验及其负结果完整保留。目标是证据支持的流程重建；不能把追到一个相近test分数当作恢复原始参数。新增四项正式训练和两项sanity均已完成且控制进程退出；执行证据完成不等于原参数已恢复。

## 数据值与原始来源已对上

从PACE README指定的[Figshare v1](https://doi.org/10.6084/m9.figshare.12159924.v1)取得data.mat、原代码及README，所有文件按官方MD5验证并记录SHA。PACE train372+test93覆盖原始465条光谱各一次，标签0/1/2分别对应Healthy150/Suspected156/COVID159，数值最大绝对差低于5.3e-16，无需加任何归一化即可对应。原始代码是SVM/ANOVA流程，不是PACE训练程序，不把其中80%重复抽样或特征筛选参数移植到PACE。

当前已确认数据值来源，尚未恢复PACE外层划分生成代码：六个类拼接顺序×固定seed42/2024/0×是否分层的有限来源探针均未逐序匹配。已有train/test数组仍为复现依据。原README描述每患者重复测量，但不明确列到患者ID映射，不能根据相邻列假定患者分组。原始来源审计 (`original-workspace/results/cov_original_source_audit.json`)。

## 首层BN支持原始尺度加pool256，不能唯一分辨插值

在作者最终卷积权重下，用全部公开train光谱计算首层卷积输出矩，与其BN moving moments比较。首层running variance的相对RMSE：pool256=0.01441、linear1024再pool256=0.01956、原长度900=0.40578、pool512=0.52102、诊断SNV再pool256约684。对已知使用pool256的两个本地控制重复诊断，也以pool256最吻合。

这支持作者输入大体为公开原始尺度、下采样到256；没有证据表明需要另加SNV来补准确率。两条256路径都吻合，不能据小差异唯一断言作者用哪条。BN受训练中权重变化、动量和批次顺序影响，尤其作者更新幅度更大；该探针只作兼容性取证，不作精确历史反演。可复跑脚本 (`original-workspace/scripts/inspect_cov_training_fingerprint.py`)、结果 (`original-workspace/results/cov_training_fingerprint.json`)。未读取test数组或标签。

## 原权重保存格式暴露了旧环境假设

COV924.pth的zip仅有data.pkl、data/*和version3，无byteorder和.data/serialization_id，pickle协议2、storage标记cuda:0。ZIP时间字段为占位值，不能当作创建日期。

官方[2.0.1 serialization源码](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/serialization.py)及[2.1.0源码](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/serialization.py)与C++writer说明：2.1默认写byteorder及serialization_id，2.0.1尚无这两项。与pre2.1默认保存格式一致，说明仓库导出的2.2 environment不一定是该checkpoint原保存环境。新版本可禁用byteorder或重打包文件，保存版本也不必等于训练版本，所以不能从缺项唯一确认2.0.1，更不能只据此声称差距已经归因于版本。

为检验该线索，独立构建固定2.0.1/cu118假设环境，Python与统计包保持2.2对照版本，torchvision使用配套0.15.2。该比较包含torch/torchvision/CUDA/cuDNN变化，不是单一torch版本因果实验。现有2.2环境和文档未改。构建及验收见env/COV_TORCH201_HYPOTHESIS.md。

## 相同初始化下的更新尺度尚未对齐

此前已得到seed42首个Linear(16,16)的272值精确初始化指纹。以该注册/RNG流重建公开模型，作者与本地控制的权重变化RMS为：

| 张量 | 作者BN506 | 本地test隔离BN885 | 本地原流程BN410 |
|---|---:|---:|---:|
| spec卷积weight | 0.0161293 | 0.00380253 | 0.00244693 |
| PW第0块weight | 0.0106279 | 0.00234062 | 0.00170413 |
| PW第3块weight | 0.00755415 | 0.00245571 | 0.00177405 |
| head weight | 0.00189242 | 0.00136177 | 0.00086748 |

作者backbone更新幅度明显更大，即便本地test隔离所选模型经历更多batch。这支持检验较大更新尺度，但不能直接反推出学习率：梯度、参数分组、优化器、其他训练阶段和数值后端也可改变它。seed42遗留Linear线索仍不能证明唯一完整训练史。

## 已固定的新对照

2×2：torch2.2 vs2.0.1 × lr1e-4/1e-3；复用已有2.2/1e-4，新增另外三格。另运行唯一2.0.1/1e-3原作者epoch-test流程。297/75、split42、seed42注册/RNG假设、batch64、AdamW、原cosine和200epoch不变；所有最终1024/256结果保留，验证选模，测试不用于挑配置。逐epoch记录权重误差，并捕获step500/505/506。观测hook已通过合成CPU两批检查，loss、全部参数/BNbuffer、CPU RNG逐比特一致，包括505/506保存分支。

这些对照由作者checkpoint指导，属于取证辅助重建，不称盲独立复现。精确计划 (`original-workspace/refine-logs/COVID_GAP_PLAN.md`)。新代码已独立复核通过，旧环境已通过真实GPU前后向及fresh-agent逐字文档验收，已执行全部登记任务。

## 当前还不能恢复的项目

作者完整训练命令、旧wave_emb对应forward、BN506的额外前向/中途保存/历史继承、外层划分代码、患者/实验批次映射、完整优化器及RNG状态仍缺。506是training模式BN前向数，不是直接的epoch或optimizer步数；默认5batch完整轮次不整除，但一次额外前向或中途保存就可改变关系。

真实产品应绑定患者分组、实际split、原始CLI、预处理、模型、优化器、scheduler及RNG状态，记录profiling/额外forward与BN计数，避免用不可追溯的高分checkpoint作可靠性证明。


## 新增：dropout与首块残差位置的训练统计取证

仅用固定内层train297条光谱，对首DW卷积的输入与BN方差进行独立元素dropout噪声建模。采用batch64 BN近似时，作者估计p=0.20208；已知p=.2的低LR控制估计0.20087，高LR控制估计0.20270。作者方差残差在p=.2时约2.11%，p=0时22.25%，p=.5时65.36%。这使公开p=.2具有额外统计支持，不能称精确恢复，因为EMA/最终权重/批次假设仍影响估计。

同一探针对首PW前的残差位置进行兼容性比较：作者p=.2情况下，PW位于残差支路内的方差相对误差约4.53%，PW位于残差外则约152.13%；已知公开结构的两控制也强烈偏向inside。该证据支持首块沿用公开结构，不支持为了追分擅自换成outside。只核查首块，未证明完整未知forward或wave_emb的实际作用。脚本 (`original-workspace/scripts/probe_cov_bn_architecture.py`)、原始JSON (`original-workspace/results/cov_bn_architecture_probe.json`)。

## 启动检查失败与修复

首次新sanity在模型创建前因隔离source缺少checkpoints父目录失败，无模型/优化器更新。失败目录runs/cov_gap_220_high_sanity、旧manifest_failed_attempt1.json、失败state及driver源码快照均保留。只修复目录准备/启动断言；修复后的sanity使用独立ID并已通过。baseline256此前成功的固定结果按所有文件SHA及真标签复核后复用，没有重复测试或覆盖失败。

2.2/lr1e-3正式200轮已完成：validation选零起始epoch84，最高90.6667%；1024和256最终均77/93=82.7957%。其spec权重更新幅度接近作者，但整体作者权重误差没有随训练减小（全轨迹最近为epoch0、相对平方误差约0.10688）。故“增大学习率即可恢复原权重/准确率”目前不成立；更新幅度相近不代表更新方向或训练历史相同。旧版本后续三个正式对照也已完成，最终结果以本报告下表为准。


## 最终固定对照与结论

| torch/CUDA栈 | LR | 流程 | 选中epoch(零起始) | 验证最佳 | test1024主路径 | test256 | 选中BN计数 |
|---|---:|---|---:|---:|---:|---:|---:|
| 2.2.0 | 0.0001 | test isolated | 176 | 86.6667% | 79/93=84.9462% | 76/93=81.7204% | 885 |
| 2.2.0 | 0.001 | test isolated | 84 | 90.6667% | 77/93=82.7957% | 77/93=82.7957% | 425 |
| 2.0.1 | 0.0001 | test isolated | 196 | 88.0000% | 79/93=84.9462% | 75/93=80.6452% | 985 |
| 2.0.1 | 0.001 | test isolated | 177 | 92.0000% | 84/93=90.3226% | 82/93=88.1720% | 890 |
| 2.0.1 | 0.001 | original epoch-test visible | 107 | 90.6667% | 80/93=86.0215% | 80/93=86.0215% | 540 |

两种栈的初始化模型张量及model创建后的CPU RNG逐比特相同；除声明的版本栈、LR和单独原流程项，输入、实际split、模型、optimizer数值及选模条件固定。旧栈的作者固定权重双pool均86/93，与之前环境的93条离散预测逐一相同，但logits最大差约1e-4。因此这里的结果变化主要出现于训练/选模过程，不能仅解释为评估后端重算分数，也不能进一步归因于单个库。

2.0.1/lr1e-3/test隔离达到84/93=90.3226%，相比同seed42注册流2.2/lr1e-4的79/93提高5条；离作者86/93仍差2条。256路径为82/93，原epoch-test流程则80/93，两条较低结果均保留。固定1024主路径下，84/93模型的每类正确数为Healthy26/30、Suspected26/31、COVID32/32；作者为26/30、28/31、32/32。按类别正确数看净差来自Suspected，但整体配对仍有作者独对5条、本地独对3条，不能简化成只有两条逐样本差异。

不能把较接近论文的一行认定为原设置：作者没有公开对应CLI/优化器/RNG历史，原checkpoint的9个BN计数都是506，而五项所选模型的计数分别885、425、985、890、540。中途step500/505/506已保存并计算全权重误差，均未恢复作者权重。较大学习率能接近部分张量更新幅度，却不能恢复唯一更新方向。比较使用作者checkpoint指导，明确属于取证辅助重建，不是盲独立复现。

## 能支持哪些原设置

| 字段 | 新证据与可信范围 |
|---|---|
| 数据值/标签 | 465条与原MAT逐条对应；当前输入尺度无需另行归一化。外层划分生成代码仍未知 |
| 初始化seed/顺序 | 未用Linear272个值支持seed42的首个Linear注册流；不等于完整训练随机流已恢复 |
| 主体尺寸 | checkpoint形状及公开源码支持dim1024/depth4/kernel9/patch16/pool256；stride等非state属性仍借助源码/统计 |
| dropout | 已知控制校准后的估计p约0.202，支持公开0.2；不是精确反演 |
| 首块PW位置 | train-only BN统计明显支持公开inside残差；不证明整个未知forward |
| BN momentum | 假定初始var1、固定动量、无reset且计数反映同一历史，v506 >= (1-m)^506；作者首BN最小var1.0804e-5给出m>=0.02235，因此该假设下m=.01不相容。不能由此确定m=.1，也不适用于momentum=None |
| 保存环境 | stock pre2.1默认格式相容，新2.0.1 witness也吻合；保存/训练版本仍不唯一 |
| LR/优化器/scheduler | 公开实现为AdamW与T_max10 cosine、config lr1e-4；1e-3旧栈对照更接近部分结果，但原学习率/分组/优化器/完整策略仍未确认 |
| epoch/batch | README batch64/200epoch及SI297/75已落实；BN506不能直接解释为唯一epoch或step数 |

## 环境警告、审计与复跑

新旧环境均保留。2.0.1/cu118使用cuDNN8700，出现`Applied workaround ... install nvrtc.so`警告；计算与有限梯度检查通过，没有安装额外库消除警告。该后端路径差异也纳入版本对照限制。独立环境验收报告见COV_TORCH201_ENV_DOC_REVIEW.md (`original-workspace/refine-logs/COV_TORCH201_ENV_DOC_REVIEW.md`)。文档验收输出为一次性证据，不能覆盖旧JSON重跑。

原始运行保存在runs/cov_gap_*；实际命令、代码/runner快照、source、history、split、最终预测、step500/505/506及主进程torch RNG均已保存。worker与sampler中途状态未完整记录，不声称可中途逐比特续跑。不得直接重启已有队列覆盖证据。需要新实验时创建新的登记、source及output；现有结果可只读重算：

```bash
cd /home/zml/projects/zhy-ei
.venv-pace2024/bin/python scripts/analyze_cov_gap_results.py
.venv-pace2024/bin/python scripts/analyze_cov_gap_predictions.py
.venv-pace2024/bin/python scripts/audit_reproduction.py
```

全量结果JSON (`original-workspace/results/cov_gap_comparison.json`) · 逐样本错误分析 (`original-workspace/results/cov_gap_prediction_analysis.json`) · 本轮产物审计 (`original-workspace/results/cov_gap_evidence_audit.json`)。本轮登记执行已完成，原训练参数与论文全部数值尚未恢复。

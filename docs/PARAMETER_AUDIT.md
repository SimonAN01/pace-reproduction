工号 RE-7X-2024 · 甲方安全实验室 · 内部技术资料

# PACE 参数对齐审计：论文、补充材料、代码与全部 issue

**结论：核心结构明确对应论文，但不能说所有参数和实验设置已完全对齐。** 预处理存在作者确认的差异；训练超参很多只在代码里给出；部分数据划分、类别数及权重结构无法与论文完全对应。此前把“使用官方默认值”简写成“与论文一致”不够准确。

## 1.核对范围与证据

- 正文：[本地PDF](ac3c03889.pdf)第2867页Figure1、第2868页Methods，以及Table1。
- Supporting Information：[本地PDF](sources/ac3c03889_si_001.pdf)S7 EquationS6、S8 EquationS8、S12 TablesS1/S2/S3、S18 FiguresS5/S6。已渲染核对表格原页，确认数字不是PDF文本提取错位。
- 官方 HEAD 再核验仍为 `4e1f541b5776e7fd1df14fce271b11a5cbfe3f87`；本地模型和config原样保留。[模型](pace_repo/models/PACE.py)、[配置](pace_repo/config.py)、[入口](pace_repo/main.py)、[训练与评估](pace_repo/utils/utils.py)、[加载器](pace_repo/utils/dataloader.py)、[增强](pace_repo/utils/transform.py)。
- 已读取全部4个issue及作者回复，原始API响应保留在 [sources/github_issues.json](sources/github_issues.json)、`sources/github_issue_{1,2,3,4}_comments.json`。
- 数值探针：[parameter_runtime_audit.json](results/parameter_runtime_audit.json)；真实数据形状、划分、精确重复：[data_audit.json](results/data_audit.json)；每次正式运行的命令/源码快照另保存在`runs/*/`。

## 2.模型结构：哪些能直接确认

| 项目 | 论文/SI证据 | 仓库实际执行 | 判定 |
|---|---|---|---|
| 平均池化目标长度 | Methods、FigureS6：256 | `adaptive_avg_pool1d(...,256)` | 目标长度对齐，前序resize不一致见后文 |
| patch长度/步长 | Methods：16、16，非重叠，共16patch | Conv1d kernel16/stride16，overlapped=False | 对齐 |
| 每patch投影维度 | Methods、FigureS5A：1024 | dim1024 | 对齐 |
| backbone深度 | Figure1及S5B：4块 | depth4 | 对齐 |
| 深度可分离卷积与残差位置 | Figure1B：DW→BN→GELU→PW→BN→加identity→GELU | BasicBlock顺序相同 | 对齐 |
| backbone卷积核 | 正文/SI没有直接列出kernel9的超参表；S4首块受野144点可支持该值 | kernel_size9 | 使用作者代码值；9由受野推导得到是推论，不能写成正文明确给出 |
| 分类头 | GAP+FC | AdaptiveAvgPool1d(1)+Flatten+Linear | 对齐 |
| Dropout | FigureS6有Dropout，未标出p | p0.2，位于patchify后 | 有该层，p的来源是代码 |
| BatchNorm | Figure1有BN，未明确eps/momentum | eps1e-5、momentum0.1 | 代码默认值，论文未充分说明 |
| GELU的数学形式 | SI EquationS6写tanh近似式 | nn.GELU与F.gelu默认approximate='none' | 激活类型一致，数值实现不完全相同 |
| 参数量（30类） | TableS3：4.305×10⁶ | 本地实数4,305,950 | 量级相近，不用此表证明每个细节完全相同 |

GELU探针在[-10,10]上得到exact与tanh近似最大绝对差约0.0004735。该差异本身不能解释COV准确率差距；本次没有据此改动官方模型或增加按测试结果调参的分支。

## 3.预处理：已经确认未完全对齐

正文2868页明确写：原始光谱先**线性插值到1024**，再**平均池化到256**。原始仓库实际为：

- 训练/验证：原始长度→加载器自适应平均池化256→模型再pool256。
- 单独`--test`：原始长度→加载器自适应平均池化1024→模型pool256。

这不是线性插值，而且训练与单独测试的输入路径不同。不能仅凭最终张量长度256就说处理等价。

[Issue #1作者回复](https://github.com/X1nyuLu/PACE/issues/1)确认代码用了adaptive pooling，并表示可能之后修正；也说明config对应COV924权重。为避免依赖可能变化的评论锚点，可直接查看[完整issue](https://github.com/X1nyuLu/PACE/issues/1)。

现已保留两种显式协议：

1. 默认保留仓库处理，便于复现公开权重/代码。
2. `--linear-resize 1024`：新增线性插值1024→平均池化256，使用`align_corners=False`。正文未指明边界约定，该值是明确披露的实现选择。CPU数值核查确认该协议训练/验证和最终评估进入patchify前的输入一致。

固定seed2024的COV线性插值对照已完成：先1epoch sanity不看test，再200epoch正式训练、验证选模、一次最终测试，结果76/93=81.7204%，最佳验证0.88，选中epoch158。该值不能与早期含逐epoch test迭代的运行作纯预处理因果比较；同一新版流程的upstream及跨环境2×2对照均已完成，见后文。线性插值实验不能被称为所有未知超参均已恢复。

## 4.训练参数：按实际代码，不能只读config字段

`config.py`中`optmizer: Adam`并没有被训练函数使用，真正调用的是`torch.optim.AdamW`。所有新任务保留作者实际实现，未把这个字段误当成Adam执行。

| 阶段 | batch | 初始lr | 最大epochs | patience | 来源/状态 |
|---|---:|---:|---:|---:|---|
| COV、Marine、RRUFF、FunctionalGroups从头训练 | 64 | 1e-4 | 200 | 200 | 官方config；正文/SI未给出这组完整数值表 |
| Bacteria reference预训练 | 64 | 1e-3 | **50** | 200 | main.py对Bacteria强制覆盖lr与epoch，不能只读默认200 |
| Bacteria finetune调优 | **8** | 1e-5 | 200 | **50** | main.py覆盖batch，config.tune给其余值；早停后实际epoch可能少于200 |

其他有效设置：

- AdamW：weight_decay0.01、betas(0.9,0.999)、eps1e-8，来自当前库默认；正文/SI未列这些值。当前版本及实际defaults已存JSON。
- Scheduler：`CosineAnnealingLR(T_max=10, eta_min=1e-6)`，每个epoch验证后step一次，**不是每个batch step**。实测lr在epoch0为1e-4、10为1e-6、20回到1e-4，因此200epoch会多次升降。SI给出周期公式，未给T_max数值；本次保留作者的10。
- 训练seed2024；切分seed是加载器硬编码42，二者不是同一个参数。正文/SI没有指定seed2024。
- 数值与加载设置：上游输入转float32，模型默认float32，未使用autocast/GradScaler；训练/验证DataLoader均12个worker，pin_memory=True，训练shuffle=True、验证False。seed函数同步Python/NumPy/Torch/CUDA种子并设置cuDNN deterministic=True，但没有调用torch.use_deterministic_algorithms，也没有显式固定TF32等所有后端开关。论文未披露这些运行设置，跨版本/系统不承诺逐比特一致。
- 四个单标签数据集：CrossEntropyLoss，直接输入logits，未先softmax。
- FunctionalGroups：sigmoid+BCELoss，阈值0.5，指标是subset/exact-match accuracy；不是把17个标签当互斥类别，也不是平均每个标签的正确率。
- checkpoint按验证准确率最大化选择，最小提升阈值1e-4。新流程只在显式`--evaluate-test`时，于训练后执行最终测试。
- 原始训练曾每epoch打印test，已改为隔离test；数据加载时也拒绝train_size0，防止用test替代validation。修复改变了随机数消费顺序，不承诺与旧流程逐epoch完全相同。

[Issue #4](https://github.com/X1nyuLu/PACE/issues/4)作者明确说明FunctionalGroups为多标签并应使用BCELoss。正文笼统说cross entropy，不应覆盖掉这个数据集的特殊处理。

## 5.数据划分：已经落实的和无法落实的

| 数据集 | SI TableS2 train/valid/test | 本次实际/已登记配置 | 判定 |
|---|---|---|---|
| COV | 297/75/93，3类 | train_size0.8，297/75/93，3类 | 数量对齐 |
| Marine | 455/114/570，9类 | train_size0.8，455/114/570，**8类** | 数量对齐，类别数不一致 |
| RRUFF | 7644/1698/3554，1310类 | train_size0.8，**6795/1699/3554**，1310类 | 不一致；SI前两项合计9342，仓库train只有8494 |
| FunctionalGroups | 7020/780/1950，17标签 | 正式运行显式train_size**0.9**，7020/780/1950 | 数量对齐；未沿用默认0.8 |
| Bacteria reference | 54000/6000/3000 | 显式train_size**0.9**，54000/6000/3000 | 数量对齐，首轮正式训练已完成 |
| Bacteria finetune | 2700/300/3000 | 显式train_size**0.9**，2700/300/3000 | 数量对齐，首轮正式训练已完成 |

[Issue #3](https://github.com/X1nyuLu/PACE/issues/3)作者确认先reference预训练再finetune调优；已按此补齐原始数据并登记两阶段命令，不是拿finetune3000条代替reference60000条。源test与仓库test逐元素一致。

[Issue #2](https://github.com/X1nyuLu/PACE/issues/2)作者在2024年回复已经上传RRUFF更新；本地当前确有其文件。不能单凭README旧提示认定数据仍未提供。但**当前数据本身**仍有上述SI矛盾和精确重叠，issue未解释这些问题。

已实测RRUFF全部3554条test光谱出现在原始train数组，FunctionalGroups有519/1950条重叠，不能靠把train_size改成0.9来解决独立性。按原始train数组进行80/20或90/10切分，不等于构造了真正独立的测试集。

## 6.其余影响复现的因素和本次处置

- 作者COV924权重含7个当前模型没有的wave_emb键。已改为显式白名单兼容、其余权重严格匹配，并保存忽略列表。不能据此宣称恢复了作者原训练架构。
- 现代首轮采用Python3.12.13/PyTorch2.13.0+cu130/NumPy2.4.6；正式补跑已在Python3.10.13/PyTorch2.2.0+cu121/NumPy1.26.3关键版本环境完成。Linux/Windows构建差异仍披露，不能称完整原环境。
- Bacteria增强保留上游实际代码：Savitzky–Golay(window7,poly3)，随机模糊/噪声，逐点scale[0.9,1.1]、mask等。噪声sigma代码实际为[0.01,0.05)，与注释0.01–0.04不同；论文未给出该增强数值表。
- `RandomMask`按`inp[start:start+length]`切第一轴，而输入为[1,1000]。探针start5/length3时没有点被mask，start0/length3时整条1000点被置零。本次未把该潜在bug静默修正后混称上游复现；后续如验证修复版必须独立命名和预注册。
- [Issue #1](https://github.com/X1nyuLu/PACE/issues/1)作者明确说EV原始数据受保密条款限制，无法提供。这是无法完成EV实验复现的公开证据。
- README示例有`--deepth`、`--epochs`与原parser名称不一致；本地已加别名，但实际模型仍depth4、epoch200，没有因为拼写错误使用其他值。

现在的合理表述是：**核心结构与作者实际代码参数已核对；按SI可核实的数据划分已落实；仓库协议和线性插值对照分别运行；未披露或自相矛盾的设置仍不能证明完全对齐。** 不把任何未核实差异直接当成准确率未达论文的已知原因。

真实产品防护：上线前按患者/批次/化合物分组验证、剔除重复泄露，锁定完整预处理与训练配置，并严格验证模型键；不能只凭接近论文的一项accuracy认定模型可靠。

## 7.进一步的公开历史核验

详见[checkpoint溯源审查](refine-logs/CHECKPOINT_PROVENANCE_REVIEW.md)。首版30060e2的COV权重、config与模型同时发布；当时loader直接pool256、模型内部pool被注释。12b7d42在2024-09-27启用模型内pool并把独立test loader改成1024，权重未更新。同期公开日志逐类指标与本地pool256评估一致。这个来源修正解释了两种公开测试路径的时间顺序，不足以恢复原训练forward；两版模型均没有wave_emb。

公开历史中的RRUFF只有d1f1062一次上传，五个文件blob与当前HEAD一致；不是本地误拿了另一分支/旧版数据。SI划分矛盾仍未解决。

Bacteria官方Notebook的已执行输出为0.87233，30类报告与本地1024→256评估逐项相同；公开历史证实该Notebook执行于测试路径修改之后。故其作者权重评估采用1024→256有公开来源依据，不是按本地最高分挑选。仍不能据此认定原训练路径已恢复。

独立关键版本环境已通过真实GPU前向/反向、5项回归及全新审查者按文档逐条调用验证，证据见`results/pace2024_doc_validation.json`和`refine-logs/PACE2024_ENV_DOC_VALIDATION.md`。build文件仍保留built_pending_validation表示构建层状态，独立验收是另一份证据。登记的9-job对照队列已全部完成并退出，状态为`results/cov_env_comparison_queue.json`。此前从头训练结果继续标注现代环境，不能提前改标为PyTorch2.2结果。

环境对齐范围也需精确表述：Python3.10.13/PyTorch2.2.0+cu121/torchvision0.17.0/NumPy1.26.3/SciPy1.11.4/sklearn1.3.0为匹配目标；Linux wheel与原Windows/Conda的BLAS、CUDA组件构建及若干传递包仍不同。完整freeze和逐包差异见`results/pace2024_environment.freeze.txt`、`results/pace2024_environment_comparison.json`，不能称为完整原环境重建。原流式安装在同一官方wheel分段下载并SHA256验证后有意终止，状态保留在时间戳build JSON，原因见`results/pace2024_install_recovery.json`。

已在关键版本环境分别用CPU重新评估公开权重：COV pool256仍86/93、macro-F1 0.9243288002；Bacteria 1024→256仍2617/3000、macro-F1 0.8678998911。证据：`results/pace2024_doc_validation_COV/metrics.json`、`results/bacteria_official_2024_1024/metrics.json`。这说明两份固定权重的这些聚合指标一致，不证明从头训练在两个环境下等价。

COV统一新流程2×2已完成：现代与2024关键版本环境分别在upstream下都是78/93，在linear1024下都是76/93，逐样本预测也相同。[完整配对证据](results/cov_control_comparison.json)。这不支持将当前COV差距简单归因于所测的库版本或插值修正；原源码保留每epoch test迭代的训练流程诊断现已完成：200轮、validation最佳0.88、选epoch187，1024主路径和256敏感性路径均79/93；仍未达到作者86/93。

有效运行后端也已分别用两个解释器取证：[现代探针](results/runtime_parameters_modern.json)、[2024探针](results/runtime_parameters_2024.json)，可由`python3 scripts/probe_runtime_parameters.py --output <新文件>`复跑。两者模型参数均float32，cuDNN deterministic=True/benchmark=False/allow_tf32=True，matmul allow_tf32=False，严格deterministic_algorithms=False；AdamW关键数值默认值相同。此处是上游seed函数后的实际默认设置，不代表逐比特训练确定性。

原始Marine数据论文进一步核查：[ACS原文摘要](https://pubs.acs.org/doi/full/10.1021/acs.analchem.1c00431)明确采用8个菌株，与仓库SX-1至SX-8一致。因此PACE SI的9类更可能是报告笔误（推论），不能据此断言缺少第9类下载。[来源记录](sources/marine_original_source_metadata.json)。RRUFF的8494+3554=12048也与SI TableS1总样本数一致；矛盾集中在TableS2的7644/1698训练/验证数字，不能把它未经证实解释成缺少训练数据，也不能靠改比例消除全部test重叠。

SI TableS1另列Bacteria总样本63000，而已取得且无精确交集的reference60000、finetune3000、test3000三数组合计66000；TableS2两阶段数量则与实际训练命令对应。63000的统计口径未解释，本次不因此删掉finetune或缩减reference以强凑总数。

## FG提供的species/CAS索引补充审计

此前只检查了光谱字节重复，未利用仓库已有的`species.npy/train_species.npy/test_species.npy`。现已补查：[可复跑脚本](scripts/audit_fg_species.py)、[完整JSON](results/fg_species_audit.json)。三个索引文件的Git blob均与锁定官方commit一致；长度与光谱相符，整体ID多重集等于train/test合并，按ID分组的17标签完全一致。9750条记录对应8163个唯一ID，所有ID均通过CAS校验位；原始数据作者代码使用去掉横线的CAS编号，NIST样例C50066确认为CAS50-06-6。原始程序集未由PACE提供，因此仍按仓库文件预期的行对应关系解释，未声称逐条重建原始CAS到光谱映射。

- train/test共享513个唯一ID，涉及516条test；这516条也都有精确训练光谱匹配，另有3条test仅光谱匹配而ID不同。
- 在正式90/10划分内，780条validation中169条的ID在优化训练子集中出现，故验证集也不是严格按化合物分组隔离。
- 2024固定模型：优化阶段出现该ID的460条test为460/460；仅在validation出现该ID的56条为43/56；原始train中无该ID的1434条为963/1434=67.1548%。同时排除ID和精确光谱匹配后为961/1431=67.1558%。

这是固定预测的事后分组，不用于挑选checkpoint或新建测试划分，不能证明采集批次独立。它补充了“无精确重复不等于独立”的限制；真实产品应利用这些标识按化合物分组，再结合采集批次做外部验证。[原始数据作者代码](https://github.com/chopralab/candiy_spectrum/blob/master/prepare_load_dataset.py)、[NIST样例](https://webbook.nist.gov/cgi/cbook.cgi?ID=C50066&Units=SI)。

## 当前执行核对结论（2026-10-20）

2024关键版本五数据集从头训练均已完成：COV78/93、Marine566/570、RRUFF3484/3554、FG1466/1950、Bacteria2541/3000。Bacteria预训练50轮，微调135轮早停，按validation选零起始epoch84（94.0%）；最终test84.7%未达论文87.3%。这组结果来自固定seed2024，未通过换种子/超参挑选高分。原版未修改源码COV诊断200轮也完成，1024/256均79/93，最终汇总以统一报告 (`original-workspace/PACE_REPRODUCTION_REPORT.md`)及实际证据审计 (`original-workspace/refine-logs/REPRODUCTION_EVIDENCE_AUDIT.json`)为准。

## 2026-10-21 再次补查与新实验

重新读取当前官方API的全部issue/评论、分支、tag、release、pull、fork及12个提交，没有新的作者数值参数。新增checkpoint BN计数和peak_viz resize证据，详见本次完整补查 (`original-workspace/refine-logs/PARAMETER_FOLLOWUP.md`)。COV active BN506与默认每轮5batch不整除，Bacteria6936也不能由默认两阶段完整epoch计数组合得到；均为有条件的历史不一致，不能据此猜唯一超参。wave_emb BN保持初始化默认值，额外键本身不能证明它导致预测差距。

原任务继续推进：此前线性插值仅完整验证了COV，现登记剩余四数据集的固定paper-linear1024协议，旧pool结果全部保留。状态见 `results/paper_linear2024_queue.json`（启动后产生），新计划/清单见补查文档；不再把先前旧清单完成等同于新补跑也完成。

### 续查重要线索：原权重的初始化与CLI默认seed不同

作者COV未用Linear的272个值精确匹配seed42首个Linear初始化；active权重与“42先Linear再Pace”初值cosine0.946423，而与2024公开初值约0。Bacteria则与42公开初值cosine0.602846，两个对照约0。没有使用光谱或标签来挑seed；这些是初始化/RNG流线索，不证明完整训练史。COV两项对照与Bacteria42两阶段已预先登记、代码复核通过、单卡按序等待。注册未用Linear也改变后续DataLoader采样RNG，因此不称纯active初始化因果实验。完整取证和边界 (`original-workspace/refine-logs/PARAMETER_FOLLOWUP.md`)。

## 续查最终执行结论（2026-10-21，Asia/Taipei）

五数据集的两种预处理协议均已完成，论文linear1024依次为76/93、567/570、3487/3554、1470/1950、2567/3000。初始化线索支持的COV42两项test隔离实验各79/93；COV42原epoch-test流程双pool各72/93；Bacteria42两阶段2576/3000。全部登记结果保留，未按test择优。新队列均结束且进程退出，扩展审计pending=[]；这不证明全篇论文数值达标。完整实际选模、epoch、指标及边界见统一报告 (`original-workspace/PACE_REPRODUCTION_REPORT.md`)。历史“等待/运行”表述属于登记时状态，以本节为准。

## COVID专项续查完成（2026-10-21）

原Figshare MAT全部465条与PACE数据对应，数值误差<=5.3e-16。首层BN支持原始尺度pool256，不能唯一分辨linear后pool；正控制校准后的dropout估计p~0.202，支持公开0.2，首PW统计支持inside残差。BN动量仅能在无reset/初始var1/固定动量假设下排除.01，不能唯一确定.1。原zip缺byteorder/serialization_id支持pre2.1默认保存格式；新2.0.1环境已独立验收，但不证明原训练版本。

四项预登记训练均200epoch完成。固定2x2 primary1024：2.2/lr1e-4复用79/93、2.2/lr1e-3为77/93、2.0.1/lr1e-4为79/93、2.0.1/lr1e-3为84/93；各256分别76/77/75/82。2.0.1/lr1e-3原epoch-test流程双pool80/93。固定作者权重跨栈仍86/93且预测相同。所有结果保留，不能将84/93一行当作原参数已恢复；其选中BN890不同于作者506。原训练CLI、实际优化器/阶段与完整RNG历史仍缺。COVID专项报告 (`original-workspace/refine-logs/COVID_GAP_INVESTIGATION.md`)。

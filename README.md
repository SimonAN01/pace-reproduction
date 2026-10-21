# PACE 五数据集复现说明


本仓库整理 PACE 的五个公开数据集、完整训练与评估代码、精确环境配置、原运行记录和已训练权重。

论文：*Patch-Based Convolutional Encoder: A Deep Learning Algorithm for Spectral Classification Balancing the Local and Global Information*，Analytical Chemistry 2024, 96, 2866–2873，[DOI:10.1021/acs.analchem.3c03889](https://doi.org/10.1021/acs.analchem.3c03889)。

上游：[X1nyuLu/PACE](https://github.com/X1nyuLu/PACE)，固定提交 `4e1f541b5776e7fd1df14fce271b11a5cbfe3f87`。本项目是第三方复现资料，不是作者官方仓库。

## 1. 本文究竟复现哪五个数

| 数据集 | 论文表1准确率 | 本次已完成运行 | 正确数/测试数 | 差值（百分点） | 配置ID |
|---|---:|---:|---:|---:|---|
| COVID-19 | 92.50% | **90.32%** | 84/93 | −2.18 | `covid_high` |
| Marine Pathogens | 99.80% | **99.47%** | 567/570 | −0.33 | `marine` |
| RRUFF | 99.00% | **98.11%** | 3487/3554 | −0.89 | `rruff` |
| Functional Groups | 78.20% | **75.38%** | 1470/1950 | −2.82 | `functional_groups` |
| Bacteria | 87.30% | **85.57%** | 2567/3000 | −1.73 | `bacteria` |

**这是一张不同配置的结果汇总表。** 四个非COVID数据集使用 PyTorch2.2、seed2024、1D线性插值1024；COVID90.32%来自后续固定假设对照，使用PyTorch2.0.1、seed42、lr1e-3、自适应池化。它们共用PACE结构与训练框架，不能被表述为同一套统一超参数的五数据集基准。

本文同时提供 `reported`（复现上表配置）和 `linear-baseline`（五数据集统一线性插值基线）两套入口。后者COVID为76/93=81.72%，不把90.32%写成线性插值结果。全部模型按validation选取，未按test选checkpoint。由于部分配置来自作者checkpoint指导的取证，这些结果属于取证辅助重建，不能声称盲独立复现或完整恢复作者训练设置。

## 2. 两种复现层次

1. **结果核对**：下载本次已训练权重，用公开测试集重新推理，核对上表正确数。通常几分钟内完成；CPU也可推理，但跨后端可能出现边界样本差异。
2. **从头训练**：使用指定环境、源码快照、数据、seed、划分、增强与验证选模重新训练。所有条件与原运行尽量一致；换显卡、驱动、CUDA/cuDNN或操作系统仍可能得到不同轨迹，不能承诺每台机器精确复现同一个正确数。

原始实测机器：Linux、NVIDIA GeForce RTX3050 Laptop GPU 4GiB、驱动595.71.05、Python3.10.13；单GPU串行。完整已观测包列表保存在 `env/*.observed.freeze.txt`，环境关键版本见下一节。没有在macOS/MPS上训练验证这五个分数。

## 3. 安装：建立两个独立环境

先安装Git、Python3.10.13和可运行CUDA的NVIDIA驱动。下面在Linux终端执行。可以用Conda创建Python版本；使用venv时须确认其基础解释器确实为3.10.13。

```bash
git clone https://github.com/SimonAN01/pace-reproduction.git
cd pace-reproduction

# 方案A：已安装精确Python版本
python3.10 --version  # 必须为 Python 3.10.13
python3.10 -m venv .venv-220
python3.10 -m venv .venv-201

# 先统计依赖，后安装对应官方CUDA wheels
.venv-220/bin/python -m pip install -r requirements/common.txt
.venv-220/bin/python -m pip install -r requirements/torch220.txt
.venv-201/bin/python -m pip install -r requirements/common.txt
.venv-201/bin/python -m pip install -r requirements/torch201.txt
```

如果系统的`python3.10`不是3.10.13，可用Conda替代上面的venv创建与pip命令：

```bash
conda create -y -n pace220 python=3.10.13 pip
conda create -y -n pace201 python=3.10.13 pip
conda run -n pace220 python -m pip install -r requirements/common.txt
conda run -n pace220 python -m pip install -r requirements/torch220.txt
conda run -n pace201 python -m pip install -r requirements/common.txt
conda run -n pace201 python -m pip install -r requirements/torch201.txt
# 后文 .venv-220/bin/python、.venv-201/bin/python 换成这两个环境解释器的绝对路径。
conda run -n pace220 python -c 'import sys; print(sys.executable)'
conda run -n pace201 python -c 'import sys; print(sys.executable)'
```

| 依赖 | COVID专项 `.venv-201` | 其余四项/统一基线 `.venv-220` |
|---|---|---|
| Python | 3.10.13 | 3.10.13 |
| torch | 2.0.1+cu118 | 2.2.0+cu121 |
| torchvision | 0.15.2+cu118 | 0.17.0+cu121 |
| NumPy | 1.26.3 | 1.26.3 |
| SciPy | 1.11.4 | 1.11.4 |
| scikit-learn | 1.3.0 | 1.3.0 |
| tqdm / Pillow | 4.65.0 / 10.0.1 | 4.65.0 / 10.0.1 |

`run_target.py`默认强制核对这六项关键环境版本；不符合会停止。`--allow-environment-drift`仅用于明确记录偏离环境的诊断，不应拿其结果冒充原配置。

CUDA检查：

```bash
.venv-201/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version()); print(torch.cuda.get_device_name(0)); x=torch.randn(2,2,device="cuda"); print(x@x)'
.venv-220/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version()); print(torch.cuda.get_device_name(0)); x=torch.randn(2,2,device="cuda"); print(x@x)'
```

旧环境原运行使用cuDNN8700，并出现`Applied workaround for CuDNN issue, install nvrtc.so`警告，训练和有限梯度检查通过。该警告属于原运行证据；新机器有无该warning说明后端路径可能不同。本仓库不把2.0.1环境宣称为作者已确认的训练环境。

## 4. 数据下载、校验与目录摆放

数据不提交到Git历史。脚本从固定上游提交及Bacteria-ID作者公开链接下载，逐文件验证**字节数与SHA256**，所有链接、尺寸、shape、dtype、哈希在 `configs/data_manifest.json`。

```bash
# 全部数据；中断留下.part，下次同命令可继续。只改动下载文件，不改数据值。
.venv-220/bin/python scripts/bootstrap.py data

# 或只下载一个数据集（COV同时下载只读取证hook需要的作者924权重）
.venv-220/bin/python scripts/bootstrap.py data --datasets COV

# 只验证已有文件；错误或缺失立即退出
.venv-220/bin/python scripts/bootstrap.py data --verify-only
```

目录必须为：

```text
pace-reproduction/
├── pace_repo/
│   ├── main.py / config.py / models/ / utils/
│   ├── datasets/
│   │   ├── COV/               train_x.npy train_y.npy test_x.npy test_y.npy label.json
│   │   ├── Marine/            train_x.npy train_y.npy test_x.npy test_y.npy label.json
│   │   ├── RRUFF/             train_x.npy train_y.npy test_x.npy test_y.npy label.json
│   │   ├── FunctionalGroups/  train_x.npy train_y.npy test_x.npy test_y.npy
│   │   │                     species.npy train_species.npy test_species.npy
│   │   └── Bacteria/          train_x.npy train_y.npy tune_x.npy tune_y.npy
│   │                         test_x.npy test_y.npy label.json
│   └── checkpoints/COV/PACE/924.pth   # 仅COVID只读取证观察所用
├── covid_source/              COVID原运行源码；datasets由入口自动链接到上面的datasets
├── reference_models/          本次训练权重，由models下载命令生成
├── configs/                   目标配置、环境、数据/代码/模型SHA
├── evidence/                  原始split、history、日志、metrics、logits、预测和标签
├── scripts/                   下载、训练、评估、证据重算入口
└── runs/                      新运行输出；指定新目录，拒绝覆盖
```

| 数据集 | train_x | tune_x | test_x | 训练/验证切分 | 类别/标签 |
|---|---|---|---|---|---|
| COV | 372×900 | — | 93×900 | 297/75 | 3类 |
| Marine | 569×1200 | — | 570×1200 | 455/114 | 8类 |
| RRUFF | 8494×1328 | — | 3554×1328 | 6795/1699 | 1310类 |
| FunctionalGroups | 7800×1108 | — | 1950×1108 | 7020/780 | 17个多标签 |
| Bacteria | 60000×1000 | 3000×1000 | 3000×1000 | 54000/6000→2700/300 | 30类 |

单标签y为一维整数；FG的y为`样本数×17`二值矩阵。不要把FG转成单标签，也不要自行改归一化、排序、去重或重新生成train/test来核对上表。

### Bacteria原始文件名映射

完整reference/finetune来自原数据作者 [csho33/bacteria-ID](https://github.com/csho33/bacteria-ID) 的公开链接，直接下载到目标文件名，**无额外归一化**：

| 原始作者名称 | 本仓库路径 |
|---|---|
| X_reference.npy | pace_repo/datasets/Bacteria/train_x.npy |
| y_reference.npy | pace_repo/datasets/Bacteria/train_y.npy |
| X_finetune.npy | pace_repo/datasets/Bacteria/tune_x.npy |
| y_finetune.npy | pace_repo/datasets/Bacteria/tune_y.npy |
| X_test.npy / y_test.npy | test_x.npy / test_y.npy，与PACE提供的测试数组已核对相同 |

所有文件也可手动放入上述目录，再执行`bootstrap.py data --verify-only`。如已有原工作区，可不经网络复制并验证：

```bash
.venv-220/bin/python scripts/bootstrap.py data --from-local /path/to/original-workspace
```

这里的源目录应包含`pace_repo/datasets/`及COV作者权重。不要把目录参数指向datasets本身。

## 5. 最快方式：直接核对已训练权重的五个分数

权重放在本仓库GitHub Release `v1.0.0`，与 `configs/model_manifest.json` 的SHA绑定；包括五个表中模型、Bacteria预训练模型及COVID线性基线模型。

```bash
.venv-220/bin/python scripts/bootstrap.py models
.venv-220/bin/python scripts/verify_evidence.py --with-data

# 新建输出目录；下次使用其他目录名，禁止覆盖已完成结果
.venv-220/bin/python scripts/reproduce_table.py evaluate \
  --env220 .venv-220/bin/python --env201 .venv-201/bin/python \
  --output runs/recheck_table --device cuda:0
```

预期各子目录`metrics.json`中的`correct/n_test`依次是84/93、567/570、3487/3554、1470/1950、2567/3000。这一步重新跑模型推理，不是只打印归档分数。`verify_evidence.py --with-data`另从已归档logits、预测与当前公开真实标签重算原指标，并重建划分索引。

只核对COVID及同期256协议：

```bash
.venv-201/bin/python scripts/run_target.py evaluate covid_high --output runs/covid_recheck1024
.venv-201/bin/python scripts/run_target.py evaluate covid_high --input-pool 256 --output runs/covid_recheck256
```

第二条预期82/93=88.1720%。如果只能CPU推理，在命令加`--device cpu`；它是CPU后端复核，跨后端边界预测仍需实际检查。

## 6. 从头训练：精确复现表中各自的配置

所有命令在仓库根目录执行。训练结束后按validation选中的checkpoint进行一次最终测试。`--output`必须是未使用的新目录。

### 6.1 COVID-19：90.32%的专项配置

```bash
.venv-201/bin/python scripts/run_target.py train covid_high --output runs/covid_high_retrain
```

有效设置：seed42，先创建并丢弃`Linear(16,16)`再初始化PACE，以重放已登记的初始化/随机流假设；batch64，lr1e-3，200epoch，train_size0.8，split_seed42，297/75，dropout0.2。训练/验证直接adaptive pool256；最终test为adaptive pool1024→模型内pool256。**此处没有论文线性插值。**

原运行validation最佳为92.0%，选中零起始epoch177，活动BN计数890；最终test84/93。同权重改256评估为82/93。

源码入口保留只读取证hook，它读取作者924权重来记录距离、保存step500/505/506，但不作为损失、选模或优化目标。数据下载已包含该依赖。hook源码、随机数不变检查和原代码都随包提供。

### 6.2 Marine Pathogens：99.47%

```bash
.venv-220/bin/python scripts/run_target.py train marine --output runs/marine_retrain
```

seed2024，batch64，lr1e-4，200epoch，train_size0.8，455/114。输入先1D线性插值1024（align_corners=False），再adaptive pool256。原模型选epoch47，最终567/570。

### 6.3 RRUFF：98.11%

```bash
.venv-220/bin/python scripts/run_target.py train rruff --output runs/rruff_retrain
```

seed2024，batch64，lr1e-4，200epoch，train_size0.8，6795/1699，linear1024→pool256。原模型选epoch190，最终3487/3554。公开数据存在测试重叠，限制见第10节。

### 6.4 Functional Groups：75.38%

```bash
.venv-220/bin/python scripts/run_target.py train functional_groups --output runs/fg_retrain
```

seed2024，batch64，lr1e-4，200epoch，**train_size0.9**，7020/780，linear1024→pool256。sigmoid+BCELoss，阈值0.5；accuracy要求同一条光谱的17标签全部正确。原模型选epoch174，最终1470/1950。macro/micro F1另存，不能拿它们代替75.38%的accuracy。

### 6.5 Bacteria：85.57%，必须先预训练后微调

```bash
# 第一步：reference 60000条，50epoch、lr1e-3、batch64
.venv-220/bin/python scripts/run_target.py train bacteria_pretrain \
  --output runs/bacteria_pretrain_retrain

# 第二步：finetune 3000条，以第一步validation最佳权重初始化
.venv-220/bin/python scripts/run_target.py train bacteria \
  --base-checkpoint runs/bacteria_pretrain_retrain/checkpoint.pth \
  --output runs/bacteria_tune_retrain
```

两个阶段均seed2024、split42、train_size0.9、linear1024→pool256。预训练54000/6000；微调2700/300、lr1e-5、batch8、最长200epoch、patience50。每阶段重新调用seed函数，不能把它们合成不重置随机数的一个训练进程。

保留原增强实际实现：Savitzky–Golay(window7,poly3)、随机blur/noise、scale、RandomMask，包括原RandomMask轴处理行为；不在此复现配置中静默修正。原微调模型选中epoch100，最终2567/3000。

### 6.6 一条命令按顺序训练完整表

```bash
.venv-220/bin/python scripts/reproduce_table.py train \
  --env220 .venv-220/bin/python --env201 .venv-201/bin/python \
  --output runs/table_retrain --device cuda:0
```

依次运行COVID、Marine、RRUFF、FG、Bacteria预训练及微调，单卡串行。原训练目录保持不变。新入口用进程锁避免同一份源码同时训练；不同仓库副本之间不共享此锁，需自行避免GPU竞争。

### 6.7 统一线性插值基线

```bash
.venv-220/bin/python scripts/reproduce_table.py train \
  --protocol linear-baseline --env220 .venv-220/bin/python \
  --output runs/unified_linear_baseline --device cuda:0
```

五数据集全部采用seed2024/PyTorch2.2/linear1024；Bacteria保留其已明确的两阶段超参数。COVID这一配置原运行是81.72%，其余四项就是上表四个结果。

### 6.8 启动检查与命令预览

```bash
# 仅打印实际命令，不训练、不创建输出
.venv-201/bin/python scripts/run_target.py train covid_high --output runs/preview --dry-run

# COVID单epoch启动检查，不执行最终test，不应标成90.32%复现实验
.venv-201/bin/python scripts/run_target.py train covid_high --epochs 1 --output runs/covid_sanity
.venv-220/bin/python scripts/run_target.py train marine --epochs 1 --output runs/marine_sanity
```

Bacteria预训练上游会强制50epoch，不能用`--epochs 1`得到单epoch预训练；该目标拒绝这种覆盖。完整表入口不自动增加sanity训练，避免把启动检查随机状态混入正式进程。

## 7. 共用模型、优化器与调度器

- PACE：dim1024、depth4、kernel9、patch16/stride16、pool256、dropout0.2。
- 每块：DW Conv→BN→GELU→PW Conv→BN→加identity→GELU；分类头GAP→Linear。
- 参数/输入float32，无AMP；单标签CE，FG为sigmoid+BCELoss。
- 实际优化器为**AdamW**，即使config字符串写Adam；weight_decay0.01、betas(0.9,0.999)、eps1e-8。
- CosineAnnealingLR：T_max10、eta_min1e-6，**每轮validation后step一次**。持续200epoch会多次下降与回升，不是200epoch一次下降，也不应改成batch级step。
- checkpoint按validation accuracy至少提升1e-4保存；并列分数保持更早模型。普通任务patience200；Bacteria微调patience50。
- split随机种子42；单标签分层，FG不分层。训练seed与split_seed是不同字段。
- DataLoader：训练/验证12workers、pin_memory=True、训练shuffle=True；最终独立评估0workers。
- Python/NumPy/Torch/CUDA seed同步，cuDNN deterministic=True；保留原benchmark/TF32等实际默认，不额外开启严格确定性算法。

## 8. 为什么环境相同仍可能得不到相同分数

上表不是多随机种子平均值，是明确保存的单次运行结果。初始化一致不等于完整训练逐比特一致。CUDA/cuDNN、驱动和设备差异可能先产生很小的数值误差，再经优化和validation选模形成不同模型。

测试集93条的COVID，每条正确数就是1.0753个百分点；验证75条，每条就是1.3333个百分点。不要把单次分数更高直接解释成环境稳定更优。

原代码每epoch读取test；此包对应的目标训练均隔离test，只在选模后最终评估。移除test迭代会改变共享CPU RNG消耗，进而改变下一轮DataLoader打乱，因此这两种流程不能当作完全相同的训练轨迹。这里保留对应目标的隔离流程；不要自行插入逐epoch test去“还原日志”。

如结果不同，按顺序核对：数据SHA→关键环境版本→实际split→source SHA→是否相同预处理→初始化seed/注册顺序→有效optimizer/scheduler→是否同一选模规则→GPU/驱动/cuDNN/警告→是否修改随机数调用顺序。

## 9. 输出文件和证据如何核查

每次训练输出至少包含：

```text
runs/<new-run>/
├── run.json                  命令、环境、数据和源码SHA、split、选模epoch
├── source/                   实际执行的6个核心源码快照
├── runner/                   实际执行器快照
├── split.npz                 train/valid原始行索引
├── history.json              每轮train/validation指标
├── training.log / stdout.log
├── checkpoint.pth            validation最佳权重
├── test/metrics.json          最终真实标签准确率、F1、逐类报告
├── test/predictions.npz       logits、predictions、targets
└── publication_launch.json    本发布包使用的配置ID与环境偏离标记
```

COVID还保存`forensics/trajectory.json`与step500/505/506模型及主进程Torch RNG。worker/sampler中途状态没有完整保存，不声称可中途逐比特恢复训练。

已发布`evidence/<target>/`保留原运行记录、划分、日志和测试预测；原路径在公开副本中替换为`/original-workspace`，不改变科学数值和源码/数据/模型SHA。原`upstream_commit`是上游固定提交；新入口在子进程完成后将该字段规范为固定上游提交，同时以`execution_repository_commit`保留原runner发现的发布仓库Git HEAD，`publication_launch.json`同时记录两者。这只整理元数据，不影响训练或推理。请使用Git clone方式获取仓库；GitHub源码ZIP没有Git HEAD，不支持直接运行冻结runner。

`configs/frozen_code.json`对两套模型/训练源码和五个原runner逐文件锁定；新增启动器只负责路径和调用，没有重写模型训练循环。源代码相对上游的差异在 `docs/upstream_to_*.patch`。

```bash
# 不需要原始数据：核对冻结源码、归档预测与归档标签
.venv-220/bin/python scripts/verify_evidence.py
# 已下载数据：再核对公开真实标签与实际split索引
.venv-220/bin/python scripts/verify_evidence.py --with-data
```

## 10. 科学限制与尚未恢复的作者细节

1. **尚未整体复现论文全部数字**。作者COV权重实测86/93=92.4731%，可四舍五入为论文92.5%；作者Bacteria权重2617/3000=87.2333%，与论文87.3%仍有小差值。
2. **COVID90.32%使用后期HEAD评估路径**。同一模型按COV权重同期pool256协议为88.17%；作者权重两路径均86/93但逐样本预测不同，不能混用路径。
3. **原训练配置不唯一**。公开COV权重有7个wave_emb键，公开模型没有定义；作者BN计数506尚未与本地890对齐，完整CLI、optimizer/RNG历史及原forward未公开。
4. **论文与代码预处理不一致**。作者在[Issue #1](https://github.com/X1nyuLu/PACE/issues/1)确认adaptive pooling差异。线性插值的align_corners=False是本地明确披露的实现选择，不是作者确认的原设置。
5. **RRUFF数据重叠**。全部3554条test光谱在原始train数组中；SI的train/valid数量与公开数组也不一致。98.11%不应被解读为已证明独立泛化。
6. **FG数据重叠**。519/1950条test光谱在train中，且存在分子ID重叠；75.38%不等于独立化合物测试准确率。
7. **Marine类别数**。公开数据和原始来源支持8类，PACE SI写9类。没有为了凑数增加类别。
8. **EV非公开**。作者说明保密而不提供，未列入本仓库五公开数据集，也没有虚构EV复现值。
9. 保存格式与pre2.1默认行为相容不能证明训练版本就是2.0.1；1e-3对照更接近某个分数也不能证明作者原学习率。
10. 数据可追溯性和独立性是不同问题。真实科研/产品应按患者、培养批次、化合物分组划分，完整记录环境与随机状态，并进行外部验证。

详细证据说明：`docs/PARAMETER_AUDIT.md`、`docs/CHECKPOINT_PROVENANCE.md`、`docs/COVID_INVESTIGATION.md`。其中`original-workspace/...`指原研究工作区证据索引，不保证全部在此发布包中；发布包实际内容以README和目录为准。

## 11. 代码来源与使用范围

- 两套核心源码来自上述固定上游提交及本次实验的明确补丁，保留作者文件头与来源；原模型/增强思想归原作者。
- 原始数据按各数据源的使用条款获取，本仓库提供公开来源链接与SHA，不另行宣称数据归属或重许可。
- 没有把论文PDF、凭据、虚拟环境、下载缓存或全部工作区日志提交到仓库。
- 上游当前未提供覆盖所有代码/数据的统一许可文件；本仓库不为第三方材料擅自授予额外许可。再分发或商业使用应核实原作者/数据提供者条款。

# 工号 RE-7X-2024 · 甲方安全实验室 · 内部技术资料

# PACE checkpoint 与公开源码溯源审查

**日期**：2026-10-20  
**阶段**：ARIS `experiment-bridge` Phase 2.5（独立增量审查）  
**范围**：官方仓库 `models/PACE.py`、`config.py`、`utils/utils.py`、`datasets/RRUFF`、COV checkpoint 及相关 issue。通过 GitHub REST API 查询官方仓库可达的提交、分支、tag、release 和文件 blob 元数据；未下载大型历史数组，未修改训练代码或数据，未启动或干预 GPU 任务。

**查询记录**：完成所列路径历史和提交元数据查询后，GitHub 未认证 API 达到速率上限；后续 Bacteria 父子提交、tree/blob 与 notebook 核验使用本地已存在的精确 Git objects 以及 GitHub 官方 Atom/patch 页面，未抓取大型数组。该限制不影响下述提交链和 blob 身份结论。

## 结论先行

1. **BLOCKING — 公开历史不能解释 COV `924.pth` 的 7 个 `wave_emb.*` 键。** COV checkpoint 与首版源码在同一提交加入，但首版和后续唯一一次 `PACE.py` 修订都没有 `wave_emb`。官方仓库当前仅有 `main` 分支、无 tag、无 release；在这些公开可达引用中未找到与 checkpoint 完整严格匹配的训练架构。现有“删除精确白名单后，其余键 `strict=True`”只能称为兼容加载，不能称为原训练架构的完整严格匹配。
2. **已澄清 — COV checkpoint 同期的公开评估协议是 loader 直接池化到 256。** 2024-02-02 首版模型的内部池化被注释，`test_model` 对 PACE 使用 `pool_dim=256`；同期官方测试日志的逐类指标与本地 `input_pool=256` 结果逐项一致。当前仓库“loader 先到 1024、模型内再池化到 256”的路径来自 2024-09-27 的后续提交，checkpoint 本身没有随之更新。该证据能确定公开评估路径，**不能反推未知训练源码中的 forward**。
3. **BLOCKING — 公开提交历史不能调和 RRUFF 与 SI 的划分冲突。** `datasets/RRUFF` 只在 2024-09-27 上传过一次；初次上传与当前 `main` 的 5 个文件 blob SHA 全部相同。公开历史中没有 SI 所列 7644/1698/3554 的另一版数组或 split，`config.py` 也只有首版的 `train_size=0.8`。
4. **已确认 — Bacteria 的同期官方公开评估采用当前 1024→模型内 256 路径。** `873.pth` 上传后 19 秒，`12b7d42` 同时启用模型内 pool256 并把 `test_model` 的 loader 输入改为 1024；其直接子提交又在 26 秒后加入已执行的 `case_analysis.ipynb`，明确用 `873.pth` 调用 `test_model`，保存输出 accuracy 0.87233。该 notebook 的 30 类逐类指标与本地 `input_pool=1024` 结果全部一致，而非 `input_pool=256` 的 0.869。协议判定来自提交依赖和完整预测指标，不是因为 87.2333% 更接近论文 87.3%。

## 公开时间线与代码证据

| 时间 | 提交 | 可核实事实 | 判定 |
|---|---|---|---|
| 2024-02-02 | [`30060e2` — upload scripts](https://github.com/X1nyuLu/PACE/commit/30060e2559524519dc40cdd3f26c19e4e63faa0e) | 同时加入 COV `924.pth`、首版 [`models/PACE.py`](https://github.com/X1nyuLu/PACE/blob/30060e2559524519dc40cdd3f26c19e4e63faa0e/models/PACE.py#L40-L73)、[`config.py`](https://github.com/X1nyuLu/PACE/blob/30060e2559524519dc40cdd3f26c19e4e63faa0e/config.py)、[`utils/utils.py`](https://github.com/X1nyuLu/PACE/blob/30060e2559524519dc40cdd3f26c19e4e63faa0e/utils/utils.py#L205-L215) 和 COV 测试日志。首版模型无 `wave_emb`，内部 `adaptive_avg_pool1d` 被注释。 | checkpoint 的额外层在公开首版源码中已经无法解释。 |
| 2024-04-12 | [Issue #1 作者回复](https://github.com/X1nyuLu/PACE/issues/1#issuecomment-2050792360) | 作者称 config 与 COV 924 checkpoint 对应，并确认仓库代码使用 adaptive pooling，而论文描述 linear interpolation；表示可能后续修正。 | 说明论文/仓库预处理存在已知差异；没有提供含 `wave_emb` 的训练源码。 |
| 2024-09-27 10:55 UTC | [`d1f1062` — upload datasets](https://github.com/X1nyuLu/PACE/commit/d1f1062dae95512731e177de666a33e8c979f76b) | `datasets/RRUFF` 和 Bacteria `test_x.npy`/`test_y.npy` 首次进入可达历史。 | 两组数据后来均未更新；没有另一版公开数组可供选择。 |
| 2024-09-27 11:28:37 UTC | [`37483e2` — upload checkpoint](https://github.com/X1nyuLu/PACE/commit/37483e2adbe347202825123aff8a9c2ce9c7a035) | 只加入 Bacteria `873.pth`；其父提交是 `d1f1062`。该快照仍是首版模型内部 pool 注释、standalone test loader pool256。 | 证明 checkpoint 上传瞬间的代码快照，但紧接着的官方修订和 notebook 给出更完整的评估证据。 |
| 2024-09-27 11:28:56 UTC | [`12b7d42` — modify some details](https://github.com/X1nyuLu/PACE/commit/12b7d4229cb4f189af3163a177f84d549816747f) | 只修改 `models/PACE.py` 和 `utils/utils.py`：启用模型内 `adaptive_avg_pool1d(..., 256)`，并把 standalone `test_model` 的 loader 池化由 PACE 的 256 改为统一 1024；两个版本均无 `wave_emb`。 | 这是后期公开评估行为变更；没有实现论文所述 linear interpolation，也没有更新 COV checkpoint。 |
| 2024-09-27 11:29:22 UTC | [`9bd72d3` — add scripts about visualization and analysis](https://github.com/X1nyuLu/PACE/commit/9bd72d3c70cc8e96c50f2e186ae439257881381e) | 父提交正是 `12b7d42`；加入官方 [`case_analysis.ipynb`](https://github.com/X1nyuLu/PACE/blob/9bd72d3c70cc8e96c50f2e186ae439257881381e/case_analysis.ipynb)，载入 `873.pth` 后调用 `test_model`，保存 0.87233 和完整分类报告。 | 直接确认作者随该代码版本公开的 Bacteria 评估路径和结果。 |

GitHub API 的逐路径历史计数为：`models/PACE.py` 2 次、`utils/utils.py` 2 次、`config.py` 1 次、`utils/transform.py` 1 次、`utils/dataloader.py` 1 次、`datasets/RRUFF` 1 次、`datasets/Bacteria` 1 次、两个公开 checkpoint 各 1 次。官方仓库只列出 `main`（当前 `4e1f541`）一个分支，tag 和 release 均为 0。这里的“未发现”严格限定为**官方仓库当前公开可达的 Git 引用**；它不排除未公开、已不可达或从未推送的训练源码。

## COV checkpoint 的键与加载行为

本地 checkpoint SHA-256 为 `f9e8a2e104e01c59520a752a4e731096da489bb9bf410d00cff40fc1aca444c1`，共有 72 个 state keys。公开模型能匹配其中 65 个，另有 7 个 checkpoint-only 键：

- `wave_emb.0.weight`：`[16, 16]`
- `wave_emb.0.bias`：`[16]`
- `wave_emb.1.weight`、`wave_emb.1.bias`、`wave_emb.1.running_mean`、`wave_emb.1.running_var`：各 `[16]`
- `wave_emb.1.num_batches_tracked`：标量

首版 [`load_net_state`](https://github.com/X1nyuLu/PACE/blob/30060e2559524519dc40cdd3f26c19e4e63faa0e/utils/utils.py#L205-L215) 只遍历当前模型自己的键并逐个复制，最后再加载 `net.state_dict()`；因此 checkpoint 中多出的 7 个键会被静默忽略。这个行为解释了为何首版源码可以运行 `924.pth`，但不证明额外模块在训练时是否参与 forward。

同期官方 [`2024-02-02_20_51_test.log`](https://github.com/X1nyuLu/PACE/blob/30060e2559524519dc40cdd3f26c19e4e63faa0e/logs/COV/PACE/2024-02-02_20_51_test.log) 记录 seed 2024、checkpoint 924、accuracy 0.92473，并给出三类的 precision/recall/F1。本地 `input_pool=256` 兼容评估得到 86/93、macro-F1 0.9243288，逐类指标与日志一致；`input_pool=1024` 虽也得到 86/93，但 macro-F1 为 0.9234149，逐类预测分布不同。这使 256 路径成为该 checkpoint 的**同期公开评估协议**，而不是原训练 forward 的证据。

因此，结果命名应保持以下边界：

- 可写：`COV 924，按同期公开 pool256 评估；显式忽略 7 个遗留 wave_emb 键，其余 65 键严格加载`。
- 不可写：`checkpoint 与原作者训练架构完全 strict 匹配`、`wave_emb 未参与训练`，或根据 92.47% 的数值吻合反推未知 forward。
- 当前 HEAD 的 1024→模型内 256 结果可以作为后期仓库协议对照；论文的 linear1024→pool256 应继续作为单独的预注册协议，不应混名或按 test 分数选择。

## Bacteria checkpoint、预处理与官方评估证据

Bacteria 相关公开对象的提交链是连续的：

```text
d1f1062 上传 test 数组
  └─ 37483e2 上传 873.pth
       └─ 12b7d42 改为 test loader 1024 + 模型内 pool256
            └─ 9bd72d3 上传带已执行输出的 case_analysis.ipynb
```

`utils/transform.py` 和 `utils/dataloader.py` 都只在首版 `30060e2` 提交过一次，之后 Git blob 始终分别为 `4aecbfbc6a6d73b3095cbee98c7d25b9620e0a01` 和 `5fa4121bfe68cf4139759ab7126d13c92e33cb9d`。Bacteria validation/test transform 固定为 Savitzky–Golay `Smooth(window=7, polyorder=3)` 后转 tensor；`dataloader.py` 再根据调用方传入的 `pool_dim` 执行 adaptive average pooling。它们自身没有决定 256 或 1024，差异来自 `utils.test_model` 的调用参数和模型 forward。

Bacteria `test_x.npy`/`test_y.npy` 并非 2024-02 首版内容，而是在 `d1f1062` 首次上传；从该提交到 HEAD 的 blob 一直是：

| 对象 | 首次提交 | Git blob SHA | HEAD 是否相同 |
|---|---|---|---|
| `datasets/Bacteria/test_x.npy` | `d1f1062` | `e9af52ce56d9a9e97223ea6ce7f67c5ed5cab03c` | 是 |
| `datasets/Bacteria/test_y.npy` | `d1f1062` | `779f531304c29a450e5409d9003de1a97cdb52bf` | 是 |
| `checkpoints/Bacteria/PACE/873.pth` | `37483e2` | `d413373a19d0643f1b0010dd42a266ee68633276` | 是 |

公开仓库没有单独的 Bacteria `logs/...test.log`；HEAD 的官方 `logs/` 树只有 COV 日志。不过，`case_analysis.ipynb` 是更具体的同期官方执行记录：

- notebook 源码构造 30 类 PACE，载入 `checkpoints/Bacteria/PACE/873.pth`，然后调用 `from utils.utils import test_model`。
- notebook 的父提交已经把 `test_model` 设为 loader pool1024，并启用了模型内部 pool256。
- 保存输出为 2617/3000，即 accuracy 0.87233、macro-F1 0.8679；本地 `input_pool=1024` 评估的 30 个类别 precision/recall/F1 按四位小数全部与 notebook 相同。
- 本地 `input_pool=256` 为 2607/3000，即 0.869，且多个类别指标不同，不能生成 notebook 保存的报告。

因此，对 **Bacteria 873 checkpoint 的作者公开评估**，证据支持 `Smooth → loader adaptive pool1024 → PACE internal adaptive pool256`。这项结论限定于 checkpoint 的公开测试路径；它不证明原训练数据预处理或 checkpoint 训练 forward。论文所述 linear interpolation 1024→pool256 仍是另一协议，不能把这里的两级 adaptive pooling改称为论文线性协议。论文 87.3% 与 notebook 87.2333% 的数值关系只作结果核对，不承担协议选择证据。

## RRUFF 历史与 SI 对齐

`d1f1062` 初次上传和当前 `main` 返回相同的文件大小与 Git blob SHA：

| 文件 | 大小（bytes） | Git blob SHA |
|---|---:|---|
| `label.json` | 28,458 | `88cf807f671aa8b36a298f2c26c323e70850bd89` |
| `test_x.npy` | 37,757,824 | `701148406ef6dc30f7f3efec198e0e7e0d49d7ab` |
| `test_y.npy` | 28,560 | `777aee455c356a2a84bb02c6da837d3790cab6f9` |
| `train_x.npy` | 90,240,384 | `9c36763d00941256ebb80cfcb99387cabf921dcf` |
| `train_y.npy` | 34,104 | `96034dfbf89aa92e69d2ae76ea3decf0a16857af` |

所以当前本地 RRUFF 的 8494 个 `train` 和 3554 个 `test` 就是官方仓库唯一公开版本。默认 0.8 内部分割产生 6795/1699；SI Table S2 的 7644/1698/3554 合计 12896，而 SI Table S1 总量为 12048。公开历史没有 8494 之外的训练数组，也没有显式 split indices，可以把 8494 恢复成 SI 的 7644+1698=9342。当前数据审计还确认 3554/3554 个 test rows 都能在 train 中找到，因此该公开版本仍只能用于仓库流程诊断，不能作为独立测试泛化证据。

没有发现与 SI 更一致的公开历史版本：

- `config.py` 从首版起只有 `train_size=0.8`，无 0.9 或 RRUFF 专用 7644/1698 split。
- RRUFF 路径只有一次数据提交，初次与 HEAD 的 blob 完全一致。
- 2024-09-27 的模型修改仍使用 adaptive pooling，没有引入论文所述 linear interpolation。

## 建议与阻断状态

- **BLOCKING**：在取得对应 `924.pth` 的原训练源码、明确的 `wave_emb` 说明或作者可验证说明前，保留 checkpoint 架构来源缺口；兼容白名单策略可以继续使用并写入每份 JSON。
- **BLOCKING**：在取得论文对应的 RRUFF 数组与原始 split indices 前，不把公开 RRUFF 结果升级为严格论文复现或独立泛化结果。
- **NON-BLOCKING**：将 COV 主 checkpoint 评估协议定名为“2024-02 同期公开 pool256 协议”；把 2024-09 HEAD 的 1024→256 和论文 linear1024→256 分列为协议审计。
- **NON-BLOCKING**：将 Bacteria 873 的作者公开 checkpoint 结果定名为“2024-09 官方 notebook 协议：Smooth→adaptive pool1024→模型内 adaptive pool256，87.2333%”；pool256 的 86.9% 保留为相邻旧快照对照，论文 linear1024→256 继续单列。
- **NON-BLOCKING**：最终报告引用提交 SHA、checkpoint SHA、RRUFF blob SHA，并明确“公开可达历史未发现”而非声称相关源码从未存在。

## 防御视角

真实产品中应将模型结构源码、预处理实现、checkpoint、数据 split indices 和评估日志绑定成同一签名制品。CI 应拒绝未知 checkpoint 键、隐式丢键加载、checkpoint 后改动预处理却不更新版本，以及 train/test 内容哈希重合；兼容迁移必须使用精确白名单并把忽略键、源码 commit 和数据 blob 写入不可变审计日志。

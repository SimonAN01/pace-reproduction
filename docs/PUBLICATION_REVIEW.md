# PACE 公开复现包独立发布复核

工号 RE-7X-2024 · 甲方安全实验室 · 内部技术资料

**复核结论：PASS，无 BLOCKING 问题。** 以实现提交 `03cb9b894dbb7f3a53604bad8306427a286a640e` 为基线，本包的目标参数、数据与权重锁定、训练/评估入口、真实标签核验和公开说明彼此一致，可以进入公开推送与 Release 发布。无需为本次发布再扩大科学实验。

本次按 ARIS `experiment-bridge` Phase 2.5 做独立代码复核。复核范围包括 `configs/`、两套源码树、冻结 runner/hook、新增发布脚本、归档 evidence、README 与发布验证记录。除本报告外，复核过程未修改实现文件，也未运行额外 GPU 实验；GPU 结果来自已完成的发布验证记录并经本次审阅。

## 1. 协议实现结论

- `covid_high` 正确实现 PyTorch 2.0.1、seed42、lr1e-3、200 epoch、train/validation 0.8、split seed42；训练/验证采用 adaptive pool256，最终主评估采用输入 adaptive pool1024 后由模型内部 pool256。未混入 linear1024 协议。
- COVID 专项通过 `scripts/train_cov_gap.py` 选择 `covid_source`，`PYTHONPATH` 提供只读观察 hook；作者 `924.pth` 由数据清单按 SHA 下载到硬编码依赖路径。hook 不读取测试标签、不改变 loss、optimizer step 或选模逻辑。
- Marine、RRUFF、Functional Groups 与 Bacteria 使用 PyTorch 2.2、seed2024、linear1024；Marine/RRUFF 的 train_size 为0.8，FG/Bacteria 为0.9。FG 使用17标签 subset accuracy。
- Bacteria 保持两阶段协议：reference 预训练50 epoch、lr1e-3、batch64；随后以预训练最佳权重初始化微调，最长200 epoch、lr1e-5、batch8、patience50。两阶段分别重置训练 seed。入口拒绝对预训练使用不成立的 `--epochs` sanity 覆盖。
- checkpoint 仅按 validation accuracy 选择，正式训练完成后才执行一次最终测试。评估直接加载公开 `test_y.npy`，没有以模型输出替代真实标签。

## 2. 证据门检查

| 检查 | 结果 |
|---|---|
| 12个核心源码 + 5个原 runner/hook 的 `frozen_code.json` SHA256 | 17/17 PASS |
| 数据及 COVID 观察依赖的 size/SHA256 | 30/30 PASS |
| Release 权重的 size/SHA256 | 7/7 PASS |
| 归档 logits→prediction、prediction→公开真标签、指标重算 | 普通模式及 `python -O` 均 PASS |
| 7个登记配置的 train/validation split 重建 | PASS |
| reported 五权重实际 CUDA 推理 | 全部 PASS，预测逐条一致，最大 logit 绝对差均为0.0 |
| COVID/Marine 单 epoch CUDA sanity | PASS，首轮 history 与原正式运行逐字段一致 |
| Markdown 本地链接、敏感路径/凭据模式、`git diff --check` | PASS |

reported 五项实际推理结果为：

| 目标 | 正确数/测试数 | Accuracy |
|---|---:|---:|
| COVID-19 | 84/93 | 90.3226% |
| Marine | 567/570 | 99.4737% |
| RRUFF | 3487/3554 | 98.1148% |
| Functional Groups | 1470/1950 | 75.3846% |
| Bacteria | 2567/3000 | 85.5667% |

机器可读验证记录见 `docs/PUBLICATION_VALIDATION.json`，人工摘要见 `docs/VALIDATION.md`。

## 3. 复核中发现并已修复的问题

1. 冻结 runner 原先通过父 Git 仓库读取 HEAD，却把它写成上游提交。发布启动器现保留该值为 `execution_repository_commit`，并把 `upstream_commit` 规范为固定上游 `4e1f541b5776e7fd1df14fce271b11a5cbfe3f87`；同时记录 launcher SHA。
2. `verify_evidence.py` 原用 Python `assert`，在 `python -O` 下会跳过校验。现已改为显式 `require`/`RuntimeError`，优化模式复测通过。
3. `reproduce_table.py` 曾用 `Path.resolve()` 展开 venv Python 符号链接，导致落到基础解释器并缺少 torch。现改为 `Path.absolute()` 保留 venv 入口；失败目录与日志保留，新目录复测五项通过。
4. evidence 与环境记录中的工作站绝对路径已替换为 `/original-workspace`；文档中的本地链接已映射到发布文件或明确标为未随包发布。敏感路径与常见凭据模式复扫无命中。
5. README 已明确区分 reported 混合配置与统一 linear baseline，修正环境强制核对项数量，并披露数据重叠、未知作者参数和 checkpoint-guided reconstruction 的边界。

## 4. 非阻断限制与发布条件

- 本轮没有重新执行五数据集完整200 epoch训练；已有完整原运行证据，新增验证覆盖两项真实单轮启动与五权重完整推理。这一范围足以验证发布包装和入口，不等同于再次证明跨硬件训练逐比特确定性。
- 正式入口依赖 Git HEAD，因此支持 `git clone`，不支持无 `.git` 的 GitHub Source ZIP。README 与启动错误信息均已说明。
- GPU、驱动、CUDA/cuDNN或操作系统变化仍可能改变从头训练轨迹；发布文档没有承诺跨机器得到同一 checkpoint。
- `bootstrap.py models` 依赖 GitHub Release `v1.0.0`。公开发布时须上传 `configs/model_manifest.json` 列出的7个同名资产，并在发布端再次执行 size/SHA256 校验。
- 上游未提供覆盖全部代码/数据的统一许可。README 已保留来源、权属与再分发限制说明；这属于发布治理事项，不改变本次科学实现结论。

## 5. 防护建议

真实科研或产品复现包应继续采用本包已落实的控制：数据、权重和执行代码分别做 SHA256 锁定；测试标签只在最终评估读取；训练失败使用新目录并保留失败日志；来源提交与执行仓库提交分字段记录；发布前扫描凭据、用户名和内部绝对路径；Release 资产上传后从公开端点重新下载并核验。对于存在样本或主体重叠的数据集，应按患者、培养批次或化合物分组重新划分并做外部验证，避免把重叠测试准确率解释为独立泛化能力。

**最终建议：GO。** 提交本报告、`docs/VALIDATION.md` 与 `docs/PUBLICATION_VALIDATION.json` 后即可推送公开仓库；创建 `v1.0.0` Release 并完成7个权重资产的公开端下载核验后，复现入口闭环完成。

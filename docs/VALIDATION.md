# 发布包验证记录

工号 RE-7X-2024 · 甲方安全实验室 · 内部技术资料

以下检查验证发布代码的可执行性，不表示已经再次从头训练全部五个数据集。原始200epoch训练证据位于 evidence/。

- 30项数据及观察依赖文件的字节数与SHA256：通过。
- 17个原训练核心/runner/hook文件与原运行快照：逐文件相同。
- 归档模型指标：普通及Python -O模式均从公开真实标签重算通过；7个配置的实际split重建一致。
- 公共下载端点：COV label、Bacteria tune_y实际下载SHA通过；Bacteria 480000128字节reference端点支持Range且返回正确NPY头。
- 实际CUDA单epoch训练：covid_high、marine均完成，输出不含最终test（sanity与正式结果分离）。
- 失败记录：首次整表推理启动将venv Python用Path.resolve解析到基础解释器，导致找不到torch，模型尚未创建。修复为Path.absolute以保留venv入口；失败目录与日志在原工作区保留，不覆盖。
- 修复后五权重实际CUDA推理：84/93、567/570、3487/3554、1470/1950、2567/3000，全部与原运行预测逐条相同。实际环境、logit差、权重SHA、sanity首轮对照保存在 `PUBLICATION_VALIDATION.json`。
- 发布包独立代码审查：见 `PUBLICATION_REVIEW.md`，完成后与本报告一同提交。

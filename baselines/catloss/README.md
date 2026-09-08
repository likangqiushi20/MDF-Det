# CATLoss

本目录保存 CATLoss 的独立复现代码、配置与测试。论文中未披露的实现选择记录在 `paper_spec.json` 和 `configs/` 中。

当前状态：

- L1 网络结构、热图和 CATLoss 已实现。
- 真实 AOI01 四帧 GPU 小批量过拟合门禁已通过。
- 下一阶段是按帧动态构建正样本、普通负样本与 BGS 困难负样本池。
- `evaluate_self_test.py` 和 `evaluate_all_self_test.py` 已支持冻结阈值后的
  单 AOI/六 AOI SELF-TEST 聚合。时序启动重复 SELF-TEST 首帧，禁止跨 split
  读取 TRAIN 历史；滑动帧缓存避免重复读取 NITF。

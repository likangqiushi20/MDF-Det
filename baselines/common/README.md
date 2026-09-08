# 公共复现模块

Phase 0 在此实现以下共享能力：

1. 扫描 NITF 并生成 `resolution_audit.json` 与 `dataset_manifest.json`。
2. 定义逐论文的 `ResolutionProfile`、`AOIProfile`、`SplitProfile` 和 `EvaluationProfile`。
3. 实现地理坐标与像素坐标变换、`TruthPolicy` 和严格一对一检测匹配。
4. 用首、中、末三帧及人工 TP/FP/FN 样例完成单元测试。

公共协议通过验收前，不批量导出训练 patch，也不启动模型训练。

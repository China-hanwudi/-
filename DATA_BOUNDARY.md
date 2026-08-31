# 数据与发布边界（科研协作版）

最后修订：2026-08-09

本仓库服务 CARMA-Affect / N3 **科研协作**。默认**尽量允许**上传有利于复现与合作的材料；仅保留法律与安全上不能放开的底线。

仓库 MIT 只覆盖本仓库自有代码与文档；**不替代**第三方数据/模型原许可证。

---

## 鼓励上传（科研友好）

- 代码、配置、测试、文档、流程图、聚合指标与曲线；
- **开源许可允许再分发的预训练权重**（如 MIT 的 EmoBERTa 等），放在 `模型/artifacts/pretrained/`，并附 `MODEL_CARD.md`（model_id、许可、来源、引用）；
- **团队自训 checkpoint**、指标卡，放在 `模型/artifacts/checkpoints/`；
- 合成数据、合同测试夹具；
- 在许可证允许前提下的特征缓存、中间结果（须在目录内写明来源与许可）；大文件用 **Git LFS**。

---

## 底线禁止（不是“抠门”，是不能靠改协议变合法）

1. **官方禁止再分发的数据集正文镜像**  
   例如未获再分发权时，把 MELD / EmotionTalk / IEMOCAP / CPED / M3ED 的原始音视频、转写全文包、官方压缩包整包推进本仓。  
   → 请继续用 [`datasets/`](datasets/) 脚本在**仓库外**下载。若已取得**书面再分发授权**，在下方备案后即可上传对应范围。

2. **密钥与隐私**  
   token、密码、Cookie、私钥、未脱敏的伦理批复/授权邮件原文、可直接识别被试的隐私材料。

---

## 授权备案（有书面再分发权时填写）

| 日期 | 范围 | 授权方 | 证明存放（勿提交密钥） | 经办人 |
|---|---|---|---|---|
|  |  |  |  |  |

---

## 推荐目录

```text
模型/artifacts/pretrained/     # 开源预训练权重（本仓已放 EmoBERTa-base）
模型/artifacts/checkpoints/    # 团队自训 N3 权重
模型/artifacts/metrics/        # 聚合指标
```

## Temporal N3 v4 products

`temporal_n3/` only contains public source, documentation, and synthetic tests.
Candidate manifests, per-example routing outputs, OOF predictions, extracted
features, receipts that reveal private paths or identifiers, checkpoints, and
all run logs remain local-only under the ignored `temporal_n3/artifacts/`,
`temporal_n3/features/`, or `temporal_n3/runs/` paths. Public aggregate results
may be added only after the corresponding data license and final-evaluation
protocol permit release.

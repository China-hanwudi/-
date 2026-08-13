# 官方数据获取与本地目录规范

本目录只保存下载工具、固定版本和校验清单，**不保存 MELD、IEMOCAP、EmotionTalk 或其他第三方数据的原始文本、标签、音频、视频、压缩包和派生特征**。

这样做有三个目的：

1. 每位协作者从官方入口取得数据并独立接受许可证或 gated 条款；
2. 原始数据不经过公开 GitHub 二次托管；
3. MELD、EmotionTalk 的正式 test 以及 IEMOCAP 每个外层 fold 的 held-out 角色继续按各自协议隔离，避免相应折的开发阶段读取评估标签。

## 固定来源

| 数据集 | 固定来源 | 许可证/访问状态 | 仓库策略 |
|---|---|---|---|
| MELD | [`declare-lab/MELD@e8cedf27`](https://github.com/declare-lab/MELD/tree/e8cedf27b5d2877e198332c957127e16eb214afe)；媒体由官方 README 指向密歇根大学下载地址 | 官方代码/标注仓库标记 GPL-3.0；媒体包含第三方影视内容，不能由本仓库重新授权 | 下载 train/dev 标注；test 仅 evaluator 下载；媒体始终从官方地址获取 |
| IEMOCAP | [USC SAIL 官方入口](https://sail.usc.edu/iemocap/) | 需按官方流程取得个人授权；本仓库不代理授权，也不重新分发语料 | 合法取得并在仓库外校验/解压；按预注册外层 Session 五折分别评估；原始内容和派生产物均不提交 |
| EmotionTalk | [`BAAI/Emotiontalk@adbc17fc`](https://huggingface.co/datasets/BAAI/Emotiontalk/tree/adbc17fc944e8cf2873643906160c6ca0259ab61) | CC BY-NC-SA 4.0，Hugging Face gated/auto；每位使用者须自行登录并接受条款 | 使用官方 gated 下载；任何 archive 不进入本仓库 |

本项目的 MIT License 只覆盖本仓库自行编写的内容，不覆盖上述数据。

## 推荐本地目录

数据根目录应放在 Git 仓库之外。如果必须放在仓库工作区内，只能放入已被 `.gitignore` 排除的 `datasets/local/`。

```text
<DATA_ROOT>/
├── MELD/
│   └── e8cedf27b5d2877e198332c957127e16eb214afe/
│       ├── annotations/
│       │   ├── train_sent_emo.csv
│       │   ├── dev_sent_emo.csv
│       │   └── test_sent_emo.csv       # evaluator-only
│       └── MELD.Raw.tar.gz             # 可选，约 10.13 GiB
└── EmotionTalk/
    └── adbc17fc944e8cf2873643906160c6ca0259ab61/
        ├── Audio.tar                   # 约 13.79 GiB
        ├── Multimodal.tar              # 约 19.83 GiB
        ├── Text.tar
        ├── Video.tar
        └── README.md
```

不要把真实绝对路径写入配置或提交记录。实验配置应从命令行参数或未纳入 Git 的本地环境文件读取 `<DATA_ROOT>`。

## 下载命令

### MELD

默认只下载固定 commit 的 train/dev 标注：

```powershell
$dataRoot = 'D:\N3_data'
powershell -NoProfile -ExecutionPolicy Bypass -File datasets\scripts\download_official_data.ps1 `
  -Dataset MELD `
  -Destination (Join-Path $dataRoot 'MELD\e8cedf27b5d2877e198332c957127e16eb214afe')
```

只有指定 evaluator 才能加入 `-IncludeTest`。需要官方媒体时再加入 `-IncludeMedia`；媒体约 10.13 GiB，脚本支持通过 `curl` 断点续传。官方媒体端点没有随本仓库发布的可信 SHA-256，因此下载完成后必须在内部冻结清单中记录本地哈希，不能把未核验媒体用于确认性运行。

### EmotionTalk

先安装官方下载依赖，并以每位协作者自己的 Hugging Face 账号完成 gated 授权：

```powershell
python -m pip install requests huggingface_hub
hf auth login

$dataRoot = 'D:\N3_data'
powershell -NoProfile -ExecutionPolicy Bypass -File datasets\scripts\download_official_data.ps1 `
  -Dataset EmotionTalk `
  -Destination (Join-Path $dataRoot 'EmotionTalk\adbc17fc944e8cf2873643906160c6ca0259ab61')
```

默认下载较小的 `Text.tar`、`Video.tar` 和数据卡；加入 `-IncludeMedia` 后才下载 `Audio.tar` 与 `Multimodal.tar`。脚本调用仓库内的可恢复下载器，并在完成后自动检查大小和 SHA-256。

EmotionTalk archive 同时覆盖多个数据角色。模型开发人员即使已合法下载，也不得读取或提取 test 标签；test 只能交给一次性 label-only evaluator。

## SHA-256 校验

- [MELD train/dev 清单](manifests/meld-development-e8cedf27.sha256)
- [MELD evaluator-only test 清单](manifests/meld-evaluator-e8cedf27.sha256)
- [EmotionTalk 核心标注与数据卡清单](manifests/emotiontalk-core-adbc17fc.sha256)
- [EmotionTalk 大体积媒体清单](manifests/emotiontalk-media-adbc17fc.sha256)

手动复核示例：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File datasets\scripts\verify_sha256.ps1 `
  -Manifest datasets\manifests\meld-development-e8cedf27.sha256 `
  -Root 'D:\N3_data\MELD\e8cedf27b5d2877e198332c957127e16eb214afe'
```

校验脚本只读文件并计算哈希，不读取 CSV/JSON 标签内容。

## 已解压后的后续执行

“已解压”只说明归档内容已经落盘，**不等于数据、代码或训练入口已经通过可训练性验收**。MELD 与 IEMOCAP 应分别按以下操作单推进：

- [MELD 已解压数据后续执行全流程](../docs/15_MELD_已解压数据后续执行全流程_2026-08-13.md)；
- [IEMOCAP 已解压数据后续执行全流程](../docs/16_IEMOCAP_已解压数据后续执行全流程_2026-08-13.md)。

两份操作单都要求从只读 preflight 开始，逐 Gate 生成审计证据。若代码能力审计发现缺少真实数据的正式 CLI，应先实现、测试并通过相应 Gate；不得用 synthetic trainer、旧 sidecar 或不符合当前 Qwen 三模态合同的入口替代。MELD、IEMOCAP、EmotionTalk 使用同一冻结框架但分别训练、验证和评估，不合并数据集，也不以一个数据集的权重替代另外两个数据集内实验。

## 协作者交付规则

- 可以提交：下载脚本、固定 revision、文件名、字节数、SHA-256、官方链接和许可说明；
- 不得提交：原始 archive、解压文件、标签、媒体、逐样本 manifest/索引、派生特征、任何 Qwen/训练权重、服务器 checkpoint/日志、服务器地址和私有绝对路径；
- 传递数据时只发送官方链接和校验清单，不通过 GitHub、聊天附件或网盘二次分发；
- 评估标签访问者与模型开发者分离：MELD/EmotionTalk 按一次性 test 协议，IEMOCAP 按预注册 outer-fold OOF 协议；最终只返回 aggregate-only 结果。

# 官方来源与冻结版本

访问日期：2026-08-22（Asia/Shanghai）。只采用官方仓库、官方模型页和 MLX 官方文档；搜索摘要不作为实现依据。

## 模型与适配器

- 正式 RL 基座：[ModelScope `SenseNova/SenseNova-U1.5-8B-MoT`](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT)，冻结 repository revision `1c75534f02a734f7bb6319a16fd82a67769d12bb`。
- 正式 8 步适配器：[ModelScope `SenseNova/SenseNova-U1.5-8B-MoT-LoRAs`](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-LoRAs)，冻结 repository revision `da0b25652ee8f8b447bf817e4bfdbb4221d521ec`，目标文件 `SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`。
- 基座和 LoRA 的官方 README 均声明 Apache License 2.0。LoRA README 明确写明它只适配正式 `SenseNova-U1.5-8B-MoT`，不兼容 `SenseNova-U1.5-8B-MoT-Preview`。

所有下载 URL 必须带上述不可变 revision。`master`、`main` 或模型页当前态只用于发现，不进入下载 manifest。

## 官方实现

- 论文：[SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture](https://arxiv.org/abs/2605.12500)，arXiv `2605.12500`。
- 参考实现：[OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)，本次冻结到 commit [`35fb2fc520869aa09223a819882abdb3dc328ecd`](https://github.com/OpenSenseNova/SenseNova-U1/tree/35fb2fc520869aa09223a819882abdb3dc328ecd)。正式模型卡仍指向仓库的 `feat/u1.5` 路线；实现时以冻结 commit 中的 U1.5 路径和正式 checkpoint 的 config/header 双重校验。
- 核心语义来源：[`modeling_neo_chat.py`](https://github.com/OpenSenseNova/SenseNova-U1/blob/35fb2fc520869aa09223a819882abdb3dc328ecd/src/sensenova_u1/models/neo_unify/modeling_neo_chat.py)、[`modeling_qwen3.py`](https://github.com/OpenSenseNova/SenseNova-U1/blob/35fb2fc520869aa09223a819882abdb3dc328ecd/src/sensenova_u1/models/neo_unify/modeling_qwen3.py)、[`modeling_fm_modules.py`](https://github.com/OpenSenseNova/SenseNova-U1/blob/35fb2fc520869aa09223a819882abdb3dc328ecd/src/sensenova_u1/models/neo_unify/modeling_fm_modules.py) 与 [`examples/t2i/inference.py`](https://github.com/OpenSenseNova/SenseNova-U1/blob/35fb2fc520869aa09223a819882abdb3dc328ecd/examples/t2i/inference.py)。
- 上游环境声明：[官方 `pyproject.toml`](https://github.com/OpenSenseNova/SenseNova-U1/blob/35fb2fc520869aa09223a819882abdb3dc328ecd/pyproject.toml)，Python `>=3.10,<3.14`；官方参考环境是 PyTorch 2.8/CUDA 12.8，不是本项目运行时。

## MLX

- MLX 版本基线：[`mlx` 0.32.1](https://github.com/ml-explore/mlx/releases/tag/v0.32.1)，本机 `mlx-metal` 同为 0.32.1。
- API 依据：[MLX `scaled_dot_product_attention`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)、[`Conv2d`](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html)、[`RMSNorm`](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RMSNorm.html)、[`mlx.core.load`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html)。

## 打包

- sdist 文件筛选依据：[Hatch 官方 Build configuration](https://hatch.pypa.io/1.10/config/build/#file-selection)。`exclude` 使用 Git 风格 glob，源码包完整排除本地 `artifacts/`、`.venv/`、`dist/` 和内部进度记录；wheel/sdist 还由 `tools/release_audit.py` 二次检查。

## 公开证据边界

- 正式权重的真实 layer 0、1024 基座、8-step LoRA 和 2048 官方桶已经完成。公开仓库只保留一张精选样图与一份脱敏机器可读摘要，不保留原始 profile、提示词历史或本机路径。
- GitHub CI 只验证无权重单元/契约测试、构建和包内容，不下载 47.537 GiB 权重，也不把它表述为正式模型端到端验证。

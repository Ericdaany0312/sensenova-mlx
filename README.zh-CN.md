# SenseNova MLX

[English](README.md)

这是一个面向 Apple Silicon 的独立 MLX 文生图运行时，支持正式版
`SenseNova-U1.5-8B-MoT`、官方 8-step LoRA、命令行生成，以及常驻的 OpenAI
Images 兼容本地接口。模型 forward 由 MLX GPU 执行，运行时不依赖 PyTorch。

> 本项目是实验性的社区移植，不是 OpenSenseNova 官方发行版。模型权重需要
> 单独下载，永远不会进入本仓库、wheel 或 sdist。

![SenseNova U1.5 8-step LoRA 样图](docs/assets/showcase.png)

上图使用官方 8-step LoRA，以 2048×2048 生成。图片 SHA-256 和脱敏跑分记录见
[`docs/verification/e2e-results.json`](docs/verification/e2e-results.json)，精确
提示词、seed 和机器信息见 [`docs/assets/showcase.md`](docs/assets/showcase.md)。

## 已实现能力

- 冻结 revision、逐文件 SHA-256 校验的正式 RL 基座。
- 官方 8-step LoRA 快速生成。
- 基于 MLX GPU 的多模态 Transformer、prefix KV cache、flow 采样和像素解码。
- 基座质量模式与 LoRA 快速模式。
- 默认仅监听回环地址、带 Bearer 鉴权和输入校验的 OpenAI Images 兼容服务。
- 独立的 PyTorch CPU ↔ MLX BF16 正式 layer 0 数值对齐工具。

## 环境要求

- Apple Silicon Mac。
- 支持 MLX GPU 的 macOS。
- Python 3.11、3.12 或 3.13。
- 正式基座与 LoRA 约占 48 GiB，另外还需安装和输出空间。

已验证机器为 Apple M5 Max、40 核 GPU、128 GB 统一内存。其他芯片和内存容量
尚未系统测试，因此本文峰值是实测证据，不代表最低硬件保证。

## 安装

先安装 [`uv`](https://docs.astral.sh/uv/)，再克隆仓库并创建锁定环境：

```bash
git clone https://github.com/<owner>/sensenova-mlx.git
cd sensenova-mlx
uv sync --frozen --python 3.12
```

模型目录按以下优先级解析：

1. `SENSENOVA_MODEL_DIR`
2. `$XDG_CACHE_HOME/sensenova-mlx/SenseNova-U1.5-8B-MoT`
3. `~/.cache/sensenova-mlx/SenseNova-U1.5-8B-MoT`

所有命令也都可以显式传入 `--model-dir`。

## 下载与校验权重

包内 manifest 冻结正式基座 revision
`1c75534f02a734f7bb6319a16fd82a67769d12bb` 和 LoRA revision
`da0b25652ee8f8b447bf817e4bfdbb4221d521ec`。完整下载量为
51,042,331,251 字节，也就是 47.537 GiB。

```bash
export SENSENOVA_MODEL_DIR="$HOME/.cache/sensenova-mlx/SenseNova-U1.5-8B-MoT"
.venv/bin/sensenova-mlx download
.venv/bin/sensenova-mlx verify
```

下载器使用 `.partial` 文件断点续传，完成后核对精确大小和 SHA-256，再原子改名；
已有但校验失败的正式文件不会被静默覆盖。

## 生成图片

基座质量模式：

```bash
.venv/bin/sensenova-mlx generate \
  --prompt "一只戴红围巾的狐狸，坐在雨后的上海弄堂，电影摄影" \
  --width 1024 --height 1024 --steps 50 --seed 42 \
  --output output.png
```

官方 8-step LoRA：

```bash
.venv/bin/sensenova-mlx generate \
  --prompt "一只戴红围巾的狐狸，坐在雨后的上海弄堂，电影摄影" \
  --width 2048 --height 2048 --steps 8 --lora-8step --seed 42 \
  --output output-lora8.png
```

未显式指定时，基座 CFG 为 `4.0`，8-step LoRA 使用官方推荐的 `1.0`。每张图
旁边都会写入 JSON profile，记录冻结 revision、seed、耗时、内存、swap 和像素统计。

## 启动本地图片 API

快速模式会加载一次官方 8-step LoRA，后续请求复用常驻 MLX pipeline：

```bash
export SENSENOVA_API_KEY='请替换成至少16字符的本机随机密钥'
.venv/bin/sensenova-mlx serve --mode fast
```

健康检查无需鉴权：

```bash
curl http://127.0.0.1:11235/health
```

通过 OpenAI Images 兼容接口生成一张图：

```bash
curl http://127.0.0.1:11235/v1/images/generations \
  -H "Authorization: Bearer $SENSENOVA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "SenseNova-U1.5-8B-MoT-LoRA-8step",
    "prompt": "抽象蓝白会议背景，干净的几何图形和光影，不含文字",
    "size": "1440x2560",
    "n": 1,
    "seed": 42
  }'
```

服务限制单次请求不超过 32 KiB、提示词不超过 4000 字符、每次只生成一张图，
只接受已验证尺寸，GPU 忙时返回 HTTP 429。默认只绑定回环地址；考虑
`--allow-remote` 前必须先阅读 [`SECURITY.md`](SECURITY.md)。

## 本机跑分

以下耗时均包含模型加载。测试机器为 Apple M5 Max、40 核 GPU、128 GB 统一
内存，MLX 0.32.1；所有记录的 swap 增量均为 0。

| 模式 | 分辨率 / 步数 | 耗时 | MLX 峰值 |
| --- | ---: | ---: | ---: |
| 基座烟测 | 1024×1024 / 1 | 20.6 秒 | 32.18 GiB |
| 基座质量模式 | 1024×1024 / 50 | 85.9 秒 | 32.94 GiB |
| 8-step LoRA | 1024×1024 / 8 | 25.9 秒 | 33.66 GiB |
| 8-step LoRA 官方桶 | 2048×2048 / 8 | 44.1 秒 | 34.39 GiB |

正式 layer 0 对齐中，理解专家 cosine 为 `0.9999977709`，生成专家读取 prefix
KV cache 的 cosine 为 `0.9999998869`，验收下限为 `0.999`。机器可读结果见
[验证摘要](docs/verification/e2e-results.json)。这些数字只代表单机实测，不做泛化承诺。

## 测试与构建

默认测试既不需要 PyTorch，也不需要模型权重：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.test_config_flow tests.test_download tests.test_generate_cli \
  tests.test_packaging tests.test_rope_attention tests.test_server tests.test_weights
uv build
```

数值对齐工具独立运行，因为它需要可选的 `parity` 环境和正式权重：

```bash
uv sync --extra parity
uv run python tools/real_layer0_parity.py --model-dir "$SENSENOVA_MODEL_DIR"
```

## 可复现边界

- 运行时包内含精确的 28 文件下载账本。
- 1111 个所需 tensor 按 shard 加载，正式发布的 F32 tensor 在内存中转 BF16，
  不创建第二份持久化 checkpoint。
- MLX active 超过 90 GiB 或 swap 相对启动时增加超过 4 GiB 时会停止；1024
  验收上限为 70 GiB。
- 1024×1024 是超出官方质量桶的迁移烟测尺寸；2048×2048 是已验证的正式方图桶。
- 0.2.0 不包含图像编辑、训练、量化、4K、GUI 或 ComfyUI 节点。

官方来源、冻结 revision 和权重 header 审计见
[`docs/SOURCES.md`](docs/SOURCES.md) 与
[`docs/weights-audit.md`](docs/weights-audit.md)。

## 许可证与引用

项目代码采用 Apache-2.0。模型权重和 LoRA 是独立产物，使用前请核对其官方仓库
条款。归属说明见 [`NOTICE`](NOTICE)，引用元数据见 [`CITATION.cff`](CITATION.cff)。

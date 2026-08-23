# SenseNova MLX

[简体中文](README.zh-CN.md)

An independent MLX runtime for local text-to-image inference with the formal
`SenseNova-U1.5-8B-MoT` checkpoint on Apple silicon. It provides a command-line
generator and a persistent, OpenAI Images-compatible local HTTP endpoint. The
runtime performs model forward passes with MLX and does not depend on PyTorch.

> This is an experimental community port, not an official OpenSenseNova release.
> Model weights are separate downloads and are never included in this repository
> or its Python packages.

![SenseNova U1.5 8-step LoRA sample](docs/assets/showcase.png)

The sample above was generated at 2048×2048 with the official 8-step LoRA. Its
SHA-256 and the sanitized benchmark record are in
[`docs/verification/e2e-results.json`](docs/verification/e2e-results.json); exact
prompt, seed, and machine metadata are in
[`docs/assets/showcase.md`](docs/assets/showcase.md).

## What works

- Formal RL base checkpoint with pinned revision and per-file SHA-256 checks.
- Official 8-step LoRA for fast generation.
- MLX GPU inference for the multimodal transformer, prefix KV cache, flow
  sampling, and pixel decoder.
- Base quality mode and LoRA fast mode.
- Loopback-only OpenAI Images-compatible service with Bearer authentication,
  validated inputs, and single-GPU concurrency.
- Independent PyTorch CPU ↔ MLX BF16 layer-0 parity tooling.

## Requirements

- An Apple silicon Mac.
- macOS with MLX GPU support.
- Python 3.11, 3.12, or 3.13.
- About 48 GiB for the formal base checkpoint and LoRA, plus installation and
  output space.

The verified machine was an Apple M5 Max with a 40-core GPU and 128 GB unified
memory. Other chips and memory sizes have not yet been characterized, so the
recorded peak values are evidence, not a minimum-hardware guarantee.

## Install

Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and create the
locked environment:

```bash
git clone https://github.com/<owner>/sensenova-mlx.git
cd sensenova-mlx
uv sync --frozen --python 3.12
```

The model directory is resolved in this order:

1. `SENSENOVA_MODEL_DIR`
2. `$XDG_CACHE_HOME/sensenova-mlx/SenseNova-U1.5-8B-MoT`
3. `~/.cache/sensenova-mlx/SenseNova-U1.5-8B-MoT`

You can also pass `--model-dir` to every command.

## Download and verify weights

The bundled manifest freezes the formal base revision
`1c75534f02a734f7bb6319a16fd82a67769d12bb` and LoRA revision
`da0b25652ee8f8b447bf817e4bfdbb4221d521ec`. The complete download is
51,042,331,251 bytes (47.537 GiB).

```bash
export SENSENOVA_MODEL_DIR="$HOME/.cache/sensenova-mlx/SenseNova-U1.5-8B-MoT"
.venv/bin/sensenova-mlx download
.venv/bin/sensenova-mlx verify
```

Downloads use `.partial` files for resume, verify exact size and SHA-256, and
rename atomically only after verification. Existing invalid files are not
silently overwritten.

## Generate an image

Base quality mode:

```bash
.venv/bin/sensenova-mlx generate \
  --prompt "a red fox in a rain-washed Shanghai alley, cinematic photography" \
  --width 1024 --height 1024 --steps 50 --seed 42 \
  --output output.png
```

Official 8-step LoRA:

```bash
.venv/bin/sensenova-mlx generate \
  --prompt "a red fox in a rain-washed Shanghai alley, cinematic photography" \
  --width 2048 --height 2048 --steps 8 --lora-8step --seed 42 \
  --output output-lora8.png
```

Unless overridden, base mode uses CFG `4.0`, while 8-step LoRA mode uses the
officially recommended CFG `1.0`. A JSON profile is written next to each image
with the frozen revisions, seed, timing, memory, swap, and pixel statistics.

## Run the local image API

Fast mode loads the official 8-step LoRA once and reuses the persistent MLX
pipeline:

```bash
export SENSENOVA_API_KEY='replace-with-a-random-secret-of-16-or-more-characters'
.venv/bin/sensenova-mlx serve --mode fast
```

Health checks do not require authentication:

```bash
curl http://127.0.0.1:11235/health
```

Generate one image using the OpenAI Images-compatible route:

```bash
curl http://127.0.0.1:11235/v1/images/generations \
  -H "Authorization: Bearer $SENSENOVA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "SenseNova-U1.5-8B-MoT-LoRA-8step",
    "prompt": "abstract blue and white conference background, no text",
    "size": "1440x2560",
    "n": 1,
    "seed": 42
  }'
```

The server accepts at most 32 KiB per request, limits prompts to 4,000
characters, generates only one image per call, validates supported dimensions,
and returns HTTP 429 while the GPU is busy. It binds only to loopback by default.
Read [`SECURITY.md`](SECURITY.md) before considering `--allow-remote`.

## Measured performance

All timings include model loading. Measurements were recorded on an Apple M5 Max
(40-core GPU, 128 GB unified memory) with MLX 0.32.1; swap delta was zero in every
listed run.

| Mode | Resolution / steps | Time | MLX peak |
| --- | ---: | ---: | ---: |
| Base smoke | 1024×1024 / 1 | 20.6 s | 32.18 GiB |
| Base quality | 1024×1024 / 50 | 85.9 s | 32.94 GiB |
| 8-step LoRA | 1024×1024 / 8 | 25.9 s | 33.66 GiB |
| 8-step LoRA, official bucket | 2048×2048 / 8 | 44.1 s | 34.39 GiB |

Formal layer-0 parity measured cosine `0.9999977709` for the understanding expert
and `0.9999998869` for the generation expert reading the prefix KV cache, against
a minimum threshold of `0.999`. See the
[verification summary](docs/verification/e2e-results.json) for machine-readable
values. These are single-machine measurements, not broad performance claims.

## Test and build

The default suite requires neither PyTorch nor model weights:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.test_config_flow tests.test_download tests.test_generate_cli \
  tests.test_packaging tests.test_rope_attention tests.test_server tests.test_weights
uv build
```

The numerical parity tool is intentionally separate because it requires the
optional `parity` environment and formal weights:

```bash
uv sync --extra parity
uv run python tools/real_layer0_parity.py --model-dir "$SENSENOVA_MODEL_DIR"
```

## Reproducibility and boundaries

- The exact 28-file download ledger is packaged with the runtime.
- 1,111 required tensors are loaded shard by shard; released F32 tensors are cast
  to BF16 in memory without creating a second persistent converted checkpoint.
- A 90 GiB MLX active-memory ceiling and a 4 GiB swap-growth ceiling stop unsafe
  runs. The 1024 acceptance ceiling is 70 GiB.
- 1024×1024 is a migration smoke size outside the official quality buckets;
  2048×2048 is the verified square production bucket.
- Image editing, training, quantization, 4K, GUI, and ComfyUI nodes are not part of
  version 0.2.0.

Official sources, frozen revisions, and the weight-header audit are documented in
[`docs/SOURCES.md`](docs/SOURCES.md) and
[`docs/weights-audit.md`](docs/weights-audit.md).

## License and citation

Project code is available under Apache-2.0. Model weights and LoRA adapters are
separate artifacts; review their official repository terms before use. See
[`NOTICE`](NOTICE) for attribution and [`CITATION.cff`](CITATION.cff) for citation
metadata.

# SenseNova U1.5 正式权重头部审计

审计时间：2026-08-22。方法：读取官方 ModelScope 文件清单、config/index 和每个 safetensors 的 8 字节长度前缀与 JSON header；未下载任何 tensor payload。机器可读逐 tensor 结果见 `weights-audit.json`。

## 结论

- 基座：`SenseNova/SenseNova-U1.5-8B-MoT@1c75534f02a734f7bb6319a16fd82a67769d12bb`。
- 分片：13 个、`50,222,152,952` 字节（46.773 GiB）。编号是 `00001`、`00005` 至 `00016`；`00002` 至 `00004` 从未出现在该正式 revision 的 index/文件清单中，不是本地漏下。
- 张量：1116 个；参数：`17,532,854,464`。
- BF16：597 个张量、`9,954,706,880` 参数、`19,909,413,760` payload 字节。
- F32：519 个张量、`7,578,147,584` 参数、`30,312,590,336` payload 字节。
- index 与 13 个 header 完全闭合：1116 个 key 无缺失、无多余，所有 payload offset 都与官方文件大小一致。
- header 审计总传输 `5,447,762` 字节，其中 config/tokenizer/index 等元数据 `5,298,906` 字节，分片前缀加 header `148,856` 字节，低于 20MB 授权上限。

正式文件体积大的原因已由真实 header 证明：文本/理解分支大多是 BF16，而生成专家 `*_mot_gen`、时间/噪声嵌入等大量参数以 F32 发布。运行时基线会逐 shard 读入并把 F32 生成权重转换为 MLX BF16；不先制造第二份持久化模型副本。

## 分片账本

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `model-00001-of-00016.safetensors` | 105,413,672 | `3a6c016a43a782da79e4c69b0eb9ef9ef2b929f74913146eb5f642bd8984b410` |
| `model-00005-of-00016.safetensors` | 162,434,760 | `c1716a331e5293bc7c4fbf687388fa26043781f7910db2063edc17657e9d1566` |
| `model-00006-of-00016.safetensors` | 7,120,064,448 | `5f373bb4710af83b50c94faf348f08e56fddaf2aa868e3c5cfa9ab014eba4ab9` |
| `model-00007-of-00016.safetensors` | 4,630,728,192 | `4855daa0dcb29f49ed660ebfa9e6f00da8503b03aeaba01ee96dcb3f40dae1ff` |
| `model-00008-of-00016.safetensors` | 4,630,728,248 | `d74ead8dd35eeefd2d493496318435d00edeb7243ba33eec9950bb5581b3c0ec` |
| `model-00009-of-00016.safetensors` | 4,630,728,296 | `825cd519602c1d7ef6a0b839772fdadecf64c6ce468ba4fdb3fe07e5406dffc2` |
| `model-00010-of-00016.safetensors` | 4,630,728,296 | `18e94914ab4c7353954cbadef021c43072b36a5d70ac6f20d9d1e5ad87b66f59` |
| `model-00011-of-00016.safetensors` | 4,630,728,296 | `54af6c2750912cf5c79555d9c6746db2c9e0dbddd288ed22e4fc39057ca2a52c` |
| `model-00012-of-00016.safetensors` | 4,630,728,296 | `9641bad8a7bb3e7d37a1c4044451a6acdcf246c8e95b564ccbd79d7bd50c0e11` |
| `model-00013-of-00016.safetensors` | 4,630,728,296 | `60aa3d2c16d7d2e21e494f6098c0dcbd8b8ebc03c5d349ea145f8517c3b5cff1` |
| `model-00014-of-00016.safetensors` | 4,630,728,296 | `c83dacc6e0e37c4be3996ef2ade29a1703485d8d46e6d6efc052cc6f7aef9879` |
| `model-00015-of-00016.safetensors` | 4,244,835,448 | `b3f5a68852260476ab79a92fac3365dbde1766d56d18c8e392cc6da26f2840f3` |
| `model-00016-of-00016.safetensors` | 1,543,578,408 | `05f14b29768d6059a1110fd91976c408dca03d5f324c4747374053fe3bff17be` |

## 正式 8 步 LoRA

- 仓库 revision：`da0b25652ee8f8b447bf817e4bfdbb4221d521ec`。
- 文件：`SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`，`814,867,236` 字节，SHA-256 `3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708`。
- header：882 个 BF16 张量、`407,372,070` 参数、`814,744,140` payload 字节；294 组 `alpha/lora_down/lora_up`，rank 128。
- header metadata 写明 `tensor_kind=neo_hf_lora`、`lora_alpha=8.0`。下载后还要读取每个 scalar alpha 并验证实际缩放，header 本身不能证明所有 scalar 值。
- 官方 README 明确标注 8-step distilled LoRA、正式 RL 基座专用、Preview 不兼容、Apache-2.0。

## MLX 映射与内存策略

- Linear 权重保持 `[out, in]`，RMSNorm/embedding 直接映射；PyTorch Conv2d 的 `OIHW` 转成 MLX 的 `OHWI`。
- Q/K/V 使用 32 个 query heads、8 个 KV heads、128 head dim；RoPE 按官方实现拆成 t=64、h=32、w=32 三段。
- 理解专家用于 prompt prefix；生成专家 `*_mot_gen` 用于图像 token 的 flow 预测；KV prefix cache 必须保留 GQA 形状。
- 逐 shard 加载、验证 SHA 后才消费；F32 在装入参数时转 BF16并及时释放 shard，避免源模型与目标整模双驻留。
- 全 BF16 tensor payload 的理论下限为 `35,065,708,928` 字节（32.657 GiB）；1024 与 2048 的真实 active/cache/峰值仍必须实测，不能用该下限替代内存门槛。

## 下载前磁盘账本

- 基座全部文件（分片加元数据）：`50,227,451,858` 字节。
- LoRA 全部文件（权重加仓库元数据）：`814,879,393` 字节。
- 首次正式下载总量：`51,042,331,251` 字节（47.537 GiB）。
- 2026-08-22 实测目标卷可用：`563,075,592,192` 字节（524.405 GiB）。下载后预计仍可用 `512,033,260,941` 字节（476.868 GiB），未计文件系统块开销。
- 默认目标：`$SENSENOVA_MODEL_DIR`，或未设置时使用用户缓存目录下的 `sensenova-mlx/SenseNova-U1.5-8B-MoT/`。不创建 35GB 持久化 BF16 副本。

## 验证状态与止损

- 2026-08-22 的正式 layer 0 对齐、1024 基座生成、8-step LoRA 和 2048 官方分辨率桶均已通过；公开摘要见 `docs/verification/e2e-results.json`。
- 若 header/index/hash 变化、真实峰值超过 90GiB，或 swap 相对启动时增加超过 4GiB，运行会停止并保留证据。

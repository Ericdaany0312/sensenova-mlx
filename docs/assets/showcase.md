# Showcase reproduction metadata

**Evidence label:** locally measured on the listed machine / 本机实测。

- File: `showcase.png`
- Mode: formal `SenseNova-U1.5-8B-MoT` with the official 8-step LoRA
- Prompt: `一只戴着红围巾的猫，坐在雨后的上海弄堂，电影摄影，雨水倒影，细节丰富`
- Seed: `42`
- Resolution: `2048×2048`
- Steps: `8`
- CFG scale: `1.0`
- Machine: Apple M5 Max, 40-core GPU, 128 GB unified memory
- Runtime: Python 3.12, MLX 0.32.1, MLX GPU
- Time including model load: `44.084` seconds
- MLX peak: `36,922,187,276` bytes (`34.39` GiB)
- Swap delta: `0` bytes
- SHA-256: `0f2ec8739f8ecd8f4a1fd43e7de1bd54e8b318bd9e47aa195580bc5150211aec`

This record is a single-machine measurement, not a minimum-hardware or broad
performance claim. The formal base and LoRA revisions are recorded in
[`../verification/e2e-results.json`](../verification/e2e-results.json).

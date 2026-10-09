import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample
from resemble_enhance.enhancer.inference import enhance

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "enhance_test_inputs.json"
MODEL = BASE / "resemble_enhance/model_repo/enhancer_stage2"

# 后续调参只改这里。
NFE = 64
SOLVER = "midpoint"
LAMBD = 0.1
TAU = 0.5
SEED = 42

if sys.argv[1:]:
    paths = [Path(p).expanduser().resolve() for p in sys.argv[1:]]
elif CONFIG.exists():
    paths = [Path(p) for p in json.loads(CONFIG.read_text())]
else:
    raise SystemExit("首次运行请在命令后加音频路径，可以添加多个文件。")

for path in paths:
    if not path.is_file():
        raise SystemExit(f"文件不存在：{path}")
if not torch.cuda.is_available():
    raise SystemExit("未检测到 GPU")

CONFIG.write_text(json.dumps(
    [str(p) for p in paths], ensure_ascii=False, indent=2
))
out = BASE / "enhance_results" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
out.mkdir(parents=True)

def match_levels(a, b):
    """匹配整体 RMS 电平，再共同降低增益以留出峰值余量。"""
    pair = []
    for x in (a, b):
        rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        if rms < 1e-8:
            raise RuntimeError("音频接近静音，无法比较")
        pair.append(x * (10 ** (-20 / 20) / rms))
    peak = max(float(np.max(np.abs(x))) for x in pair)
    gain = min(1.0, 10 ** (-1 / 20) / peak)
    return [x * gain for x in pair]

for index, path in enumerate(paths, 1):
    print(f"\n处理：{path}", flush=True)
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    source = audio.mean(axis=1)
    if not source.size or not np.isfinite(source).all():
        raise RuntimeError(f"输入为空或含非有限数值：{path}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()
    edited, output_sr = enhance(
        torch.from_numpy(source.copy()), sr, "cuda",
        nfe=NFE, solver=SOLVER, lambd=LAMBD, tau=TAU,
        run_dir=MODEL,
    )
    edited = edited.detach().float().cpu().reshape(-1).numpy()
    if not edited.size or not np.isfinite(edited).all():
        raise RuntimeError("增强输出为空或含非有限数值")

    before = resample(
        torch.from_numpy(source.copy()), sr, output_sr
    ).numpy()
    before, after = match_levels(before, edited)

    folder = out / f"{index:02d}_{path.stem}"
    folder.mkdir()
    sf.write(folder / "before.wav", before, output_sr, subtype="PCM_24")
    sf.write(folder / "enhanced.wav", after, output_sr, subtype="PCM_24")
    info = dict(
        input=str(path), nfe=NFE, solver=SOLVER, lambd=LAMBD,
        tau=TAU, seed=SEED, output_sample_rate=output_sr,
        elapsed_seconds=round(time.perf_counter() - start, 2),
    )
    (folder / "result.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2)
    )
    print(f"已保存：{folder}", flush=True)

print(f"\n全部完成：{out}")

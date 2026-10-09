"""Compare six Resemble Enhance presets using Yu's fixed dots outputs."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
print("Resemble Enhance 参数对照开始", flush=True)

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample
from resemble_enhance.enhancer.inference import enhance, load_enhancer

PROJECT = Path("/home/yu/Workspace/resemble-enhance")
INPUT = Path("/home/yu/Workspace/dots.tts/test_outputs/emotion-rlqzvdr2")
MODEL = PROJECT / "resemble_enhance/model_repo/enhancer_stage2"
SEED = 42
PRESETS = [
    ("A_baseline", dict(nfe=64, solver="midpoint", lambd=0.1, tau=0.5)),
    ("B_nfe128", dict(nfe=128, solver="midpoint", lambd=0.1, tau=0.5)),
    ("C_rk4", dict(nfe=128, solver="rk4", lambd=0.1, tau=0.5)),
    ("D_tau02", dict(nfe=128, solver="midpoint", lambd=0.1, tau=0.2)),
    ("E_tau08", dict(nfe=128, solver="midpoint", lambd=0.1, tau=0.8)),
    ("F_denoise0", dict(nfe=128, solver="midpoint", lambd=0.0, tau=0.5)),
]


def validate(x, label):
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise RuntimeError(f"无效音频：{label}")
    if np.sqrt(np.mean(x.astype(np.float64) ** 2)) < 1e-8:
        raise RuntimeError(f"音频接近静音：{label}")


def match_all_levels(versions):
    """Match full-file RMS; apply one shared peak-headroom reduction."""
    normalized, gains = {}, {}
    for name, x in versions.items():
        rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        gain = 10 ** (-20 / 20) / rms
        normalized[name] = x * gain
        gains[name] = float(gain)
    peak = max(float(np.max(np.abs(x))) for x in normalized.values())
    common_gain = min(1.0, 10 ** (-1 / 20) / peak)
    return ({name: x * common_gain for name, x in normalized.items()},
            {name: gain * common_gain for name, gain in gains.items()})


def main():
    sources = [INPUT / "angry.wav", INPUT / "sad.wav"]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in ("hparams.yaml", "ds/G/latest", "ds/G/default/mp_rank_00_model_states.pt"):
        path = MODEL / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"本地模型文件缺失：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU")

    output = PROJECT / "quality_comparison" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    print(f"输出目录：{output}", flush=True)
    print("加载模型……", flush=True)
    model = load_enhancer(MODEL, "cuda")
    print(f"GPU：{torch.cuda.get_device_name(0)}；精度：{next(model.parameters()).dtype}", flush=True)
    if model.hp.force_gaussian_prior:
        raise RuntimeError("当前模型 force_gaussian_prior=True，tau 对照无效，请核对配置")

    for source_path in sources:
        wave, sr = sf.read(source_path, dtype="float32", always_2d=True)
        mono = wave.mean(axis=1)
        validate(mono, str(source_path))
        sample_rate = int(model.hp.wav_rate)
        before = resample(torch.from_numpy(mono.copy()), sr, sample_rate).numpy()
        versions = {"00_before": before}
        folder = output / source_path.stem
        folder.mkdir()
        report = dict(input=str(source_path), seed=SEED, sample_rate=sample_rate,
                      precision=str(next(model.parameters()).dtype),
                      status="running", presets=[], normalization="full-file RMS matched; shared peak headroom")
        report_path = folder / "parameters.json"
        for index, (name, parameters) in enumerate(PRESETS, 1):
            print(f"\n[{source_path.stem}] {index}/{len(PRESETS)} {name} {parameters}", flush=True)
            # Load once before reseeding, so each preset starts with the same RNG state.
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            started = time.perf_counter()
            edited, edited_sr = enhance(torch.from_numpy(mono.copy()), sr, "cuda",
                                        run_dir=MODEL, **parameters)
            torch.cuda.synchronize()
            edited = edited.detach().float().cpu().reshape(-1).numpy()
            validate(edited, name)
            if edited_sr != sample_rate:
                raise RuntimeError(f"输出采样率异常：{edited_sr}，预期 {sample_rate}")
            versions[name] = edited
            report["presets"].append(dict(name=name, **parameters,
                                          elapsed_seconds=round(time.perf_counter() - started, 2),
                                          duration_seconds=len(edited) / sample_rate))
            matched, gains = match_all_levels(versions)
            # Each completed preset remains available if a later preset fails.
            for file_name, audio in matched.items():
                sf.write(folder / f"{file_name}.wav", audio, sample_rate, subtype="PCM_24")
            report["applied_gains"] = gains
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"本组完成：{report['presets'][-1]['elapsed_seconds']} 秒", flush=True)
        report["status"] = "completed"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"试听文件已保存：{folder}", flush=True)
    print(f"\n全部完成：{output}", flush=True)


if __name__ == "__main__":
    main()

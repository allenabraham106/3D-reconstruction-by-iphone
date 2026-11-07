#!/usr/bin/env python3
"""Recreate a gap in a walk from the frames before and after it, and score it.

A stretch of the video is cut out (the real frames are kept as ground truth). The gap is
then filled from only the last frame before it and the first frame after it, two ways:

  flow  classic optical-flow interpolation (OpenCV DIS flow, both directions, blended),
        no learned model: the baseline
  ltx   LTX-Video 0.9.5 (2B video diffusion, Lightricks), conditioned on the two keyframes
        (diffusers LTXConditionPipeline), so it can invent what the camera saw in between

Each generated frame is compared with the real one it replaces: PSNR, SSIM, LPIPS (lower
is better). Writes <out>/<method>/frame_*.png, <out>/metrics.json and <out>/compare.mp4
(real | flow | ltx, with labels).

    python gpu/gap_fill.py out_vipe/default/vipe/rgb/IMG_6075.mp4 --start 300 --length 24 out_gap/IMG_6075
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def read_frames(video, start, n):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[..., ::-1].copy())  # RGB
    return out


def flow_interp(a, b, ts):
    """Bidirectional DIS optical flow; each in-between frame warps both ends and blends."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fab = dis.calc(ga, gb, None)   # a -> b
    fba = dis.calc(gb, ga, None)   # b -> a
    h, w = ga.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = []
    for t in ts:
        # backward warps: sample a at x - t*f_ab, b at x - (1-t)*f_ba (flows assumed smooth)
        wa = cv2.remap(a, xx - t * fab[..., 0], yy - t * fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(b, xx - (1 - t) * fba[..., 0], yy - (1 - t) * fba[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        out.append(((1 - t) * wa.astype(np.float32) + t * wb.astype(np.float32)).clip(0, 255).astype(np.uint8))
    return out


def ltx_interp(a, b, n_between, prompt, steps, seed, size=(768, 448), model="Lightricks/LTX-Video-0.9.5",
               gen_frames=None, src_fps=10.0, log=print):
    """Generate n_between frames between a and b with LTX-Video keyframe conditioning.
    The clip is generated with gen_frames frames (8k+1) spanning a..b and resampled in time."""
    import torch
    from PIL import Image
    from diffusers import LTXConditionPipeline
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

    W, H = size
    N = gen_frames or (8 * int(np.ceil((n_between + 1) / 8)) + 1)
    pipe = LTXConditionPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    # the T5-XXL text encoder (9.5 GB) runs once, on the CPU; only the 2B video model and the
    # VAE go to the GPU (fits next to other GPU jobs)
    neg = "worst quality, inconsistent motion, blurry, jittery, distorted, warped"
    with torch.no_grad():
        pe, pm, ne, nm = pipe.encode_prompt(prompt, neg, do_classifier_free_guidance=True,
                                            device=torch.device("cpu"), dtype=torch.bfloat16)
    pe, pm, ne, nm = (x.to("cuda") for x in (pe, pm, ne, nm))
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    to_pil = lambda x: Image.fromarray(cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA))
    conds = [LTXVideoCondition(image=to_pil(a), frame_index=0),
             LTXVideoCondition(image=to_pil(b), frame_index=N - 1)]
    t0 = time.time()
    out = pipe(conditions=conds, prompt_embeds=pe, prompt_attention_mask=pm,
               negative_prompt_embeds=ne, negative_prompt_attention_mask=nm,
               width=W, height=H, num_frames=N, num_inference_steps=steps, guidance_scale=3.0,
               frame_rate=max(1, round((N - 1) / ((n_between + 1) / src_fps))),  # real playback speed
               generator=torch.Generator("cuda").manual_seed(seed), output_type="np").frames[0]
    log(f"[ltx] {N} frames at {W}x{H} in {time.time()-t0:.0f}s, peak VRAM "
        f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    clip = (np.asarray(out) * 255).clip(0, 255).astype(np.uint8)  # (N, H, W, 3)
    # resample: generated frame j sits at time j/(N-1) of the gap; we need t = i/(n_between+1)
    res = []
    for i in range(1, n_between + 1):
        x = i / (n_between + 1) * (N - 1)
        j0 = int(np.floor(x)); j1 = min(j0 + 1, N - 1); f = x - j0
        res.append(((1 - f) * clip[j0].astype(np.float32) + f * clip[j1].astype(np.float32)).astype(np.uint8))
    del pipe
    torch.cuda.empty_cache()
    return res, clip


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    sa, sb, sab = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (sa + sb + c2))).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, help="the walk (e.g. ViPE's rgb mp4, 10 fps)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=int, required=True, help="first missing frame")
    ap.add_argument("--length", type=int, default=24, help="number of missing frames")
    ap.add_argument("--methods", default="flow,ltx")
    ap.add_argument("--prompt", default="A smooth handheld phone video walking slowly through an indoor room, "
                                        "steady camera motion, realistic, sharp, natural indoor lighting.")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=10, help="frame rate of the input video")
    a = ap.parse_args()
    import lpips
    import torch

    frames = read_frames(a.video, a.start - 1, a.length + 2)  # keyframe, gap, keyframe
    before, real, after = frames[0], frames[1:-1], frames[-1]
    h, w = before.shape[:2]
    ts = [(i + 1) / (a.length + 1) for i in range(a.length)]
    a.out.mkdir(parents=True, exist_ok=True)
    results = {"video": str(a.video), "start": a.start, "length": a.length, "resolution": [w, h]}
    gens = {}
    for m in a.methods.split(","):
        t0 = time.time()
        if m == "flow":
            g = flow_interp(before, after, ts)
        elif m == "ltx":
            g, clip = ltx_interp(before, after, a.length, a.prompt, a.steps, a.seed, src_fps=a.fps)
            g = [cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC) for x in g]
        elif (a.out / m).is_dir():  # frames made by another tool, e.g. gpu/gap_splat.py
            g = [cv2.imread(str(a.out / m / f"frame_{a.start + i:05d}.png"))[..., ::-1].copy()
                 for i in range(a.length)]
        else:
            raise SystemExit(f"unknown method {m} (and no {a.out / m} folder)")
        results.setdefault("seconds", {})[m] = round(time.time() - t0, 1)
        gens[m] = g
        d = a.out / m
        if m in ("flow", "ltx"):
            d.mkdir(exist_ok=True)
            for i, x in enumerate(g):
                cv2.imwrite(str(d / f"frame_{a.start + i:05d}.png"), x[..., ::-1])

    # metrics against the real frames
    net = lpips.LPIPS(net="alex", verbose=False).cuda()
    tt = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
    for m, g in gens.items():
        ps, ss, lp = [], [], []
        for x, r in zip(g, real):
            mse = ((x.astype(np.float64) - r) ** 2).mean()
            ps.append(10 * np.log10(255 ** 2 / max(mse, 1e-10)))
            ss.append(ssim(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)))
            with torch.no_grad():
                lp.append(float(net(tt(x), tt(r))))
        results[m] = {"psnr": round(float(np.mean(ps)), 2), "ssim": round(float(np.mean(ss)), 4),
                      "lpips": round(float(np.mean(lp)), 4),
                      "psnr_mid": round(float(ps[len(ps) // 2]), 2), "lpips_mid": round(float(lp[len(lp) // 2]), 4),
                      "per_frame_lpips": [round(v, 4) for v in lp]}
    (a.out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k not in gens} |
                     {m: {k: v for k, v in results[m].items() if k != "per_frame_lpips"} for m in gens}, indent=1))

    # side-by-side video: real | each method, labelled, slowed to 5 fps
    cols = ["real"] + list(gens)
    tw = 640
    th = int(h * tw / w)
    vw = cv2.VideoWriter(str(a.out / "compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (tw * len(cols), th))
    for i in range(a.length):
        tiles = []
        for c in cols:
            x = real[i] if c == "real" else gens[c][i]
            x = cv2.resize(x, (tw, th), interpolation=cv2.INTER_AREA)[..., ::-1].copy()
            label = {"real": "REAL (hidden)", "flow": "OPTICAL FLOW", "ltx": "LTX-VIDEO",
                     "splat": "3D SPLAT", "splat_truepose": "3D SPLAT, TRUE PATH"}.get(c, c)
            cv2.putText(x, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(x, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(x)
        vw.write(np.concatenate(tiles, 1))
    vw.release()
    mid = a.length // 2
    sheet = [cv2.resize(real[mid], (tw, th))] + [cv2.resize(gens[c][mid], (tw, th)) for c in gens]
    cv2.imwrite(str(a.out / "middle_frame.jpg"), np.concatenate(sheet, 1)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()

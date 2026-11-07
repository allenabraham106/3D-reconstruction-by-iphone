#!/usr/bin/env python3
"""Gaussian splat of an iPhone scan from ViPE output, with gsplat (PyPI, v1.4.0).

Self-contained trainer (no gsplat example code, no COLMAP step): reads ViPE's poses,
intrinsics and metric depth directly and trains at the video's full resolution.

    python gpu/splat_train.py out_vipe/default/vipe IMG_6075 path/to/IMG_6075.MOV out_splat/IMG_6075

What it does, for a crisp result on a hand-held walk:
  * frames: the original .MOV re-decoded at ViPE's frame times, HDR tone-mapped like the rest
    of the pipeline, at full resolution (1080p) -- ViPE itself ran at 720p; intrinsics are
    scaled to match (poses don't depend on resolution);
  * init: Gaussians back-projected from ViPE's reliable metric depth (not sparse SfM points);
  * loss: 0.8 L1 + 0.2 D-SSIM, plus an inverse-depth L1 against ViPE depth (decays over
    training) so geometry stays solid when the camera leaves the recorded path;
  * densification: gsplat's MCMC strategy (bounded Gaussian count, sharper fine detail);
  * exposure: a learned per-frame colour affine soaks up the iPhone's auto-exposure, so the
    splat itself isn't blurred by brightness changes;
  * evaluation: every 8th frame is held out; PSNR / SSIM are reported on those (frames the
    splat never saw), with the colour affine of the nearest training frame.

Outputs in <out>/: splat.ply (standard 3DGS layout: SuperSplat, gsplat viewers, ...),
stats.json, renders/ (held-out frame | render), flythrough.mp4, ckpt.pt.
"""
import argparse
import json
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu"))
from vipe_to_pipeline import read_exr_depth, reliable_mask  # noqa: E402

TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
           "zscale=t=bt709:m=bt709:r=tv,format=rgb24")


# ----------------------------------------------------------------------------- data
def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,color_transfer:stream_side_data=rotation",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


def decode_frames(video, fps, max_side, n_expected):
    """Decode the video at `fps` (same frame times as the 10 fps ViPE input), tone-mapped,
    auto-rotated, longest side <= max_side. Returns uint8 (N, H, W, 3) RGB."""
    s = probe(video)
    hdr = s.get("color_transfer") in ("arib-std-b67", "smpte2084")
    vf = [f"fps={fps}"] + ([TONEMAP] if hdr else ["format=rgb24"])
    vf.append(f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'")
    # one frame through the same filters (rotation included) to learn the output size
    import cv2
    png = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", ",".join(vf), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    h, w = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-vf", ",".join(vf),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    frames = []
    while True:
        buf = p.stdout.read(h * w * 3)
        if len(buf) < h * w * 3:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
    p.wait()
    if n_expected and abs(len(frames) - n_expected) > 2:
        print(f"[splat] warning: decoded {len(frames)} frames, ViPE has {n_expected}", flush=True)
    return np.stack(frames)


def load_vipe(vdir, name):
    pose = np.load(vdir / "pose" / f"{name}.npz")
    intr = np.load(vdir / "intrinsics" / f"{name}.npz")
    c2w = dict(zip(pose["inds"].tolist(), pose["data"].astype(np.float64)))
    K4 = dict(zip(intr["inds"].tolist(), intr["data"].astype(np.float64)))
    depth = {}
    with zipfile.ZipFile(vdir / "depth" / f"{name}.zip") as z:
        for fn in z.namelist():
            depth[int(fn.split(".")[0])] = fn
    return c2w, K4, depth


# ----------------------------------------------------------------------------- model
def ssim(a, b, win=11, sigma=1.5):
    """SSIM over (B, 3, H, W) in [0, 1], Gaussian window (as in 3DGS)."""
    g = torch.exp(-((torch.arange(win, device=a.device) - win // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).float()
    k = (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda x: Fnn.conv2d(x, k, padding=win // 2, groups=3)
    mu_a, mu_b = f(a), f(b)
    s_a, s_b, s_ab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2))
    return m.mean()


def knn_scale(xyz, k=4):
    """Mean distance to the k-1 nearest neighbours (initial Gaussian size)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xyz).query(xyz, k=k, workers=-1)
    return np.sqrt((d[:, 1:] ** 2).mean(1)).clip(1e-4, None)


SH_C0 = 0.28209479177387814


def save_ply(path, params):
    """Standard 3DGS .ply (x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*)."""
    means = params["means"].detach().cpu().numpy()
    sh0 = params["sh0"].detach().cpu().numpy().reshape(len(means), -1)                 # (N, 3)
    shN = params["shN"].detach().transpose(1, 2).cpu().numpy().reshape(len(means), -1)  # channel-major
    opac = params["opacities"].detach().cpu().numpy()[:, None]
    scales = params["scales"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()
    cols = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(sh0.shape[1])] + \
           [f"f_rest_{i}" for i in range(shN.shape[1])] + ["opacity"] + \
           [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.concatenate([means, np.zeros_like(means), sh0, shN, opac, scales, quats], 1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(data)}\n" + "".join(f"property float {c}\n" for c in cols) +
                 "end_header\n").encode())
        f.write(data.tobytes())


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vipe_dir", type=Path)
    ap.add_argument("name")
    ap.add_argument("video", type=Path, help="the original video (full resolution)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=float, default=10, help="must match the ViPE input (run_gpu.sh VIPE_FPS)")
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--cap", type=int, default=3_000_000, help="max Gaussians (MCMC)")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--depth-weight", type=float, default=0.2)
    ap.add_argument("--holdout", type=int, default=8, help="every k-th frame is held out for eval")
    ap.add_argument("--init-points", type=int, default=600_000)
    ap.add_argument("--video-frames", type=int, default=600, help="fly-through length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=None,
                    help="A:B = leave video frames A..B-1 out of training and eval (gap experiments)")
    ap.add_argument("--pose-opt", type=int, default=0, help="1: refine each training camera's pose (A/B on the office: -0.7 dB)")
    ap.add_argument("--mask-people", type=int, default=1,
                    help="1: leave pixels ViPE segmented as people out of training (people move: ghosts)")
    ap.add_argument("--drop-blurry", type=float, default=0.0,
                    help="leave the blurriest share of frames (motion blur) out of training")
    a = ap.parse_args()

    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    c2w, K4, depth_names = load_vipe(a.vipe_dir, a.name)
    ids = sorted(set(c2w) & set(K4))
    imgs = decode_frames(a.video, a.fps, a.max_side, len(ids))
    ids = [i for i in ids if i < len(imgs)]
    H, W = imgs.shape[1:3]
    # ViPE resolution from its depth maps
    with zipfile.ZipFile(a.vipe_dir / "depth" / f"{a.name}.zip") as z:
        d0 = read_exr_depth(z.read(depth_names[ids[0]]))
    sx, sy = W / d0.shape[1], H / d0.shape[0]
    print(f"[splat] {len(ids)} frames at {W}x{H} (ViPE ran at {d0.shape[1]}x{d0.shape[0]}), "
          f"decoded in {time.time()-t0:.0f}s", flush=True)

    Ks, viewmats, cams_C = [], [], []
    for i in ids:
        fx, fy, cx, cy = K4[i][:4]
        Ks.append([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
        viewmats.append(np.linalg.inv(c2w[i]))
        cams_C.append(c2w[i][:3, 3])
    Ks = torch.tensor(np.array(Ks), dtype=torch.float32, device=dev)
    viewmats = torch.tensor(np.array(viewmats), dtype=torch.float32, device=dev)
    cams_C = np.array(cams_C)
    test = [k for k in range(len(ids)) if k % a.holdout == a.holdout // 2]
    train = [k for k in range(len(ids)) if k % a.holdout != a.holdout // 2]
    if a.exclude:
        e0, e1 = (int(x) for x in a.exclude.split(":"))
        gap = {k for k, i in enumerate(ids) if e0 <= i < e1}
        train = [k for k in train if k not in gap]
        test = [k for k in test if k not in gap]
        print(f"[splat] frames {e0}..{e1 - 1} left out entirely ({len(gap)} frames)", flush=True)
    if a.drop_blurry > 0:  # variance of the Laplacian, relative to the frame's neighbours
        import cv2
        sharp = np.array([cv2.Laplacian(cv2.cvtColor(cv2.resize(im, (W // 3, H // 3)), cv2.COLOR_RGB2GRAY),
                                        cv2.CV_32F).var() for im in imgs])
        from scipy.ndimage import median_filter
        rel = sharp / np.maximum(median_filter(sharp, size=15, mode="nearest"), 1e-6)
        cut = np.quantile(rel[train], a.drop_blurry)
        dropped = [k for k in train if rel[k] < cut]
        train = [k for k in train if rel[k] >= cut]
        print(f"[splat] left {len(dropped)} motion-blurred frames out of training", flush=True)

    # depth (ViPE resolution, reliable pixels only), upsampled on the fly to training size
    depths = []
    with zipfile.ZipFile(a.vipe_dir / "depth" / f"{a.name}.zip") as z:
        for i in ids:
            D = read_exr_depth(z.read(depth_names[i]))
            m = reliable_mask(D) & (D < 12)
            depths.append(np.where(m, D, 0).astype(np.float16))
    depths = torch.from_numpy(np.stack(depths))  # CPU, (N, h, w)
    # static-scene mask: everything except (dilated) people, from ViPE's instance masks
    static = None
    mzip, mtxt = a.vipe_dir / "mask" / f"{a.name}.zip", a.vipe_dir / "mask" / f"{a.name}.txt"
    if mzip.exists() and mtxt.exists():  # loaded always: eval reports static-scene PSNR too
        import cv2
        person = [int(l.split(":")[0]) for l in mtxt.read_text().splitlines() if l.split(":")[1].strip() == "person"]
        static = np.ones(depths.shape, bool)
        k15 = np.ones((15, 15), np.uint8)
        with zipfile.ZipFile(mzip) as z:
            names = set(z.namelist())
            for j, i in enumerate(ids):
                fn = f"{i:05d}.png"
                if fn in names:
                    m = cv2.imdecode(np.frombuffer(z.read(fn), np.uint8), cv2.IMREAD_UNCHANGED)
                    static[j] = cv2.dilate(np.isin(m, person).astype(np.uint8), k15) == 0
        if a.mask_people:
            depths[torch.from_numpy(~static)] = 0
        static = torch.from_numpy(static)
        print(f"[splat] people masked out: {100 * (1 - static.float().mean()):.1f}% of pixels", flush=True)

    def static_mask(k):  # (H, W) bool at training resolution, or None
        if static is None:
            return None
        return Fnn.interpolate(static[k][None, None].float().to(dev), size=(H, W), mode="nearest")[0, 0] > 0.5

    # init: back-project reliable depth from training frames, colours from the full-res frame
    rng = np.random.default_rng(a.seed)
    per = max(1, a.init_points // len(train))
    P, C = [], []
    for k in train:
        D = depths[k].float().numpy()
        ys, xs = np.nonzero(D > 0)
        if len(ys) == 0:
            continue
        sel = rng.choice(len(ys), min(per, len(ys)), replace=False)
        ys, xs = ys[sel], xs[sel]
        z = D[ys, xs]
        fx, fy, cx, cy = K4[ids[k]][:4]
        Xc = np.stack([(xs + 0.5 - cx) / fx * z, (ys + 0.5 - cy) / fy * z, z], 1)
        T = c2w[ids[k]]
        P.append(Xc @ T[:3, :3].T + T[:3, 3])
        C.append(imgs[k][np.minimum((ys * sy).astype(int), H - 1), np.minimum((xs * sx).astype(int), W - 1)])
    P = np.concatenate(P).astype(np.float32)
    C = np.concatenate(C).astype(np.float32) / 255
    scene_scale = float(np.linalg.norm(cams_C - cams_C.mean(0), axis=1).max()) * 1.1
    print(f"[splat] init {len(P):,} Gaussians from ViPE depth, scene scale {scene_scale:.2f} m", flush=True)

    n = len(P)
    dist = knn_scale(P)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(P, device=dev)),
        "scales": torch.nn.Parameter(torch.log(torch.tensor(dist, device=dev, dtype=torch.float32))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=dev))),
        "sh0": torch.nn.Parameter(((torch.tensor(C, device=dev) - 0.5) / SH_C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, (a.sh_degree + 1) ** 2 - 1, 3, device=dev)),
    })
    # means: half the 3DGS default (1.6e-4 x scene scale); the default diverged on the apartments
    lrs = {"means": 0.8e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15)
            for k in params}
    means_decay = 0.01 ** (1 / a.steps)
    # per-frame exposure / white-balance: rgb' = A rgb + b, starts at identity
    expo_A = torch.nn.Parameter(torch.eye(3, device=dev).repeat(len(ids), 1, 1))
    expo_b = torch.nn.Parameter(torch.zeros(len(ids), 3, device=dev))
    expo_opt = torch.optim.Adam([expo_A, expo_b], lr=2.5e-4)
    # per-frame pose refinement (rotation vector + translation, applied in camera space)
    pose_d = torch.nn.Parameter(torch.zeros(len(ids), 6, device=dev))
    pose_opt = torch.optim.Adam([pose_d], lr=1e-5, weight_decay=1e-6)

    def refined(k_list):
        vm = viewmats[k_list]
        if not a.pose_opt:
            return vm
        d = pose_d[k_list]
        w, t = d[:, :3], d[:, 3:]
        th = w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        kx = w / th
        Kx = torch.zeros(len(k_list), 3, 3, device=dev)
        Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -kx[:, 2], kx[:, 1], -kx[:, 0]
        Kx = Kx - Kx.transpose(1, 2)
        I = torch.eye(3, device=dev)[None]
        R = I + torch.sin(th)[..., None] * Kx + (1 - torch.cos(th))[..., None] * (Kx @ Kx)
        D = torch.eye(4, device=dev).repeat(len(k_list), 1, 1)
        D[:, :3, :3], D[:, :3, 3] = R, t
        return D @ vm

    strategy = MCMCStrategy(cap_max=a.cap, refine_stop_iter=int(a.steps * 0.83), verbose=False)
    strategy.check_sanity(params, opts)
    sstate = strategy.initialize_state()
    imgs_t = torch.from_numpy(imgs).pin_memory() if torch.cuda.is_available() else torch.from_numpy(imgs)

    def render(k_list, sh_deg, expo_from=None):
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        out, alpha, info = rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            colors, refined(k_list) if expo_from is None else viewmats[k_list], Ks[k_list], W, H, sh_degree=sh_deg, render_mode="RGB+ED",
            packed=False, near_plane=0.05)
        rgb, dep = out[..., :3], out[..., 3]
        e = k_list if expo_from is None else expo_from
        rgb = torch.einsum("bij,bhwj->bhwi", expo_A[e], rgb) + expo_b[e][:, None, None, :]
        return rgb, dep, alpha, info

    print(f"[splat] training {a.steps} steps on {len(train)} frames ({len(test)} held out)", flush=True)
    t_train = time.time()
    for step in range(a.steps):
        k = int(rng.choice(train))
        gt = imgs_t[k].to(dev, non_blocking=True).float()[None] / 255
        sh_deg = min(a.sh_degree, step // 1000)
        rgb, dep, alpha, info = render([k], sh_deg)
        sm = static_mask(k) if a.mask_people else None
        if sm is not None:  # people pixels: target = own render, so they add no loss or gradient
            gt = torch.where(sm[None, ..., None], gt, rgb.detach())
        l1 = (rgb - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        if a.depth_weight > 0:
            D = depths[k].to(dev).float()[None, None]
            D = Fnn.interpolate(D, size=(H, W), mode="nearest")[0, 0]
            # only where the render is mostly opaque: on near-empty pixels the expected depth
            # is ~0, its inverse explodes and swamps the photo loss (big rooms with windows)
            m = (D > 0) & (alpha[0, ..., 0].detach() > 0.5)
            if m.any():
                w = a.depth_weight * (0.01 ** (step / a.steps))  # strong early, fades out
                err = (1 / dep[0][m].clamp(min=0.2) - 1 / D[m].clamp(min=0.2)).abs().clamp(max=1.0)
                loss = loss + w * err.mean()
        # keep the exposure correction near identity (it can otherwise run away on bright windows)
        loss = loss + 0.01 * ((expo_A[k] - torch.eye(3, device=dev)) ** 2).sum() + 0.01 * (expo_b[k] ** 2).sum()
        # MCMC regularisers (as in the paper / gsplat example)
        loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        expo_opt.step()
        expo_opt.zero_grad(set_to_none=True)
        if a.pose_opt:
            pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
        for g in opts["means"].param_groups:
            g["lr"] *= means_decay
        strategy.step_post_backward(params, opts, sstate, step, info, lr=opts["means"].param_groups[0]["lr"])
        if step % 1000 == 0 or step == a.steps - 1:
            psnr = -10 * math.log10(max(((rgb - gt) ** 2).mean().item(), 1e-10))
            print(f"[splat] step {step:6d}  loss {loss.item():.4f}  train-PSNR {psnr:5.2f}  "
                  f"gaussians {len(params['means']):,}  {time.time()-t_train:.0f}s", flush=True)
    train_s = time.time() - t_train

    # evaluation on held-out frames (exposure from the nearest training frame)
    import cv2
    (a.out / "renders").mkdir(exist_ok=True)
    psnrs, ssims, psnrs_static = [], [], []
    with torch.no_grad():
        for j, k in enumerate(test):
            near = min(train, key=lambda t: abs(t - k))
            rgb, _, _, _ = render([k], a.sh_degree, expo_from=[near])
            rgb = rgb.clamp(0, 1)
            gt = imgs_t[k].to(dev).float()[None] / 255
            psnrs.append(-10 * math.log10(((rgb - gt) ** 2).mean().item()))
            sm = static_mask(k)
            if sm is not None:
                psnrs_static.append(-10 * math.log10(((rgb - gt) ** 2)[0][sm].mean().item()))
            ssims.append(ssim(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())
            if j % max(1, len(test) // 8) == 0:
                pair = np.concatenate([(gt[0].cpu().numpy() * 255), (rgb[0].cpu().numpy() * 255)], 1)
                cv2.imwrite(str(a.out / "renders" / f"test_{k:04d}.jpg"), pair[..., ::-1].astype(np.uint8),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
    stats = {"frames": len(ids), "train_frames": len(train), "test_frames": len(test),
             "resolution": [W, H], "steps": a.steps, "gaussians": len(params["means"]),
             "psnr_test": round(float(np.mean(psnrs)), 2), "ssim_test": round(float(np.mean(ssims)), 4),
             "psnr_test_static": round(float(np.mean(psnrs_static)), 2) if psnrs_static else None,
             "pose_opt": bool(a.pose_opt), "drop_blurry": a.drop_blurry, "mask_people": bool(a.mask_people),
             "train_minutes": round(train_s / 60, 1), "total_minutes": round((time.time() - t0) / 60, 1),
             "gpu": torch.cuda.get_device_name(0),
             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    print("[splat] " + json.dumps(stats), flush=True)
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    save_ply(a.out / "splat.ply", params)
    torch.save({"params": {k: v.detach().cpu() for k, v in params.items()}, "Ks": Ks.cpu(),
                "viewmats": viewmats.cpu(), "W": W, "H": H, "sh_degree": a.sh_degree,
                "expo_A": expo_A.detach().cpu(), "expo_b": expo_b.detach().cpu(), "train": train,
                "pose_d": pose_d.detach().cpu(), "frame_ids": ids}, a.out / "ckpt.pt")

    # fly-through: the recorded path, smoothed, rendered at full resolution
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation, Slerp
    c2ws = np.array([c2w[i] for i in ids])
    t = np.linspace(0, len(ids) - 1, a.video_frames)
    pos = gaussian_filter1d(c2ws[:, :3, 3], sigma=4, axis=0, mode="nearest")
    pos = np.stack([np.interp(t, np.arange(len(ids)), pos[:, d]) for d in range(3)], 1)
    rots = Rotation.from_matrix(c2ws[:, :3, :3])
    q = gaussian_filter1d(rots.as_quat() * np.sign(rots.as_quat()[:, 3:4] + 1e-9), sigma=4, axis=0, mode="nearest")
    rots_s = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))
    rot_t = Slerp(np.arange(len(ids)), rots_s)(t)
    K = Ks[len(ids) // 2:len(ids) // 2 + 1]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                          "-pix_fmt", "yuv420p", str(a.out / "flythrough.mp4")], stdin=subprocess.PIPE)
    mid = train[len(train) // 2]
    with torch.no_grad():
        for f in range(a.video_frames):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = rot_t[f].as_matrix(), pos[f]
            vm = torch.tensor(np.linalg.inv(T), dtype=torch.float32, device=dev)[None]
            colors = torch.cat([params["sh0"], params["shN"]], 1)
            out, _, _ = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                      torch.sigmoid(params["opacities"]), colors, vm, K, W, H,
                                      sh_degree=a.sh_degree, near_plane=0.05)
            near = train[min(range(len(train)), key=lambda j: abs(train[j] - t[f]))] if train else mid
            rgb = torch.einsum("ij,hwj->hwi", expo_A[near], out[0, ..., :3]) + expo_b[near]
            p.stdin.write((rgb.clamp(0, 1) * 255).byte().cpu().numpy().tobytes())
    p.stdin.close()
    p.wait()
    print(f"[splat] done in {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

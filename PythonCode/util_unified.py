import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import imageio.v2 as imageio
from tifffile import imread as tifread, imwrite as tifwrite
from pytorch_wavelets import DWTForward, DWTInverse
from pytorch_msssim import ssim as ssim_pt

# --------------------------
# Network
# --------------------------
class ResBlock(nn.Module):
    def __init__(self, ch: int, act_slope: float = 0.2):
        super().__init__()
        self.act = nn.LeakyReLU(act_slope, inplace=True)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class NetworkRes(nn.Module):
    def __init__(self, n_chan: int, chan_embed: int = 48, n_blocks: int = 1):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.head = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(chan_embed) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(chan_embed, n_chan, 3, padding=1)

    def forward(self, x):
        f = self.act(self.head(x))
        f = self.body(f)
        noise = self.tail(f)
        return noise


# --------------------------
# Wavelet
# --------------------------
WAVE_NAME = "db2"
PAD_MODE  = "symmetric"

_dwt = None
_idwt = None
_wavelet_device = None

def get_wavelet(device: torch.device):
    """Create wavelet modules ONCE per device."""
    global _dwt, _idwt, _wavelet_device
    if (_dwt is None) or (_wavelet_device != device):
        _dwt = DWTForward(J=1, wave=WAVE_NAME, mode=PAD_MODE).to(device)
        _idwt = DWTInverse(wave=WAVE_NAME, mode=PAD_MODE).to(device)
        _wavelet_device = device
    return _dwt, _idwt


# --------------------------
# IO helpers
# --------------------------
def _is_color_lastdim(arr: np.ndarray) -> bool:
    return (arr.ndim >= 3) and (arr.shape[-1] in (3, 4))

def _drop_alpha(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 3 and arr.shape[-1] == 4:
        return arr[..., :3]
    return arr

def _normalize_to_01(arr: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Normalize to [0,1] float32 with robust float handling.

    Returns:
      arr01: float32 in [0,1]
      nm: {orig_dtype, norm_mode, scale, min/max (optional)}
    """
    orig_dtype = arr.dtype
    nm: Dict = {"orig_dtype": orig_dtype, "norm_mode": None}

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = arr.astype(np.float32) / float(info.max)
        nm["norm_mode"] = "int_div_max"
        nm["scale"] = float(info.max)
        return np.clip(out, 0.0, 1.0), nm

    out = arr.astype(np.float32)
    mn, mx = float(out.min()), float(out.max())

    if mn >= 0.0 and mx <= 1.0 + 1e-6:
        nm["norm_mode"] = "float_01"
        nm["scale"] = 1.0
        return np.clip(out, 0.0, 1.0), nm

    if mn >= 0.0 and mx <= 255.0 + 1e-6:
        nm["norm_mode"] = "float_div_255"
        nm["scale"] = 255.0
        return np.clip(out / 255.0, 0.0, 1.0), nm

    if mn >= 0.0 and mx <= 65535.0 + 1e-6:
        nm["norm_mode"] = "float_div_65535"
        nm["scale"] = 65535.0
        return np.clip(out / 65535.0, 0.0, 1.0), nm

    nm["norm_mode"] = "float_minmax"
    nm["min"] = mn
    nm["max"] = mx
    if mx > mn:
        out = (out - mn) / (mx - mn + 1e-8)
    else:
        out = np.zeros_like(out, dtype=np.float32)
    nm["scale"] = 1.0
    return np.clip(out, 0.0, 1.0), nm

def read_image_any(path: str, as_gray: bool = True, drop_alpha: bool = True) -> Tuple[torch.Tensor, Dict]:
    """
    Supports:
      - PNG/JPG/BMP/etc via imageio
      - TIFF/TIF via tifffile (including multi-page stacks)

    Returns:
      tensor: (N, C, H, W) float32 in [0,1]
      meta: dict for writing "like input"
        {
          ext, is_tiff, pages, channels,
          orig_dtype, norm_mode, scale, min/max(optional),
          was_color, as_gray
        }
    """
    ext = os.path.splitext(path)[1].lower()
    is_tiff = ext in (".tif", ".tiff")

    if is_tiff:
        arr = tifread(path)  # (H,W)/(H,W,C)/(N,H,W)/(N,H,W,C)
    else:
        arr = imageio.imread(path)  # (H,W) or (H,W,C)

    if drop_alpha:
        arr = _drop_alpha(arr)

    arr01, nm = _normalize_to_01(arr)

    # reshape to stack
    if arr01.ndim == 2:
        arr_n = arr01[None, ...]          # (1,H,W)
        was_color = False
    elif arr01.ndim == 3:
        if _is_color_lastdim(arr01):      # (H,W,C)
            arr_n = arr01[None, ...]      # (1,H,W,C)
            was_color = True
        else:                             # (N,H,W)
            arr_n = arr01
            was_color = False
    elif arr01.ndim == 4:
        arr_n = arr01
        was_color = _is_color_lastdim(arr01)
    else:
        raise ValueError(f"Unsupported image ndim={arr01.ndim} for {path}")

    if was_color:
        t = torch.from_numpy(arr_n.astype(np.float32)).permute(0, 3, 1, 2)  # (N,C,H,W)
        channels = int(t.shape[1])
    else:
        t = torch.from_numpy(arr_n.astype(np.float32))[:, None, :, :]       # (N,1,H,W)
        channels = 1

    if as_gray and channels > 1:
        r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
        t = 0.2989 * r + 0.5870 * g + 0.1140 * b
        channels = 1

    meta: Dict = {
        "path": path,
        "ext": ext,
        "is_tiff": bool(is_tiff),
        "pages": int(t.shape[0]),
        "channels": int(channels),
        "as_gray": bool(as_gray),
        "was_color": bool(was_color),
        "orig_dtype": nm.get("orig_dtype"),
        "norm_mode": nm.get("norm_mode"),
        "scale": float(nm.get("scale", 1.0)),
    }
    if "min" in nm: meta["min"] = float(nm["min"])
    if "max" in nm: meta["max"] = float(nm["max"])
    return t.contiguous(), meta

def _invert_norm_from_01(t01: torch.Tensor, meta: Dict) -> np.ndarray:
    """
    t01: (N,C,H,W) float in [0,1]
    Return numpy array in original dtype and original value domain (as best as possible).
    Shapes returned:
      - gray: (H,W) or (N,H,W)
      - color: (H,W,C) or (N,H,W,C)
    """
    t01 = t01.detach().float().clamp(0, 1).cpu()
    arr = t01.numpy()
    n, c, h, w = arr.shape

    dtype = meta.get("orig_dtype", np.uint16)
    norm_mode = meta.get("norm_mode", "int_div_max")
    scale = float(meta.get("scale", 1.0))
    mn = float(meta.get("min", 0.0))
    mx = float(meta.get("max", 1.0))

    def inv_float(x01: np.ndarray) -> np.ndarray:
        if norm_mode in ("int_div_max", "float_div_255", "float_div_65535"):
            return x01 * scale
        if norm_mode == "float_01":
            return x01
        if norm_mode == "float_minmax":
            return x01 * (mx - mn) + mn
        # fallback
        return x01 * scale

    if c == 1:
        x = inv_float(arr[:, 0, :, :])  # (N,H,W)
        if np.issubdtype(dtype, np.integer):
            x = np.round(x)
            x = np.clip(x, 0, scale)
            x = x.astype(dtype)
        else:
            x = x.astype(dtype)
        return x[0] if n == 1 else x

    # color: (N,H,W,C)
    x = np.transpose(arr, (0, 2, 3, 1))
    x = inv_float(x)
    if np.issubdtype(dtype, np.integer):
        x = np.round(x)
        x = np.clip(x, 0, scale)
        x = x.astype(dtype)
    else:
        x = x.astype(dtype)
    return x[0] if n == 1 else x

def write_image_like(out_path: str, t01: torch.Tensor, meta: Dict):
    """
    Write output with:
      - same extension as out_path
      - same dtype/bit-depth as input (meta['orig_dtype'])
      - multi-page TIFF supported
    """
    ext = os.path.splitext(out_path)[1].lower()
    if ext == "":
        ext = meta.get("ext", "").lower()

    out_np = _invert_norm_from_01(t01, meta)

    if ext in (".tif", ".tiff"):
        photometric = "minisblack" if int(meta.get("channels", 1)) == 1 else "rgb"
        tifwrite(out_path, out_np, photometric=photometric)
        return

    # JPG cannot store uint16/float reliably -> fallback to uint8
    if ext in (".jpg", ".jpeg") and out_np.dtype != np.uint8:
        if out_np.ndim == 2:
            x01 = np.clip(out_np.astype(np.float32) / float(meta.get("scale", 255.0)), 0, 1)
            out_np = np.round(x01 * 255.0).astype(np.uint8)
        else:
            x01 = np.clip(out_np.astype(np.float32) / float(meta.get("scale", 255.0)), 0, 1)
            out_np = np.round(x01 * 255.0).astype(np.uint8)

    imageio.imwrite(out_path, out_np)


# --------------------------
# Normal pipeline (single noisy)
# --------------------------
def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)

SSIM_KW_RAW = dict(
    gaussian_weights=True,
    sigma=1.5,
    use_sample_covariance=False,
    data_range=255,
)

def pair_downsamplerori(img: torch.Tensor, idx_pair=None, return_idx=False):
    """
    img: (B,C,H,W) on device
    4-phase subsampling -> choose 2 views -> DWT swap LL -> IDWT
    """
    H, W = img.shape[-2:]
    H2, W2 = H - (H % 2), W - (W % 2)
    img = img[..., :H2, :W2]

    views = [
        img[..., 0::2, 0::2],  # (0,0)
        img[..., 1::2, 0::2],  # (1,0)
        img[..., 1::2, 1::2],  # (1,1)
        img[..., 0::2, 1::2],  # (0,1)
    ]

    if idx_pair is None:
        idx = torch.randperm(4, device=img.device)[:2].tolist()
        idx_pair = (idx[0], idx[1])
    else:
        idx_pair = (int(idx_pair[0]), int(idx_pair[1]))

    output1 = views[idx_pair[0]]
    output2 = views[idx_pair[1]]

    h1, w1 = output1.shape[-2:]
    h2, w2 = output2.shape[-2:]
    hh = min(h1 - (h1 % 2), h2 - (h2 % 2))
    ww = min(w1 - (w1 % 2), w2 - (w2 % 2))
    output1_c = output1[..., :hh, :ww]
    output2_c = output2[..., :hh, :ww]

    dwt, idwt = get_wavelet(img.device)
    yl1, yh1 = dwt(output1_c)
    yl2, yh2 = dwt(output2_c)

    idwtoutput1 = idwt((yl2, yh1))
    idwtoutput2 = idwt((yl1, yh2))

    if return_idx:
        return output1, output2, idwtoutput1, idwtoutput2, idx_pair
    return output1, output2, idwtoutput1, idwtoutput2

def pair_downsampler(img: torch.Tensor, idx_pair=None, return_idx=False):
    """
    img: (B,C,H,W) on device
    4-phase subsampling -> choose 2 views -> DWT swap LL -> IDWT
    """
    H, W = img.shape[-2:]
    H2, W2 = H - (H % 2), W - (W % 2)
    img = img[..., :H2, :W2]

    views = [
        img[..., 0::2, 0::2],  # (0,0)
        img[..., 1::2, 0::2],  # (1,0)
        img[..., 1::2, 1::2],  # (1,1)
        img[..., 0::2, 1::2],  # (0,1)
    ]

    if idx_pair is None:
        idx = torch.randperm(4, device=img.device).tolist()  # length 4
        idx_pair = (idx[0], idx[1], idx[2], idx[3])
    else:
        idx_pair = tuple(int(i) for i in idx_pair)

    output1 = views[idx_pair[0]]
    output2 = views[idx_pair[1]]
    output3 = views[idx_pair[2]]
    output4 = views[idx_pair[3]]

    h1, w1 = output1.shape[-2:]
    h2, w2 = output2.shape[-2:]
    hh = min(h1 - (h1 % 2), h2 - (h2 % 2))
    ww = min(w1 - (w1 % 2), w2 - (w2 % 2))
    output1_c = output1[..., :hh, :ww]
    output2_c = output2[..., :hh, :ww]

    dwt, idwt = get_wavelet(img.device)
    yl1, yh1 = dwt(output1_c)
    yl2, yh2 = dwt(output2_c)

    idwtoutput1 = idwt((yl2, yh1))
    idwtoutput2 = idwt((yl1, yh2))

    if return_idx:
        return output1, output2, output3, output4, idwtoutput1, idwtoutput2, idx_pair
    return output1, output2, output3, output4, idwtoutput1, idwtoutput2

def loss_func_normal(noisy_img: torch.Tensor, model: nn.Module) -> torch.Tensor:
    noisy1, noisy2, noisy3, noisy4,noisyw1, noisyw2 = pair_downsampler(noisy_img)

    n1 = model(noisy1)
    n2 = model(noisy2)
    pred1  = noisy1  - n1
    pred2  = noisy2  - n2
    predw1 = noisyw1 - model(noisyw1)
    predw2 = noisyw2 - model(noisyw2)

    ssim_main = 0.5*ssim_pt(pred1, pred2, data_range=1.0, size_average=True)
    loss_ssim = 1.0 - ssim_main #+ 0.5 * (1.0 - ssim_wave)

    loss_main = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))
    loss_wave = 0.5 * (mse(noisyw1, predw2) + mse(noisyw2, predw1))
    alpha = 0.001
    return loss_main + alpha*loss_ssim + loss_wave

def random_patches(img: torch.Tensor, patch_size: int = 128, batch_size: int = 8) -> torch.Tensor:
    """
    img: (1,C,H,W)
    return: (batch_size,C,patch_size,patch_size)
    """
    if img.dim() != 4:
        raise ValueError("img must be 4D (B,C,H,W)")
    if img.shape[0] != 1:
        img = img[:1]

    _, _, H, W = img.shape

    if patch_size % 2 == 1:
        patch_size += 1
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image too small for patch_size={patch_size}: H={H}, W={W}")

    ys = torch.randint(0, H - patch_size + 1, (batch_size,), device=img.device)
    xs = torch.randint(0, W - patch_size + 1, (batch_size,), device=img.device)

    patches = []
    for i in range(batch_size):
        y = int(ys[i].item())
        x = int(xs[i].item())
        patches.append(img[0:1, :, y:y+patch_size, x:x+patch_size])

    return torch.cat(patches, dim=0)

def train_denoise(
    noisy_img: torch.Tensor,  # (1,C,H,W) CPU, [0,1]
    max_steps: int = 1000,
    patch_size: int = 128,
    batch_size: int = 8,
    lr: float = 1e-3,
    step_size: int = 800,
    gamma: float = 0.5,
) -> torch.Tensor:
    """
    returns: denoised (1,C,H,W) CPU in [0,1]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noisy_img = noisy_img.to(device=device, dtype=torch.float32)

    _, c, _, _ = noisy_img.shape
    model = NetworkRes(n_chan=c).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.train()
    for _ in range(max_steps):
        patch_batch = random_patches(noisy_img, patch_size=patch_size, batch_size=batch_size)
        loss = loss_func_normal(patch_batch, model)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    den = torch.clamp(noisy_img - model(noisy_img), 0, 1).detach().cpu()

    del model, optimizer, scheduler
    return den


# --------------------------
# SIM-4 pipeline
# --------------------------
def _crop_even_hw(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    return x[..., :h - (h % 2), :w - (w % 2)]

def _crop_to_min_even(*xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    h = min([x.shape[-2] for x in xs])
    w = min([x.shape[-1] for x in xs])
    h -= (h % 2)
    w -= (w % 2)
    return tuple(x[..., :h, :w] for x in xs)

def _down2_phase(x: torch.Tensor, phase: int) -> torch.Tensor:
    if phase == 0: return x[..., 0::2, 0::2]
    if phase == 1: return x[..., 1::2, 0::2]
    if phase == 2: return x[..., 1::2, 1::2]
    if phase == 3: return x[..., 0::2, 1::2]
    raise ValueError("phase must be 0..3")

def _down2_rand(x: torch.Tensor) -> torch.Tensor:
    ph = int(torch.randint(0, 4, (1,), device=x.device).item())
    return _down2_phase(x, ph)

def _rand_aligned_patches_4(views4: torch.Tensor, patch_size: int, batch_size: int):
    if views4.shape[0] != 4:
        raise ValueError(f"views4 must have V=4, got {views4.shape[0]}")
    _, _, H, W = views4.shape
    if patch_size % 2 == 1:
        patch_size += 1
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image too small: H={H}, W={W}, patch={patch_size}")

    ys = torch.randint(0, H - patch_size + 1, (batch_size,), device=views4.device)
    xs = torch.randint(0, W - patch_size + 1, (batch_size,), device=views4.device)

    p = [[] for _ in range(4)]
    for i in range(batch_size):
        y = int(ys[i].item()); x = int(xs[i].item())
        for k in range(4):
            p[k].append(views4[k:k+1, :, y:y+patch_size, x:x+patch_size])
    return tuple(torch.cat(p[k], 0) for k in range(4))

def sim4_make_pairs(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor):
    d1 = _down2_rand(_crop_even_hw(p1))
    d2 = _down2_rand(_crop_even_hw(p2))
    d3 = _down2_rand(_crop_even_hw(p3))
    d4 = _down2_rand(_crop_even_hw(p4))

    d1, d2, d3, d4 = _crop_to_min_even(d1, d2, d3, d4)

    dwt, idwt = get_wavelet(d1.device)
    yl1, yh1 = dwt(d1)
    yl2, yh2 = dwt(d2)

    mix1 = idwt((yl2, yh1))
    mix2 = idwt((yl1, yh2))

    mix1, mix2, d1, d2, d3, d4 = _crop_to_min_even(mix1, mix2, d1, d2, d3, d4)
    return d1, d2, d3, d4, mix1, mix2

def loss_func_sim4(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, model: nn.Module) -> torch.Tensor:
    d1, d2, d3, d4, mix1, mix2 = sim4_make_pairs(p1, p2, p3, p4)

    pred1 = d1 - model(d1)
    pred2 = d2 - model(d2)

    pred_mix1 = mix1 - model(mix1)
    pred_mix2 = mix2 - model(mix2)
    ssim_main = 0.5 * ssim_pt(pred1, pred2, data_range=1.0, size_average=True)
    loss_ssim = 1.0 - ssim_main

    loss_main = 0.5 * (mse(d1, pred2) + mse(d2, pred1))
    loss_wave = 0.5 * (mse(mix1, pred_mix2) + mse(mix2, pred_mix1))
    return loss_main + 0.001 * loss_ssim + loss_wave

def train_sim4_on_views(
    views4: torch.Tensor,  # (4,C,H,W)
    max_steps: int = 1000,
    patch_size: int = 128,
    batch_size: int = 8,
    lr: float = 1e-3,
    step_size: int = 800,
    gamma: float = 0.5,
    progress_every: int = 0,
) -> nn.Module:
    device = views4.device
    C = int(views4.shape[1])

    model = NetworkRes(n_chan=C).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    model.train()
    for step in range(1, max_steps + 1):
        perm = torch.randperm(4, device=device)
        v = views4[perm]

        p1, p2, p3, p4 = _rand_aligned_patches_4(v, patch_size=patch_size, batch_size=batch_size)
        loss = loss_func_sim4(p1, p2, p3, p4, model)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()

        if progress_every and (step == 1 or step == max_steps or step % progress_every == 0):
            print(f"STEP {step}/{max_steps} loss={float(loss.item()):.6f}", flush=True)

    return model.eval()


# ---- SIM png8 writer (contrast + multipage split) ----
def _png8_contrast_one(x_chw: torch.Tensor, mode: str) -> torch.Tensor:
    x = x_chw.detach().float().cpu()
    c, h, w = x.shape

    if mode == "none":
        return x.clamp(0, 1)

    # luminance
    if c == 1:
        base = x[0]
    else:
        base = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
    b = base.numpy()

    if mode == "minmax":
        lo, hi = float(b.min()), float(b.max())
    elif mode == "p1p99":
        lo, hi = np.percentile(b, 1.0), np.percentile(b, 99.0)
        lo, hi = float(lo), float(hi)
    else:
        lo, hi = 0.0, 1.0

    if hi > lo + 1e-8:
        x = (x - lo) / (hi - lo)
    return x.clamp(0, 1)

@torch.no_grad()
def save_png8_split(
    t01: torch.Tensor,          # (N,C,H,W)
    out_path: str,              # .png
    contrast: str = "none",     # none|minmax|p1p99
    split_multipage: bool = True
):
    root, ext = os.path.splitext(out_path)
    if ext.lower() != ".png":
        out_path = root + ".png"
        root, _ = os.path.splitext(out_path)

    if t01.ndim == 3:
        t01 = t01.unsqueeze(0)
    if t01.ndim != 4:
        raise ValueError("t01 must be (N,C,H,W) or (C,H,W)")

    t01 = t01.detach().float().cpu()
    N = t01.shape[0]

    def save_one(x_chw: torch.Tensor, p: str):
        x = _png8_contrast_one(x_chw, contrast)
        c, h, w = x.shape
        arr = x.numpy()

        if c == 1:
            img = np.round(arr[0] * 255.0).astype(np.uint8)
            cv2.imwrite(p, img)
            return

        img = np.transpose(arr, (1, 2, 0))
        img = np.round(img * 255.0).astype(np.uint8)
        if img.shape[2] == 3:
            img = img[:, :, ::-1]  # RGB->BGR
        cv2.imwrite(p, img)

    if N == 1 or (not split_multipage):
        save_one(t01[0], out_path)
        return

    for i in range(N):
        save_one(t01[i], f"{root}_page{i:03d}.png")

@torch.no_grad()
def save_tensor_png(t: torch.Tensor, out_path: str):
    """
    t: (1,C,H,W) or (C,H,W), in [0,1]
    write: 8-bit png via cv2, RGB->BGR if 3ch
    """
    if t.ndim == 4:
        t = t[0]
    t = t.detach().float().clamp(0, 1).cpu()
    c, h, w = t.shape
    arr = t.numpy()

    if c == 1:
        img = (arr[0] * 255.0).round().astype(np.uint8)
        cv2.imwrite(out_path, img)
    else:
        img = np.transpose(arr, (1, 2, 0))
        img = (img * 255.0).round().astype(np.uint8)
        if img.shape[2] == 3:
            img = img[:, :, ::-1]
        cv2.imwrite(out_path, img)

def denoise_folder_sim4(
    raw_path: str,
    view_paths: List[str],
    out_path: str,
    as_gray: bool = False,
    drop_alpha: bool = True,
    per_page_train: bool = True,
    # ---- output control ----
    save_like_raw: bool = False,  
    **train_kwargs,
) -> Dict:
    """
    SIM-4 (simplified as your requirement):
      - input: raw + views must be .tif/.tiff
      - ignore multipage: if stack, use only page0
      - output: .png via save_tensor_png (8-bit), unless save_like_raw=True
    """

    def _check_tif(p: str):
        ext = os.path.splitext(p)[1].lower()
        if ext not in (".tif", ".tiff"):
            raise ValueError(f"SIM mode only supports TIFF input, got: {p}")

    _check_tif(raw_path)
    for vp in view_paths[:4]:
        _check_tif(vp)

    # --- read raw + meta ---
    raw_t, raw_meta = read_image_any(raw_path, as_gray=as_gray, drop_alpha=drop_alpha)
    if raw_t.shape[0] > 1:
        raw_t = raw_t[0:1]

    # --- read 4 views (only first page) ---
    view_t_list = []
    for vp in view_paths[:4]:
        vt, _ = read_image_any(vp, as_gray=as_gray, drop_alpha=drop_alpha)
        if vt.shape[0] > 1:
            vt = vt[0:1]
        view_t_list.append(vt)

    if len(view_t_list) < 4:
        raise ValueError(f"Need 4 view files for SIM-4, got {len(view_t_list)}")

    # --- crop all to common (min even H,W) to avoid cat fail ---
    H = min([raw_t.shape[-2]] + [vt.shape[-2] for vt in view_t_list])
    W = min([raw_t.shape[-1]] + [vt.shape[-1] for vt in view_t_list])
    H -= (H % 2)
    W -= (W % 2)
    raw_t = raw_t[..., :H, :W]
    view_t_list = [vt[..., :H, :W] for vt in view_t_list]

    # --- build views4 ---
    views4 = torch.cat([vt[0:1] for vt in view_t_list], dim=0)  # (4,C,H,W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    views4 = views4.to(device=device, dtype=torch.float32)
    raw_dev = raw_t.to(device=device, dtype=torch.float32)

    # --- train + denoise (single page) ---
    model = train_sim4_on_views(views4, **train_kwargs)
    with torch.no_grad():
        den = torch.clamp(raw_dev - model(raw_dev), 0, 1).detach().cpu()
    del model

    # --- save ---
    if save_like_raw:
        write_image_like(out_path, den, raw_meta)
        return {**raw_meta, "out_path": out_path, "save_mode": "like_raw"}
    
    root, ext = os.path.splitext(out_path)
    if ext.lower() != ".png":
        out_path = root + ".png"

    save_tensor_png(den, out_path)
    return {**raw_meta, "out_path": out_path, "save_mode": "png_tensor"}


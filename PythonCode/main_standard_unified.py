import os, glob, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from natsort import natsorted

from util_unified import read_image_any, train_denoise, write_image_like

# --------- CONFIG ----------
in_dir = "/your noisy image root"
out_dir = "/your results root"
patterns = ("*.png", "*.tif", "*.tiff")
as_gray = True

# train params
max_steps  = 1200
patch_size = 128
batch_size = 8
lr         = 1e-3
step_size  = 960
gamma      = 0.5
# ---------------------------

os.makedirs(out_dir, exist_ok=True)

files = []
for pat in patterns:
    files += glob.glob(os.path.join(in_dir, pat))
files = natsorted(files)

total_start = time.perf_counter()

for i, p in enumerate(files):
    name = os.path.basename(p)
    out_path = os.path.join(out_dir, name)

    stack, meta = read_image_any(p, as_gray=as_gray)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    outs = []
    for s in range(stack.shape[0]):
        den = train_denoise(
            stack[s:s+1],
            max_steps=max_steps,
            patch_size=patch_size,
            batch_size=batch_size,
            lr=lr,
            step_size=step_size,
            gamma=gamma,
        )
        outs.append(den)

    out_stack = torch.cat(outs, dim=0)
    write_image_like(out_path, out_stack, meta)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    pages = meta.get("pages", 1)
    print(f"[{i+1}/{len(files)}] {name} | pages={pages} | time: {t1-t0:.3f}s")

total_time = time.perf_counter() - total_start
avg_time = total_time / max(len(files), 1)
print(f"TOTAL: {total_time:.2f}s | AVG per file: {avg_time:.2f}s | N={len(files)}")

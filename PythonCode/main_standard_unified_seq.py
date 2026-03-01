import os, glob, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from natsort import natsorted

from util_unified_seq import read_image_any, train_denoise, write_image_like

# --------- CONFIG ----------
in_dir = "/your noisy image root"
out_dir = "your results root"
patterns = ("*.png", "*.tif", "*.tiff")
as_gray = True

# train params
max_steps  = 1200
patch_size = 128
batch_size = 8
lr         = 1e-3
step_size  = 960
gamma      = 0.5

# sequence method
seq_denoise = True
seq_steps_first = 1200 
seq_steps_next  = 600 
min_steps = 200
decay = 0.9


os.makedirs(out_dir, exist_ok=True)

files = []
for pat in patterns:
    files += glob.glob(os.path.join(in_dir, pat))
files = natsorted(files)

total_start = time.perf_counter()

prev_state_by_page = {}

for i, p in enumerate(files):
    name = os.path.basename(p)
    out_path = os.path.join(out_dir, name)

    stack, meta = read_image_any(p, as_gray=as_gray)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    outs = []
    for s in range(stack.shape[0]):
        init_state = prev_state_by_page.get(s) if seq_denoise else None
        #steps = seq_steps_first if (init_state is None) else seq_steps_next
        steps = max(min_steps, int(seq_steps_first * (decay ** i)))

        if seq_denoise:
            den, new_state = train_denoise(
                stack[s:s+1],
                max_steps=steps,
                patch_size=patch_size,
                batch_size=batch_size,
                lr=lr,
                step_size=step_size,
                gamma=gamma,
                init_state_dict=init_state,
                return_state_dict=True,
                strict_load=True,
            )
            prev_state_by_page[s] = new_state
        else:
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

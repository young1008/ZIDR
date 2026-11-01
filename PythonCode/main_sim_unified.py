import os, glob, time
from natsort import natsorted
import torch
from util_unified import denoise_folder_sim4

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

img_root = "/your noisy image root"
out_root     = "/your results root"
patterns     = ("*.tif", "*.tiff", "*.png")

as_gray = False
per_page_train = True


save_like_raw  = False
# train params
max_steps      = 1200
patch_size     = 128
batch_size     = 8
lr             = 1e-3
step_size      = 960
gamma          = 0.5
progress_every = 0
# ------------------------

os.makedirs(out_root, exist_ok=True)

def list_images(folder):
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(folder, pat))
    return natsorted(files)

def pick_raw_and_views4(folder_files):
    raw = None
    for p in folder_files:
        if "raw" in os.path.basename(p).lower():
            raw = p
            break
    if raw is None:
        raw = folder_files[0]

    view_files = [p for p in folder_files if os.path.basename(p).lower().startswith("view")]
    if len(view_files) >= 4:
        views = view_files[:4]
    else:
        views = folder_files[:4]
    if len(views) < 4:
        raise ValueError(f"Need >=4 images in folder, got {len(views)}")
    return raw, views

folders = [p for p in glob.glob(os.path.join(img_root, "*")) if os.path.isdir(p)]
folders = natsorted(folders)

total_start = time.perf_counter()
ok, skip = 0, 0

for k, folder in enumerate(folders, 1):
    folder_name = os.path.basename(os.path.normpath(folder))
    files = list_images(folder)
    if len(files) < 4:
        print(f"[{k}/{len(folders)}] SKIP {folder_name}: images<{4}")
        skip += 1
        continue

    try:
        raw_path, view_paths = pick_raw_and_views4(files)
    except Exception as e:
        print(f"[{k}/{len(folders)}] SKIP {folder_name}: {e}")
        skip += 1
        continue

    out_path = os.path.join(out_root, f"{folder_name}.png")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    info = denoise_folder_sim4(
        raw_path=raw_path,
        view_paths=view_paths,
        out_path=out_path,
        as_gray=as_gray,
        save_like_raw=False, 

        max_steps=max_steps,
        patch_size=patch_size,
        batch_size=batch_size,
        lr=lr,
        step_size=step_size,
        gamma=gamma,
        progress_every=progress_every,
    )

    print("saved:", info["out_path"], "mode:", info["save_mode"])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(f"[{k}/{len(folders)}] OK  {folder_name} | raw={os.path.basename(raw_path)} | "
          f"views={[os.path.basename(p) for p in view_paths]} | time={t1-t0:.3f}s -> {out_path}")
    ok += 1

total_time = time.perf_counter() - total_start
print(f"\nDONE. ok={ok}, skip={skip}, total_time={total_time:.2f}s")

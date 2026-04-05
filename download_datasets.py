#!/usr/bin/env python3
"""Download all 24 Jackrong datasets from HuggingFace into High-fidelity Dataset folder."""

import os
from huggingface_hub import snapshot_download, HfApi

TARGET_DIR = os.path.join(os.path.dirname(__file__), "High-fidelity Dataset")
os.makedirs(TARGET_DIR, exist_ok=True)

# Get all datasets from Jackrong
api = HfApi()
datasets = list(api.list_datasets(author="Jackrong"))
print(f"Found {len(datasets)} datasets from Jackrong\n")

def split_large_file(file_path, chunk_size_mb=1500):
    if not os.path.exists(file_path):
        return
    file_size = os.path.getsize(file_path)
    chunk_size = chunk_size_mb * 1024 * 1024
    if file_size > chunk_size:
        print(f"  ⚠️ File too large for GitHub ({file_size/1024/1024/1024:.2f} GB). Splitting...")
        base, ext = os.path.splitext(file_path)
        part_num = 1
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk: break
                part_name = f"{base}_part{part_num:02d}{ext}"
                with open(part_name, 'wb') as out_f: out_f.write(chunk)
                part_num += 1
        os.remove(file_path)
        print(f"  ✓ Split into {part_num-1} parts.")

for i, ds in enumerate(datasets, 1):
    ds_name = ds.id.split("/")[-1]
    local_dir = os.path.join(TARGET_DIR, ds_name)
    
    print(f"[{i}/{len(datasets)}] Downloading: {ds.id}")
    print(f"  -> {local_dir}")
    
    try:
        snapshot_download(
            repo_id=ds.id,
            repo_type="dataset",
            local_dir=local_dir,
        )
        print(f"  ✓ Done")

        # Split any files larger than 2GB before pushing
        for root, _, files in os.walk(local_dir):
            for file in files:
                split_large_file(os.path.join(root, file))

        # Sync to Git
        print(f"  ⬆️ Syncing to GitHub...")
        os.system(f"git add '{local_dir}'")
        os.system(f"git commit -m 'feat: sync dataset {ds_name} to GitHub' --quiet")
        os.system(f"git push origin main --quiet")
        print(f"  ✓ Synced\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("=" * 60)
print(f"All downloads complete and synced! Files are in: {TARGET_DIR}")

import os

def split_file(file_path, chunk_size_mb=1500):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    file_size = os.path.getsize(file_path)
    chunk_size = chunk_size_mb * 1024 * 1024
    
    if file_size <= chunk_size:
        print(f"File {file_path} is already within limits ({file_size/1024/1024:.2f} MB).")
        return

    print(f"Splitting {file_path} ({file_size/1024/1024/1024:.2f} GB) into {chunk_size_mb} MB parts...")
    
    base, ext = os.path.splitext(file_path)
    part_num = 1
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            part_name = f"{base}_part{part_num:02d}{ext}"
            with open(part_name, 'wb') as out_f:
                out_f.write(chunk)
            print(f"  ✓ Created {part_name}")
            part_num += 1
            
    # Delete original large file
    os.remove(file_path)
    print(f"  ✓ Deleted original file {file_path}")

# Fix the two known files
split_file("High-fidelity Dataset/Competitive-Programming-python-blend/clean.jsonl")
split_file("High-fidelity Dataset/Natural-Reasoning-gpt-oss-120B-S1/gpt-oss-120B-natural-reasoning.jsonl")

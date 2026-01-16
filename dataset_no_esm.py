# build_split_only.py
import os
import re
import pickle as pk
import numpy as np
import argparse
from tqdm import tqdm

from utils import load_samples_from_csv   # hoặc class Sample bạn đang dùng

# --------------------------------------------------
# Parse args
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True, help='path to total.csv')
parser.add_argument('--root', type=str, required=True, help='dataset root')
args = parser.parse_args()

csv_path = args.csv
root = args.root

assert os.path.exists(csv_path), f"{csv_path} not found"

# --------------------------------------------------
# Load samples (NO ESM, NO DSSP)
# --------------------------------------------------
print("[INFO] Loading samples from CSV")
samples = load_samples_from_csv(csv_path)  
# ⚠️ hàm này chỉ tạo object (name, label, date, seq...)

print(f"[INFO] Total samples: {len(samples)}")

# --------------------------------------------------
# Filter (giữ giống paper)
# --------------------------------------------------
samples = [
    s for s in samples
    if len(s) < 1024
    and getattr(s, 'label', None) is not None
    and s.label.sum() > 0
]

print(f"[INFO] After filter: {len(samples)}")

# --------------------------------------------------
# Split by date
# --------------------------------------------------
month = {
    'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
    'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12
}

TEST_CUTOFF = 20210401
trainset, testset, dates = [], [], []

for s in tqdm(samples, desc="Splitting"):
    raw = str(s.date).strip()
    parts = re.split(r'[-/\s]+', raw)

    try:
        if len(parts) >= 3 and parts[1].isalpha():
            d = int(parts[0])
            m = month[parts[1].upper()[:3]]
            y = int(parts[2]) if len(parts[2]) == 4 else 2000 + int(parts[2])
        elif len(parts) >= 3 and parts[0].isdigit():
            y, m, d = map(int, parts[:3])
        else:
            continue
        date = y*10000 + m*100 + d
    except:
        continue

    if date < TEST_CUTOFF:
        trainset.append(s)
        dates.append(date)
    else:
        testset.append(s)

# --------------------------------------------------
# Save
# --------------------------------------------------
os.makedirs(root, exist_ok=True)

with open(f'{root}/train.pkl', 'wb') as f:
    pk.dump(trainset, f)

with open(f'{root}/test.pkl', 'wb') as f:
    pk.dump(testset, f)

idx = np.argsort(np.array(dates))
np.save(f'{root}/cross-validation.npy', idx)

print("[DONE]")
print(f"  Train: {len(trainset)}")
print(f"  Test : {len(testset)}")
print(f"  CV   : {idx.shape}")

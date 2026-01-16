# dataset_no_esm.py
import os
import torch
import warnings
import argparse
import re
import numpy as np
import pickle as pk
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import initial_no_esm   # <-- HÀM MỚI, KHÔNG DÙNG ESM

warnings.simplefilter('ignore')


# ============================================================
# Dataset dùng cho TRAIN / VAL / TEST (KHÔNG ESM)
# ============================================================
class PDB(Dataset):
    def __init__(
        self,
        mode='train',
        fold=-1,
        root='./data/BCE_633',
        self_cycle=False
    ):
        self.root = root
        assert mode in ['train', 'val', 'test']

        if mode in ['train', 'val']:
            with open(f'{self.root}/train.pkl', 'rb') as f:
                self.samples = pk.load(f)
        else:
            with open(f'{self.root}/test.pkl', 'rb') as f:
                self.samples = pk.load(f)

        self.data = []

        # ----------------------------------------------------
        # Cross-validation split
        # ----------------------------------------------------
        if mode == 'test' and not os.path.exists(f'{self.root}/cross-validation.npy'):
            order = list(range(len(self.samples)))
        else:
            if not os.path.exists(f'{self.root}/cross-validation.npy'):
                raise FileNotFoundError(
                    f"[ERROR] cross-validation.npy not found in {self.root} "
                    f"but mode='{mode}' requires it."
                )

            idx = np.load(f'{self.root}/cross-validation.npy')
            cv = 10
            inter = len(idx) // cv
            ex = len(idx) % cv

            if mode == 'train':
                order = []
                for i in range(cv):
                    if i == fold:
                        continue
                    order += list(idx[i*inter:(i+1)*inter + ex*(i == cv-1)])
            elif mode == 'val':
                order = list(idx[fold*inter:(fold+1)*inter + ex*(fold == cv-1)])
            else:
                order = list(range(len(self.samples)))

        order.sort()

        # ----------------------------------------------------
        # Load DSSP + Graph (NO ESM)
        # ----------------------------------------------------
        tbar = tqdm(order, desc=f'Loading {mode}')
        for i in tbar:
            sample = self.samples[i]
            name = sample.name
            tbar.set_postfix(chain=name)

            dssp_path = f"{self.root}/dssp/{name}.npy"
            graph_path = f"{self.root}/graph/{name}.npz"

            if not os.path.exists(dssp_path):
                print(f"[SKIP] Missing DSSP: {name}")
                continue
            if not os.path.exists(graph_path):
                print(f"[SKIP] Missing graph: {name}")
                continue

            try:
                sample.load_dssp(self.root)
                sample.load_adj(self.root, self_cycle)
            except Exception as e:
                print(f"[SKIP] {name} load failed: {repr(e)}")
                continue

            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return {
            'feat': seq.dssp,     # (L, 13) DSSP ONLY
            'label': seq.label,   # (L,)
            'adj': seq.adj,       # adjacency
            'edge': seq.edge,     # edge features
        }


# ============================================================
# Main: Build train.pkl / test.pkl WITHOUT ESM
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default='./data/BCE_633',
        help='dataset path'
    )
    args = parser.parse_args()

    root = args.root

    # --------------------------------------------------------
    # Prepare folders (NO feat/, NO esm/)
    # --------------------------------------------------------
    for d in [
        root,
        f'{root}/PDB',
        f'{root}/purePDB',
        f'{root}/dssp',
        f'{root}/graph'
    ]:
        os.makedirs(d, exist_ok=True)

    print(f"[INFO] Prepared folders under {root}")

    # --------------------------------------------------------
    # Build dataset from total.csv (NO ESM)
    # --------------------------------------------------------
    csv_path = 'total.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] {csv_path} not found")

    print("[INFO] Building dataset WITHOUT ESM")
    initial_no_esm(csv_path, root)

    # --------------------------------------------------------
    # Load full dataset
    # --------------------------------------------------------
    with open(f'{root}/total.pkl', 'rb') as f:
        dataset = pk.load(f)

    # Filter
    filt_data = [
        i for i in dataset
        if len(i) < 1024
        and getattr(i, 'label', None) is not None
        and i.label.sum() > 0
    ]

    # --------------------------------------------------------
    # Split by date (same logic as original)
    # --------------------------------------------------------
    month = {
        'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
        'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12
    }

    TEST_CUTOFF = 20210401
    trainset, testset = [], []
    dates_for_cv = []

    for it in filt_data:
        raw = str(it.date).strip()
        parts = re.split(r'[-/\s]+', raw)

        try:
            if len(parts) >= 3 and parts[1].isalpha():
                d = int(parts[0])
                m = month[parts[1].upper()[:3]]
                y_str = parts[2]
                y = int(y_str) if len(y_str) == 4 else (2000 + int(y_str))
            elif len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 4:
                y = int(parts[0]); m = int(parts[1]); d = int(parts[2])
            else:
                continue

            date = y * 10000 + m * 100 + d
        except:
            continue

        if date < TEST_CUTOFF:
            trainset.append(it)
            dates_for_cv.append(date)
        else:
            testset.append(it)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    with open(f'{root}/train.pkl', 'wb') as f:
        pk.dump(trainset, f)
    with open(f'{root}/test.pkl', 'wb') as f:
        pk.dump(testset, f)

    idx = np.array(dates_for_cv).argsort()
    np.save(f'{root}/cross-validation.npy', idx)

    print(f"[INFO] Done.")
    print(f"       Train: {len(trainset)}")
    print(f"       Test : {len(testset)}")
    print(f"       CV idx shape: {idx.shape}")

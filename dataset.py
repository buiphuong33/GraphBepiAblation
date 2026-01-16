import os
import esm
import torch
import warnings
import argparse
import torch.nn as nn
import re 
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader,Dataset
warnings.simplefilter('ignore')
class PDB(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data = []

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

        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            name = self.samples[i].name
            tbar.set_postfix(chain=name)

            feat_path = f"{self.root}/feat/{name}_esm2.ts"
            dssp_path = f"{self.root}/dssp/{name}.npy"
            graph_path = f"{self.root}/graph/{name}.npz"

            # thiếu feat → skip
            if not os.path.exists(feat_path):
                print(f"[SKIP] Missing feat: {name}")
                continue

            # thiếu dssp → skip
            if not os.path.exists(dssp_path):
                print(f"[SKIP] Missing dssp: {name}")
                continue

            try:
                self.samples[i].load_feat(self.root)
                self.samples[i].load_dssp(self.root)
                self.samples[i].load_adj(self.root, self_cycle)
            except Exception as e:
                print(f"[SKIP] {name} load failed: {repr(e)}")
                continue

            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        seq=self.data[idx]
        feat=torch.cat([seq.feat,seq.dssp],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }
        

if __name__ == "__main__":
    import pickle as pk
    import numpy as np
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/BCE_633', help='dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index, -1 for cpu')
    parser.add_argument('--esm_size', type=str, default='650M', choices=['150M','650M','3B'],
                        help='ESM-2 variant: 150M | 650M | 3B')
    parser.add_argument('--cache', type=str, default='/kaggle/working/graphbepi_cache',
                        help='TORCH_HOME model cache path')
    args = parser.parse_args()

    # 1) Cache cho ESM (để lần sau không tải lại)
    if args.cache:
        os.environ['TORCH_HOME'] = args.cache
        os.makedirs(args.cache, exist_ok=True)
        print(f"[INFO] TORCH_HOME = {os.environ['TORCH_HOME']}")

    # 2) Bảo đảm thư mục tồn tại (không dùng 'cd && mkdir')
    root = args.root
    for d in [root, f'{root}/PDB', f'{root}/purePDB', f'{root}/feat', f'{root}/dssp', f'{root}/graph']:
        os.makedirs(d, exist_ok=True)
    print(f"[INFO] Prepared folders under {root}")

    # 3) Chọn device an toàn
    if args.gpu == -1 or (not torch.cuda.is_available()):
        device = 'cpu'
    else:
        n = torch.cuda.device_count()
        device = f'cuda:{args.gpu}' if 0 <= args.gpu < n else 'cpu'
    print(f"[INFO] Device: {device}")

    # 4) Tải ESM-2 (mặc định 650M cho Kaggle)
    try:
        if args.esm_size == '3B':
            print("[INFO] Loading ESM-2 t36_3B_UR50D (large)")
            model, _ = esm.pretrained.esm2_t36_3B_UR50D()
        elif args.esm_size == '150M':
            print("[INFO] Loading ESM-2 t30_150M_UR50D")
            model, _ = esm.pretrained.esm2_t30_150M_UR50D()
        else:
            print("[INFO] Loading ESM-2 t33_650M_UR50D (recommended)")
            model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    except Exception as e:
        print("[ERROR] Failed to load ESM-2:", repr(e))
        raise

    model = model.to(device)
    model.eval()

    # 5) Build dataset (đọc 'total.csv' ở thư mục làm việc hiện tại)
    csv_path = 'total.csv'
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found. Check your working directory.")
    try:
        initial(csv_path, root, model, device)  # hàm trong utils.py
    except Exception as e:
        print("[ERROR] initial() failed:", repr(e))
        raise

    # 6) Tách train/test theo mốc 2021-04-01
    with open(f'{root}/total.pkl','rb') as f:
        dataset = pk.load(f)
    dates = {i.name: i.date for i in dataset}

    filt_data = [i for i in dataset if len(i) < 1024 and getattr(i, 'label', None) is not None and i.label.sum() > 0]
    month = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    trainset, testset, DATES_FOR_CV = [], [], []
    TEST_CUTOFF = 20210401

    TEST_CUTOFF = 20210401  # yyyymmdd
    dates_ = []
    trainset, testset = [], []

    for it in filt_data:
        raw = str(dates[it.name]).strip()  # ví dụ '11-FEB-21' hoặc '2019-07-03'
        # Chuẩn hóa phân tách
        parts = re.split(r'[-/\s]+', raw)

        try:
            if len(parts) >= 3 and parts[1].isalpha():
                # dạng 'DD-MON-YY' hoặc 'DD-MON-YYYY'
                d = int(parts[0])
                m = month[parts[1].upper()[:3]]
                y_str = parts[2]
                if len(y_str) == 4:
                    y = int(y_str)
                else:
                    y2 = int(y_str)
                    y = 2000 + y2 if y2 < 23 else 1900 + y2
            elif len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 4:
                # dạng 'YYYY-MM-DD'
                y = int(parts[0]); m = int(parts[1]); d = int(parts[2])
            else:
                print(f"[WARN] Unrecognized date '{raw}' for {it.name}; skipping")
                continue

            date = y*10000 + m*100 + d
        except Exception as e:
            print(f"[WARN] Bad date '{raw}' for {it.name}: {e}; skipping")
            continue

        if date < TEST_CUTOFF:
            dates_.append(date)
            trainset.append(it)
        else:
            testset.append(it)


    with open(f'{root}/train.pkl','wb') as f:
        pk.dump(trainset, f)
    with open(f'{root}/test.pkl','wb') as f:
        pk.dump(testset, f)

    #idx = np.array(DATES_FOR_CV).argsort()
    idx = np.array(dates_).argsort()
    np.save(f'{root}/cross-validation.npy', idx)
    print(f"[INFO] Done. Train: {len(trainset)}, Test: {len(testset)}, CV idx shape: {idx.shape}")


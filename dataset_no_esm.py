# dataset_no_esm.py
import os
import re
import argparse
import warnings
import pickle as pk
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.simplefilter('ignore')


# ============================================================
# Sample class (đúng chuẩn GraphBepi, KHÔNG ESM)
# ============================================================
class Sample:
    def __init__(self, name, seq, date=None):
        self.name = name
        self.seq = seq
        self.date = date

        L = len(seq)
        self.label = np.zeros(L, dtype=np.int64)

        self.dssp = None
        self.adj = None
        self.edge = None

    def __len__(self):
        return len(self.seq)

    def load_dssp(self, root):
        self.dssp = np.load(f"{root}/dssp/{self.name}.npy")

    def load_adj(self, root, self_cycle=False):
        data = np.load(f"{root}/graph/{self.name}.npz")
        self.adj = data["adj"]
        self.edge = data["edge"]


# ============================================================
# Utils
# ============================================================
def read_sequence_from_pdb(pdb_path):
    """Minimal PDB sequence reader"""
    seq = []
    aa_map = {
        'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
        'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
        'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
        'TRP':'W','TYR':'Y'
    }
    seen = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                resi = int(line[22:26])
                resn = line[17:20].strip()
                if resi not in seen and resn in aa_map:
                    seq.append(aa_map[resn])
                    seen.add(resi)
    return "".join(seq)


def load_sequences(root):
    seqs = {}
    pdb_dir = f"{root}/purePDB"
    for fn in os.listdir(pdb_dir):
        if fn.endswith(".pdb"):
            name = fn.replace(".pdb", "")
            seqs[name] = read_sequence_from_pdb(os.path.join(pdb_dir, fn))
    return seqs


def load_samples_from_csv(csv_path, sequences):
    import csv
    samples = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading CSV"):
            name = row["PDB chain"].strip()
            if name not in sequences:
                continue

            seq = sequences[name]
            sample = Sample(name=name, seq=seq)

            epi = row["Epitopes (resi_resn)"]
            items = re.split(r",\s*", epi)

            for it in items:
                try:
                    resi = int(it.split("_")[0])
                    idx = resi - 1
                    if 0 <= idx < len(seq):
                        sample.label[idx] = 1
                except:
                    continue

            samples.append(sample)

    return samples


# ============================================================
# Dataset class (GIỮ NGUYÊN GIAO DIỆN)
# ============================================================
class PDB(Dataset):
    def __init__(self, mode='train', fold=-1, root='./data/BCE_633', self_cycle=False):
        self.root = root
        assert mode in ['train', 'val', 'test']

        if mode in ['train', 'val']:
            with open(f'{root}/train.pkl', 'rb') as f:
                self.samples = pk.load(f)
        else:
            with open(f'{root}/test.pkl', 'rb') as f:
                self.samples = pk.load(f)

        self.data = []

        idx = np.load(f'{root}/cross-validation.npy')
        cv = 10
        inter = len(idx) // cv
        ex = len(idx) % cv

        if mode == 'train':
            order = []
            for i in range(cv):
                if i != fold:
                    order += list(idx[i*inter:(i+1)*inter + ex*(i==cv-1)])
        elif mode == 'val':
            order = list(idx[fold*inter:(fold+1)*inter + ex*(fold==cv-1)])
        else:
            order = list(range(len(self.samples)))

        order.sort()

        for i in tqdm(order, desc=f"Loading {mode}"):
            s = self.samples[i]
            try:
                s.load_dssp(root)
                s.load_adj(root, self_cycle)
                self.data.append(s)
            except:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "feat": s.dssp,    # (L,13)
            "label": s.label,  # (L,)
            "adj": s.adj,
            "edge": s.edge
        }


# ============================================================
# MAIN: build pkl ONLY (NO ESM, NO DOWNLOAD)
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    root = args.root
    csv_path = args.csv

    print(f"[INFO] root = {root}")
    print(f"[INFO] csv  = {csv_path}")

    sequences = load_sequences(root)
    samples = load_samples_from_csv(csv_path, sequences)

    with open(f"{root}/total.pkl", "wb") as f:
        pk.dump(samples, f)

    # ---------- split ----------
    TEST_CUTOFF = 20210401
    train, test, dates = [], [], []

    month = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
             'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}

    for s in samples:
        raw = str(s.date) if s.date else "01-JAN-20"
        parts = re.split(r'[-/\s]+', raw)
        try:
            d = int(parts[0])
            m = month[parts[1].upper()[:3]]
            y = 2000 + int(parts[2])
            date = y*10000 + m*100 + d
        except:
            date = 20200101

        if date < TEST_CUTOFF:
            train.append(s)
            dates.append(date)
        else:
            test.append(s)

    with open(f"{root}/train.pkl", "wb") as f:
        pk.dump(train, f)
    with open(f"{root}/test.pkl", "wb") as f:
        pk.dump(test, f)

    idx = np.array(dates).argsort()
    np.save(f"{root}/cross-validation.npy", idx)

    print("[INFO] DONE")
    print(f"  total: {len(samples)}")
    print(f"  train: {len(train)}")
    print(f"  test : {len(test)}")

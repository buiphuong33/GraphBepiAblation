import os
import pickle as pk
import traceback
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import torch
import esm

from utils import chain, extract_chain, process_chain


def parse_epi_list(epi_str):
    out = defaultdict(list)
    items = [x.strip() for x in str(epi_str).split(",") if x.strip()]
    for it in items:
        parts = it.split("_")
        if len(parts) != 3:
            continue
        site, resn, ch = parts
        site = site.strip()
        resn = resn.strip()
        ch = ch.strip()
        if not site or not resn or not ch:
            continue
        out[ch].append((site, resn))
    return out


def choose_device(gpu):
    if gpu == -1 or not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    if gpu < 0 or gpu >= n:
        return "cpu"
    return f"cuda:{gpu}"


def ensure_folders(root):
    for d in ["PDB", "purePDB", "feat", "dssp", "graph"]:
        os.makedirs(os.path.join(root, d), exist_ok=True)


def load_esm(esm_size, device):
    if esm_size == "3B":
        model, _ = esm.pretrained.esm2_t36_3B_UR50D()
    elif esm_size == "150M":
        model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    else:
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    return model


def main(args):
    if args.cache:
        os.environ["TORCH_HOME"] = args.cache
        os.makedirs(args.cache, exist_ok=True)

    root = args.root
    ensure_folders(root)

    device = choose_device(args.gpu)
    print(f"[INFO] Device: {device}")

    esm_model = load_esm(args.esm_size, device)

    df = pd.read_csv(args.csv)

    chain_map = defaultdict(list)
    for _, row in df.iterrows():
        pdb = str(row["PDB ID"]).lower().strip()
        epi_map = parse_epi_list(row["Epitope List (residueid_residuename_chain)"])
        for ch, pairs in epi_map.items():
            chain_map[(pdb, ch)].extend(pairs)

    print(f"[INFO] Total test chains: {len(chain_map)}")

    test_samples = []

    for (pdb, ch), epi_pairs in tqdm(chain_map.items(), desc="Building test set"):
        name = f"{pdb}_{ch}"

        try:
            ok = extract_chain(root, pdb, ch)
            if not ok:
                continue
        except Exception:
            continue

        obj = chain()
        obj.protein_name = pdb
        obj.chain_name = ch
        obj.name = name

        try:
            process_chain(obj, root, obj.name, esm_model, device)
        except Exception:
            traceback.print_exc()
            continue

        for site, resn in epi_pairs:
            try:
                obj.update(str(site), str(resn))
            except Exception:
                continue

        if obj.label.sum().item() == 0:
            print(f"[WARN] {name}: no positive residues after mapping")

        test_samples.append(obj)

    out_path = os.path.join(root, "test.pkl")
    with open(out_path, "wb") as f:
        pk.dump(test_samples, f)

    print(f"[DONE] test.pkl written: {out_path}")
    print(f"[INFO] Chains: {len(test_samples)}")
    print(f"[INFO] Dataset root: {root}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--esm_size", default="650M", choices=["150M", "650M", "3B"])
    ap.add_argument("--cache", default="/kaggle/working/graphbepi_cache")
    args = ap.parse_args()

    main(args)

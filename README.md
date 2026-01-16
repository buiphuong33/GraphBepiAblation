# Intro

GraphBepi is a novel graph-based method for accurate B-cell epitope prediction, which is able to capture
spatial information using the predicted protein structures through the edge-enhanced deep graph neural network.

We recommend you to use the [web server](http://bio-web1.nscc-gz.cn/app/graphbepi) of GraphBepi
if your input is small.

![(Variational) gcn](Framework.png)

# System requirement

GraphBepi is developed under Linux environment with:

- python 3.9.12
- numpy 1.21.5
- pandas 1.4.2
- fair-esm 2.0.0
- torch 1.12.1
- pytorch-lightning 1.6.4
- (optional) esmfold

# Software requirement

To run the full & accurate version of GraphBepi, you need to make sure the following software is in the [mkdssp](./mkdssp) directory:  
[DSSP](https://github.com/cmbi/dssp) (_dssp ver 2.0.4_ is Already in this repository)

# Build dataset

1. `git clone https://github.com/biomed-AI/GraphBepi.git && cd GraphBepi`
2. `python dataset.py --gpu 0`

It will take about 20 minutes to download the pretrained ESM-2 model and an hour to build our dataset with CUDA.

# Run GraphBepi for training

After building our dataset **_BCE_633_**, train the model with default hyper params:

```
python train.py --dataset BCE_633
```

# Run GraphBepi for prediction

1. Please execute the following command directly if you can provide the PDB file.
2. If you do not have a PDB file, you can use [AlphaFold2](http://bio-web1.nscc-gz.cn/app/alphaFold2_bio) to predict the protein structure.

```
python test.py -i pdb_file -p --gpu 0 -o ./output
```

or

We have also deployed a faster structural prediction model [ESMFold](https://github.com/facebookresearch/esm) in our project, so you can process the sequences directly by following the commands below.

```
python test.py -i fasta_file -f --gpu 0 -o ./output
```

## Optional: train/predict with XGBoost (tabular per-residue) — full workflow with GNN embeddings

This project includes a lightweight XGBoost workflow that can use per-residue tabular features (ESM-2 + DSSP + graph aggregates) and optionally incorporate per-residue **GNN embeddings** (feature-level fusion). Below is a complete step-by-step guide, commands, and verification checks.

### Prerequisites

- Python packages: `fair-esm`, `torch`, `pytorch-lightning`, `torchmetrics`, `xgboost`, `scikit-learn`, `joblib`, `pandas`, `numpy`, `tqdm`, `requests`.
- `mkdssp` binary must be executable in `./mkdssp/mkdssp` for DSSP computation.
- For Kaggle: copy read-only inputs into `/kaggle/working/` and set `TORCH_HOME` to a writable cache (example below).

---

### 1) Export per-residue tabular features (ESM + DSSP + graph)

- This writes `tabular/<split>.npz` with keys: `X`, `y`, `names`, `idxs`, `resn`.

```bash
python -c "from utils import export_tabular; export_tabular('./data/BCE_633','./tabular','train'); export_tabular('./data/BCE_633','./tabular','test')"
```

Verification:

```python
import numpy as np
d=np.load('./tabular/train.npz', allow_pickle=True)
print(d['X'].shape, d['y'].shape)  # (N, D), (N,)
```

---

### 2) Export GNN embeddings (feature-level fusion)

- Use your trained GraphBepi checkpoint to export per-residue embeddings.
- Two modes: default `model.embed()` (includes LSTM) or `--gnn-only` (skip BiLSTM, GNN-only embeddings).

```bash
# GNN-only (recommended if you want pure GNN features for XGBoost)
python export_gnn_emb.py \
  --ckpt ./model/BCE_633_GraphBepi/model_-1.ckpt \
  --root ./data/BCE_633 --out ./tabular --split train --gpu 0 --gnn-only

python export_gnn_emb.py \
  --ckpt ./model/BCE_633_GraphBepi/model_-1.ckpt \
  --root ./data/BCE_633 --out ./tabular --split test  --gpu 0 --gnn-only
```

- Output: `tabular/gnn_<split>_gnnonly.npz` (keys: `emb`, `names`, `idxs`, `resn`).

Quick smoke test (limit exported residues):

```bash
python export_gnn_emb.py --ckpt ./model/.../model_-1.ckpt --root ./data/BCE_633 --out ./tabular --split train --gpu 0 --gnn-only --limit 200
```

Verify shapes:

```python
g=np.load('./tabular/gnn_train_gnnonly.npz', allow_pickle=True)
print(g['emb'].shape, len(g['names']))
```

---

### 3) Merge tabular features + GNN embeddings and apply PCA

- Fit PCA on **train** embeddings and save the PCA model. Then transform test embeddings with the same PCA.
- Default PCA dim in scripts: `10` (configurable with `--pca-dim`).

```bash
python merge_tabular_and_gnn.py --tabular ./tabular --split train --pca-dim 10
python merge_tabular_and_gnn.py --tabular ./tabular --split test --pca-dim 10 --pca-model ./tabular/pca_train_10.joblib
```

- Output: `tabular/train_merged.npz`, `tabular/test_merged.npz`, and `tabular/pca_train_10.joblib`.

Verify merge:

```python
m=np.load('./tabular/train_merged.npz', allow_pickle=True)
print('merged X shape', m['X'].shape)
```

Tip: inspect PCA explained variance if you want to choose a different `pca-dim`.

```python
import joblib
pca=joblib.load('./tabular/pca_train_10.joblib')
print(pca.explained_variance_ratio_.cumsum())
```

---

### 4) Train XGBoost on merged features

- Train and save a model using `train_xgb.py`. If `--use-gnn` is passed and merged files are missing, the script will call the merger automatically.

```bash
python train_xgb.py \
  --root ./data/BCE_633 --out ./tabular --split train \
  --out-model ./model/xgb_gnn.joblib --use-gnn --pca-dim 10
```

- The script reports: AUROC, AUPRC and automatically selected **F1**, **MCC**, and the chosen threshold.

---

### 5) Predict & export per-protein CSVs

```bash
python predict_xgb.py --model ./model/xgb_gnn.joblib --tabular-dir ./tabular --output ./xgb_gnn_output --use-gnn --pca-dim 10
```

- Output: one CSV per protein in `./xgb_gnn_output/` with columns `resn`, `score`, `is epitope`.

---

### 6) Notes for Kaggle (read-only input, use /kaggle/working)

1. Copy inputs into writable location:

```bash
mkdir -p /kaggle/working/data
cp -r /kaggle/input/dataset/data/BCE_633 /kaggle/working/data/BCE_633
```

2. Copy ESM checkpoint to cache & set `TORCH_HOME`:

```bash
mkdir -p /kaggle/working/graphbepi_cache
cp -r /kaggle/input/dataset/esm-cp/* /kaggle/working/graphbepi_cache/
export TORCH_HOME=/kaggle/working/graphbepi_cache
```

3. Then run steps 1–5 above using `/kaggle/working/...` paths.

---

### 7) Troubleshooting & tips

- If export is slow: ensure GPU is enabled and `--gpu` points to the correct device index.
- If merge reports missing GNN rows: check `(names, idxs)` alignment between `tabular/*.npz` and `gnn_*.npz`.
- To debug quickly, use `--limit` in `export_gnn_emb.py` to export a small subset.

---

Notes: exported features include ESM-2 residue embeddings, DSSP features, degree and neighbor-aggregated edge features. The XGBoost scripts are meant as a lightweight alternative to the BiLSTM+GNN pipeline.

# Web server, citation and contact

The GrpahBepi web server is freely available: [interface](http://bio-web1.nscc-gz.cn/app/graphbepi)

Citation:

```

@article{zengys,
  title={Identifying the B-cell epitopes using AlphaFold2 predicted structures and pretrained language model},
  author={Yuansong Zeng, Zhuoyi Wei, Qianmu Yuan, Sheng Chen, Weijiang Yu, Jianzhao Gao, and Yuedong Yang},
  journal={biorxiv},
  year={2022}
 publisher={Cold Spring Harbor Laboratory}
}

```

Contact:  
Zhuoyi Wei (weizhy8@mail2.sysu.edu.cn)
Yuansong Zeng (zengys@mail.sysu.edu.cn)

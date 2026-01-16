import os
import argparse
import numpy as np
import joblib
import pandas as pd
from tool import METRICS
import torch


def main(args):
    os.makedirs(args.output, exist_ok=True)
    model = joblib.load(args.model)
    # prefer merged test file if using gnn
    if args.use_gnn:
        test_path = os.path.join(args.tabular_dir, 'test_merged.npz')
        if not os.path.exists(test_path):
            # try to merge automatically
            from merge_tabular_and_gnn import merge
            print('[INFO] Merging tabular and GNN embeddings for test...')
            merge(args.tabular_dir, 'test', args.pca_dim)
            test_path = os.path.join(args.tabular_dir, 'test_merged.npz')
    else:
        test_path = os.path.join(args.tabular_dir, 'test.npz')

    data = np.load(test_path, allow_pickle=True)
    X = data['X']
    y = data['y'] if 'y' in data else None
    names = data['names']
    idxs = data['idxs']
    resn = data['resn']

    proba = model.predict_proba(X)[:,1]

    if y is not None and args.threshold is None:
        metrics = METRICS(device='cpu')
        thr = metrics.calc_thresh(torch.tensor(proba), torch.tensor(y))
        print(f"[INFO] Auto threshold selected: {thr:.4f}")
    else:
        thr = args.threshold if args.threshold is not None else 0.1763
        print(f"[INFO] Using threshold: {thr}")

    # Save global preds
    np.savez_compressed(os.path.join(args.output, 'xgb_preds.npz'), proba=proba, names=names, idxs=idxs, resn=resn)

    # write per-protein CSV
    unique = np.unique(names)
    for u in unique:
        mask = names == u
        scores = proba[mask]
        residues = resn[mask]
        is_ep = (scores > thr).astype(int)
        df = pd.DataFrame({'resn': list(residues), 'score': list(scores), 'is epitope': list(is_ep)})
        df.to_csv(os.path.join(args.output, f'{u}.csv'), index=False)

    print(f"[DONE] Wrote {len(unique)} CSVs to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./model/xgb_model.joblib')
    parser.add_argument('--tabular-dir', type=str, default='./tabular')
    parser.add_argument('--output', type=str, default='./xgb_output')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--use-gnn', action='store_true', help='Use merged GNN embeddings')
    parser.add_argument('--pca-dim', type=int, default=10, help='PCA dim if merging GNN emb for test')
    args = parser.parse_args()
    main(args)

#train_xgb.py
import os
import argparse
import numpy as np
from utils import export_tabular
from tool import METRICS
import joblib
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch



def load_or_export(root, out_dir, split, use_gnn=False, pca_dim=0):
    # prefer merged file if using GNN embeddings
    merged_path = os.path.join(out_dir, f'{split}_merged.npz')
    if use_gnn and os.path.exists(merged_path):
        print(f"[INFO] Using merged features {merged_path}")
        return np.load(merged_path, allow_pickle=True)

    path = os.path.join(out_dir, f'{split}.npz')
    if not os.path.exists(path):
        print(f"[INFO] {path} not found, exporting via utils.export_tabular...")
        export_tabular(root, out_dir, split='all' if split=='all' else split)
    data = np.load(path, allow_pickle=True)

    # if requested and merged not present, try to merge automatically
    if use_gnn:
        from merge_tabular_and_gnn import merge
        print('[INFO] Merging tabular and GNN embeddings (PCA dim:', pca_dim, ')')
        merged_path, pca_model = merge(out_dir, split, pca_dim)
        return np.load(merged_path, allow_pickle=True)

    return data


def main(args):
    os.makedirs(args.out, exist_ok=True)
    data = load_or_export(args.root, args.out, args.split, use_gnn=args.use_gnn, pca_dim=args.pca_dim)
    X = data['X']
    y = data['y']

    # if split='all', we do train/val/test split locally
    if args.split == 'all':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    elif args.split == 'train':
        # training expects a separate test.npz for final eval
        X_train, y_train = X, y
        test_path = os.path.join(args.out, 'test_merged.npz' if args.use_gnn else 'test.npz')
        if not os.path.exists(test_path):
            print('[WARN] test.npz not found; final evaluation will be on a held-out val split')
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
        else:
            test = np.load(test_path, allow_pickle=True)
            X_test, y_test = test['X'], test['y']
    else:
        raise ValueError('split must be train or all')

    # further split X_train->train/val for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    spw = float(neg) / float(pos + 1e-9)
    print(f"[INFO] pos={pos} neg={neg} scale_pos_weight={spw:.2f}")

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if len(seeds) != 3:
        print(f"[WARN] You requested 3 models; seeds provided = {len(seeds)}. Using: {seeds}")

    eval_set = [(X_val, y_val)]
    print('[INFO] Training XGBoost ensemble on GPU...')

    param_grid = [
        dict(
            max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.65,
            min_child_weight=10,
            reg_lambda=5.0, reg_alpha=1.0,
            gamma=2.0
        ),
        dict(
            max_depth=6, learning_rate=0.03,
            subsample=0.80, colsample_bytree=0.80,
            min_child_weight=3,
            reg_lambda=2.0, reg_alpha=0.0,
            gamma=0.0
        ),
        dict(
            max_depth=7, learning_rate=0.05,
            subsample=0.70, colsample_bytree=0.90,
            min_child_weight=1,
            reg_lambda=1.0, reg_alpha=0.0,
            gamma=0.0
        ),
    ]
    probas = []
    n_models = min(len(seeds), len(param_grid))

    for i in range(n_models):
        seed = seeds[i]
        hp = param_grid[i]
        print(f"\n[INFO] Model {i+1}/{n_models} seed={seed} hp={hp}")

        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=['auc','aucpr'],   
            n_estimators=3000,      
            learning_rate=hp["learning_rate"],
            max_depth=hp["max_depth"],
            subsample=hp["subsample"],
            colsample_bytree=hp["colsample_bytree"],
            min_child_weight=hp["min_child_weight"],
            reg_lambda=hp["reg_lambda"],
            reg_alpha=hp["reg_alpha"],
            gamma=hp["gamma"],
            scale_pos_weight=spw,

            tree_method='hist',      
            random_state=seed,
            verbosity=1,
            n_jobs=max(1, os.cpu_count()-1)
        )

        clf.fit(X_tr, y_tr, eval_set=eval_set, verbose=50)

        base, ext = os.path.splitext(args.out_model)
        model_path = f"{base}_m{i+1}_seed{seed}{ext}"
        dirn = os.path.dirname(model_path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        joblib.dump(clf, model_path)
        print(f"[DONE] Saved model to {model_path}")

        probas.append(clf.predict_proba(X_test)[:, 1])

    # mean ensemble
    proba = np.mean(np.vstack(probas), axis=0)
    print(f"\n[INFO] Ensemble done: averaged {len(probas)} models")

    metrics = METRICS(device='cpu')
    pred_t = torch.tensor(proba)
    y_t = torch.tensor(y_test)
    res = metrics.calc_prc(pred_t, y_t)
    print('[RESULT] AUROC/AUPRC:', res['AUROC'], res['AUPRC'])
    # compute thresholded metrics (auto-select threshold by maximizing F1)
    thr_metrics = metrics(pred_t, y_t)
    print(f"[RESULT] F1: {thr_metrics['F1']:.4f}, MCC: {thr_metrics['MCC']:.4f}, Threshold: {thr_metrics['threshold']:.4f}")

    # save predictions
    np.savez_compressed(os.path.join(args.out, 'xgb_test_preds.npz'), proba=proba, y=y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/kaggle/input/dataset/data/BCE_633')
    parser.add_argument('--out', type=str, default='./tabular')
    parser.add_argument('--split', type=str, default='train', choices=['train','all'])
    parser.add_argument('--out-model', type=str, default='./model/xgb_model.joblib')
    parser.add_argument('--use-gnn', action='store_true', help='Use GNN embeddings merged into features')
    parser.add_argument('--pca-dim', type=int, default=10, help='PCA dim for GNN embeddings (fit on train)')
    parser.add_argument('--seeds', type=str, default='42,202,777',
                    help='Comma-separated random seeds for ensemble (e.g., 42,202,777)')
    args = parser.parse_args()
    main(args)

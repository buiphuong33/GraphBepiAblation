import os
import argparse
import numpy as np
import joblib
import torch
from tool import METRICS
from utils import export_tabular


def load_test_features(root, out_dir, use_gnn=False):
    """
    Load test features 
    """
    path = os.path.join(
        out_dir,
        'test_merged.npz' if use_gnn else 'test.npz'
    )

    if not os.path.exists(path):
        print(f"[INFO] {path} not found → exporting tabular features")
        export_tabular(root, out_dir, split='test')

    data = np.load(path, allow_pickle=True)

    X = data['X']
    y = data['y'] if 'y' in data else None
    return X, y


def load_xgb_ensemble(model_prefix):
    """
    Load all XGBoost checkpoints from prefix
    """
    model_dir = os.path.dirname(model_prefix)
    base = os.path.basename(model_prefix).replace('.joblib', '')

    model_paths = sorted([
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.startswith(base) and f.endswith('.joblib')
    ])

    if len(model_paths) == 0:
        raise RuntimeError("No XGBoost checkpoints found")

    print(f"[INFO] Found {len(model_paths)} models:")
    for p in model_paths:
        print("  ", p)

    models = [joblib.load(p) for p in model_paths]
    return models


def main(args):
    # 1. Load test features
    X_test, y_test = load_test_features(
        args.root,
        args.out,
        use_gnn=args.use_gnn
    )
    print(f"[INFO] Test shape: {X_test.shape}")

    # 2. Load XGBoost ensemble
    models = load_xgb_ensemble(args.model_prefix)

    # 3. Inference
    probas = []
    for clf in models:
        probas.append(clf.predict_proba(X_test)[:, 1])

    proba = np.mean(np.vstack(probas), axis=0)
    print("[INFO] Inference done")

    # 4. Metrics
    if y_test is not None:
        metrics = METRICS(device='cpu')
        pred_t = torch.tensor(proba)
        y_t = torch.tensor(y_test)

        res = metrics.calc_prc(pred_t, y_t)
        thr = metrics(pred_t, y_t)

        print("[RESULT] AUROC:", res['AUROC'])
        print("[RESULT] AUPRC:", res['AUPRC'])
        print("[RESULT] F1:", thr['F1'])
        print("[RESULT] MCC:", thr['MCC'])
        print("[RESULT] Threshold:", thr['threshold'])

    # 5. Save predictions
    out_path = os.path.join(args.out, 'xgb_test_inference.npz')
    np.savez_compressed(out_path, proba=proba, y=y_test)
    print(f"[DONE] Saved inference to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root of NEW test dataset (Epitope3D)')
    parser.add_argument('--out', type=str, default='./tabular',
                        help='Directory containing test.npz or test_merged.npz')
    parser.add_argument('--model-prefix', type=str, required=True,
                        help='Prefix of trained XGBoost model, e.g. xgb_gnn.joblib')
    parser.add_argument('--use-gnn', action='store_true',
                        help='Use merged GNN features (test_merged.npz)')
    args = parser.parse_args()

    main(args)

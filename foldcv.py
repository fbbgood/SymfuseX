# foldcv.py

import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

# Reuse your existing modules (no changes to them)
from modules import SymfuseX
from trainer import Trainer
from dataloader import LoadDataset

# Prefer your existing helpers if present
try:
    from Integerization import graph_collate_func, set_seed, mkdir
except Exception:
    def graph_collate_func(batch):  # very light fallback; your real func will be used if available
        return tuple(zip(*batch))
    def set_seed(seed: int = 3407):
        import random
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    def mkdir(p):
        os.makedirs(p, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _cache_from_df(df: pd.DataFrame, max_nodes: int):
    """
    Build an in-memory cache_dict using your existing LoadDataset __getitem__ logic.
    No disk writes, no interference with your current cache files.
    """
    ds = LoadDataset(df.index.values, df, max_drug_nodes=int(max_nodes), cache_dict=None)
    graphs, proteins, labels, smiles = [], [], [], []
    for i in tqdm(range(len(ds)), desc="Preprocess(struct)", dynamic_ncols=True, leave=True):
        g, p, y, s = ds[i]
        graphs.append(g)
        proteins.append(torch.as_tensor(p).long())
        labels.append(float(y))
        smiles.append(s)
    out = {
        "graphs": graphs,
        "proteins": proteins,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "smiles": smiles,
        "seqs": df["Protein"].astype(str).tolist() if "Protein" in df.columns else None,
    }
    return out


def _run_one_fold(cfg, task: str, df_tr: pd.DataFrame, df_va: pd.DataFrame, outdir: str, fold_seed: int):
    """
    Train/validate one fold using your existing Trainer. No test split here.
    """
    set_seed(int(fold_seed))

    # 1) offline cache in memory
    tr_cache = _cache_from_df(df_tr, max_nodes=int(cfg.DRUG.MAX_NODES))
    va_cache = _cache_from_df(df_va, max_nodes=int(cfg.DRUG.MAX_NODES))

    # 2) datasets/loaders (reuse your collate)
    train_dataset = LoadDataset(None, cache_dict=tr_cache)
    val_dataset   = LoadDataset(None, cache_dict=va_cache)

    params = {
        "batch_size": int(cfg.SOLVER.BATCH_SIZE),
        "shuffle": True,
        "drop_last": True,
        "collate_fn": graph_collate_func,
        "num_workers": int(cfg.SOLVER.NUM_WORKERS),
        "pin_memory": (DEVICE.type == "cuda"),
    }
    train_loader = DataLoader(train_dataset, **params)
    eval_params = dict(params); eval_params["shuffle"] = False; eval_params["drop_last"] = False
    val_loader  = DataLoader(val_dataset, **eval_params)

    # 3) optional init for DTA
    init_bias = None
    if task.upper() == "DTA":
        use_norm = bool(cfg.TRAIN.REG_NORMALIZE)
        init_bias = 0.0 if use_norm else float(tr_cache["labels"].mean().item())

    # 4) model/opt/trainer
    model = SymfuseX(task=task.upper(), init_bias=init_bias, **cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.SOLVER.LR))
    torch.backends.cudnn.benchmark = True

    mkdir(outdir)
    cfg_fold = deepcopy(cfg)
    cfg_fold.RESULT.OUTPUT_DIR = outdir

    trainer = Trainer(model, opt, DEVICE, train_loader, val_loader, None, task=task.upper(), **cfg_fold)
    metrics = trainer.train()  # same return schema as your original pipeline
    return metrics


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _write_results_txt(output_dir: str, rows: List[Dict], task: str, k: int):
    """
    Save ONLY kfold_results as a TXT table (no CSV, no additional summary file).
    """
    mkdir(output_dir)
    txt_path = os.path.join(output_dir, "kfold_results.txt")

    # Build a stable header
    all_keys = set()
    for r in rows:
        all_keys |= set(r.keys())
    front = ["Fold", "Split", "Best_epoch"]
    prio_cls = ["AUROC", "AUPRC", "F1", "Accuracy", "Sensitivity", "Specificity", "Precision", "Threshold", "Test_loss"]
    prio_reg = ["MSE", "CI", "RM2", "Val_Loss", "Test_loss"]

    if task.upper() == "DTI":
        header = front + [k for k in prio_cls if k in all_keys] + sorted(all_keys - set(front + prio_cls))
    else:
        header = front + [k for k in prio_reg if k in all_keys] + sorted(all_keys - set(front + prio_reg))

    lines = [f"kfold_results (k={k})", "\t".join(header)]
    for r in rows:
        line = "\t".join(_fmt(r.get(k, "")) for k in header)
        lines.append(line)

    with open(txt_path, "w") as fw:
        fw.write("\n".join(lines))

    print(f"[Saved] {txt_path}")


def run_5fold_cv(cfg, dataset_name: str, task: str, output_dir: str, n_splits: int = 5):
    """
    Public entry. Reads ./datasets/<dataset>/fold/full.csv, runs K-fold CV.
    Saves ONLY 'kfold_results.txt'.
    Returns: {"kfold_results": rows}
    """
    root = os.path.join("./datasets", dataset_name, "fold")
    full_csv = os.path.join(root, "full.csv")
    if not os.path.isfile(full_csv):
        raise FileNotFoundError(f"[fold] Not found: {full_csv}")

    df = pd.read_csv(full_csv).reset_index(drop=True)
    if len(df) < n_splits:
        raise ValueError(f"[fold] rows={len(df)} < n_splits={n_splits}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.SOLVER.SEED))
    rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(df), start=1):
        print(f"\n========== Fold {fold_id}/{n_splits} ==========")
        df_tr = df.iloc[tr_idx].copy().reset_index(drop=True)
        df_va = df.iloc[va_idx].copy().reset_index(drop=True)

        fold_out = os.path.join(output_dir, f"fold_{fold_id}")
        metrics = _run_one_fold(cfg, task, df_tr, df_va, fold_out, fold_seed=int(cfg.SOLVER.SEED) + fold_id)
        rec = dict(metrics) if isinstance(metrics, dict) else {}
        rec["Fold"] = fold_id
        rows.append(rec)

    _write_results_txt(output_dir, rows, task, k=n_splits)
    return {"kfold_results": rows}

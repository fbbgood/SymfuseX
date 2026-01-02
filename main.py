# -*- coding: utf-8 -*-
import os
import argparse
import warnings
from time import time

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import SymfuseX
from configs import get_cfg_defaults
from dataloader import LoadDataset
from trainer import Trainer
from Integerization import set_seed, graph_collate_func, mkdir

device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

parser = argparse.ArgumentParser(description="SymfuseX for DTI or DTA prediction")
parser.add_argument("--cfg", required=True, type=str, help="path to config file (yaml)")
parser.add_argument("--task", required=True, type=str, choices=["DTI", "DTA"], help="task type")
# ====== PATCH(1/2): extend choices to include 'fold' ======
parser.add_argument(
    "--split", default="random", type=str,
    choices=["random", "cold", "cluster", "fold"],  # <-- added 'fold'
    help="split type"
)
parser.add_argument("--dataset", required=True, type=str, help="dataset name, e.g., bindingdb, davis, kiba, human")


def _paths_for(dataset_name: str, split: str, cache_dirname: str):
    root = os.path.join("./datasets", dataset_name, split)
    csvs = {
        "train": os.path.join(root, "train.csv"),
        "val":   os.path.join(root, "val.csv"),
        "test":  os.path.join(root, "test.csv"),
    }
    cache_root = os.path.join(root, cache_dirname)
    os.makedirs(cache_root, exist_ok=True)
    caches = {
        "train": os.path.join(cache_root, "train_struct.pt"),
        "val":   os.path.join(cache_root, "val_struct.pt"),
        "test":  os.path.join(cache_root, "test_struct.pt"),
    }
    return csvs, caches


def _read_csv_if_exists(path: str):
    return pd.read_csv(path) if os.path.isfile(path) else None


@torch.no_grad()
def _build_struct_cache_one(df: pd.DataFrame, max_nodes: int):
    """
    Build graphs and integerized protein sequences offline,
    reusing LoadDataset's __getitem__ logic for consistency.
    """
    ds = LoadDataset(df.index.values, df, max_drug_nodes=max_nodes, cache_dict=None)
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


def _preprocess_always_rebuild(cfg, dataset_name: str, split: str, want_test: bool):
    """
    ALWAYS rebuild offline cache for train/val/(optional test), overwrite .pt files,
    and RETURN in-memory cache dicts. Never load from existing .pt.
    """
    csvs, caches = _paths_for(dataset_name, split, cfg.PREPROCESS.CACHE_DIRNAME)
    df_train = _read_csv_if_exists(csvs["train"])
    df_val   = _read_csv_if_exists(csvs["val"])
    df_test  = _read_csv_if_exists(csvs["test"]) if want_test else None

    if df_train is None or df_val is None:
        raise FileNotFoundError("train.csv and val.csv are required but not found.")

    # train
    tr_cache = _build_struct_cache_one(df_train, max_nodes=int(cfg.DRUG.MAX_NODES))
    torch.save(tr_cache, caches["train"])

    # val
    va_cache = _build_struct_cache_one(df_val, max_nodes=int(cfg.DRUG.MAX_NODES))
    torch.save(va_cache, caches["val"])

    # test (optional)
    if df_test is not None:
        te_cache = _build_struct_cache_one(df_test, max_nodes=int(cfg.DRUG.MAX_NODES))
        torch.save(te_cache, caches["test"])
    else:
        te_cache = None

    print("[Info] Built struct caches this run (train/val{}).".format("/test" if te_cache is not None else ""))
    return tr_cache, va_cache, te_cache


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    # Load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    # Seed and output dir
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR)

    dataset_name = str(args.dataset)
    split = str(args.split)
    task = args.task.upper()

    print(f"Using config: {args.cfg}")
    print(f"Device: {device}")

    # ====== PATCH(2/2): early branch for 5-fold CV (before any train/val reads) ======
    if split == "fold":
        from foldcv import run_5fold_cv  # local import; keeps non-fold runs untouched
        s_t = time()
        result = run_5fold_cv(cfg, dataset_name=dataset_name, task=task, output_dir=cfg.RESULT.OUTPUT_DIR)
        e_t = time()
        print(f"Total running time: {round(e_t - s_t, 2)}s")
        print(f"Result dir: {cfg.RESULT.OUTPUT_DIR}")
        return result

    # ====== legacy splits: random/cold/cluster (unchanged) ======
    eval_on_test = (task == "DTI")  # DTI: evaluate test if present

    csvs, _ = _paths_for(dataset_name, split, cfg.PREPROCESS.CACHE_DIRNAME)
    df_train = _read_csv_if_exists(csvs["train"])
    df_val   = _read_csv_if_exists(csvs["val"])
    df_test  = _read_csv_if_exists(csvs["test"]) if eval_on_test else None
    if df_train is None or df_val is None:
        raise FileNotFoundError("Missing train.csv or val.csv under dataset split path.")

    # Rebuild caches every run and keep them in memory
    tr_cache, va_cache, te_cache = _preprocess_always_rebuild(cfg, dataset_name, split, want_test=eval_on_test)

    # Build datasets/loaders from in-memory caches
    train_dataset = LoadDataset(None, cache_dict=tr_cache)
    val_dataset   = LoadDataset(None, cache_dict=va_cache)
    test_dataset  = LoadDataset(None, cache_dict=te_cache) if (eval_on_test and te_cache is not None) else None

    params = {
        "batch_size": int(cfg.SOLVER.BATCH_SIZE),
        "shuffle": True,
        "drop_last": True,
        "collate_fn": graph_collate_func,
        "num_workers": int(cfg.SOLVER.NUM_WORKERS),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, **params)

    eval_params = params.copy()
    eval_params["shuffle"] = False
    eval_params["drop_last"] = False
    val_loader = DataLoader(val_dataset, **eval_params)
    test_loader = DataLoader(test_dataset, **eval_params) if (eval_on_test and test_dataset is not None) else None

    # Init bias for DTA if needed
    init_bias = None
    if task == "DTA":
        use_norm = bool(cfg.TRAIN.REG_NORMALIZE)
        if use_norm:
            init_bias = 0.0
        else:
            tr_labels = tr_cache["labels"].numpy()
            init_bias = float(tr_labels.mean())

    # Model / optimizer / train
    model = SymfuseX(task=task, init_bias=init_bias, **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.SOLVER.LR))
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(
        model, opt, device,
        train_loader, val_loader, test_loader,
        task=task, **cfg
    )
    result = trainer.train()

    # Save model structure for record
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print(f"Result dir: {cfg.RESULT.OUTPUT_DIR}")
    return result


if __name__ == "__main__":
    s = time()
    args = parser.parse_args()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

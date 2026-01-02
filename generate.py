#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, hashlib, random
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import pandas as pd
import dgl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import SymfuseX
from dataloader import LoadDataset
from Integerization import graph_collate_func
from configs import get_cfg_defaults
from symbiosis import SYMBRIA_XFusion
from Fragment import CORES, SUBS, LINKERS

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors as rdmd
RDLogger.DisableLog("rdApp.*")

# ===================== Global Settings =====================
BASE_OUT = "/home/fbm/SymfuseX/result/generate/"
os.makedirs(BASE_OUT, exist_ok=True)

CKPT_PATH = "/home/fbm/SymfuseX/result/KI-DTA.pth"
CFG_YAML  = "/home/fbm/SymfuseX/configs/SymfuseX.yaml"
BASE_CSV  = "/home/fbm/SymfuseX/datasets/Generate-sample.csv"
SAMPLE_IDX: int = 99
REGR_CKPT_PATH = "/home/fbm/SymfuseX/result/MIX-DTI.pth"
REGR_TH = 0.835

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Deterministic Setup =====================
def set_seed(seed: int = 3407, deterministic: bool = True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


# ===================== cfg / model utilities =====================
def _cfg_get(cfg, *keys):
    x = cfg
    for k in keys:
        try:
            x = x[k] if isinstance(x, dict) else getattr(x, k)
        except Exception:
            x = x[k]
    return x


def _inject_fuser(model, cfg, device):
    d_dim = int(_cfg_get(cfg, "DRUG", "HIDDEN_LAYERS")[-1])
    t_dim = int(_cfg_get(cfg, "PROTEIN", "NUM_FILTERS")[1])
    f_hid = int(_cfg_get(cfg, "SYMBIOSIS", "HIDDEN_DIM"))
    f_hd  = int(_cfg_get(cfg, "SYMBIOSIS", "NUM_HEADS"))
    f_lyr = int(_cfg_get(cfg, "SYMBIOSIS", "NUM_LAYERS"))
    f_dr  = float(_cfg_get(cfg, "SYMBIOSIS", "DROPOUT"))
    if not hasattr(model, "fuser") or model.fuser is None:
        model.fuser = SYMBRIA_XFusion(
            drug_feat_dim=d_dim,
            target_feat_dim=t_dim,
            hidden_dim=f_hid,
            num_heads=f_hd,
            num_layers=f_lyr,
            dropout=f_dr
        ).to(device)


def _set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def _fuser_params(model):
    for n, p in model.named_parameters():
        if "fuser" in n:
            yield p


def _smart_loss(score, label_tensor):
    if score.dim() == 1:
        score = score.unsqueeze(-1)
    if score.shape[-1] >= 2:
        return torch.nn.functional.cross_entropy(score, label_tensor.long().view(-1))
    return torch.nn.functional.binary_cross_entropy_with_logits(
        score.view(-1, 1), label_tensor.float().view(-1, 1)
    )


def _print_gate(gw_tensor: torch.Tensor):
    gw = gw_tensor.detach().mean(dim=0)
    vals = [float(x.item()) for x in gw]
    print(f"[Gate] Rd_Rp={vals[0]:.4f}, Rd_Mp={vals[1]:.4f}, Md_Rp={vals[2]:.4f}, Md_Mp={vals[3]:.4f}")


def _prebuild_rule_encoder_if_needed(model, device):
    try:
        if hasattr(model, "fuser") and hasattr(model.fuser, "_ensure_rule_encoder"):
            model.fuser._ensure_rule_encoder(device=device)
    except Exception as e:
        print("[Warn] rule_encoder prebuild failed:", repr(e))


# ===================== Delta reconstruction & heatmap =====================
@torch.no_grad()
def _reconstruct_delta(fuser, v_d_feat, v_p_feat, smiles_list=None, tau: float = 0.2):
    h_d = fuser.proj_d(v_d_feat)
    h_p = fuser.proj_p(v_p_feat)
    z_raw = h_d.mean(dim=1)
    z_p   = h_p.mean(dim=1)
    if smiles_list is not None:
        fuser._ensure_rule_encoder(device=v_d_feat.device)
        rule_d = fuser._rule_encoder(smiles_list)
        z_pre = (1.0 - float(tau)) * z_raw + float(tau) * rule_d.detach()
    else:
        z_pre = z_raw
    gd = fuser.film_d(z_p)
    gamma, beta = torch.chunk(gd, 2, dim=-1)
    return (gamma - 1.0) * z_pre + beta


def _plot_delta_heatmap(delta_abs: np.ndarray, out_path: str, clip_percentile: float = 99.0):
    vmax = float(max(np.percentile(delta_abs, clip_percentile), 1e-9))
    plt.figure(figsize=(min(16, 2 + delta_abs.shape[1]/32), 4))
    im = plt.imshow(np.clip(delta_abs, 0, vmax), aspect="auto", origin="lower",
                    cmap="viridis", vmin=0, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="|Delta|")
    plt.xlabel("dimension"); plt.ylabel("sample"); plt.title("|Delta| heatmap (clipped)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


# ===================== Single-sample warmup =====================
def _warmup_fuser_on_single_sample(model, bg, p_idx, label_tensor, orig_h,
                                   smiles_arg=None, steps=200, lr=1e-3, weight_decay=1e-4, print_every=25):
    if steps <= 0:
        return None
    for p in model.parameters():
        p.requires_grad = False
    for p in _fuser_params(model):
        p.requires_grad = True

    opt = torch.optim.Adam(_fuser_params(model), lr=lr, weight_decay=weight_decay)
    model.train(False)
    _set_bn_eval(model)

    last_attn = None
    best_Mod_d = None
    best_mdrp = None
    best_step = None
    best_gstd = None
    best_bstd = None

    for step in range(1, steps + 1):
        opt.zero_grad()
        with bg.local_scope():
            bg.ndata['h'] = orig_h
            out = model(bg, p_idx, smiles_list=smiles_arg, mode="train")
            score = out[3] if isinstance(out, (list, tuple)) and len(out) >= 4 else out
            attn = model.fuser.get_last_attn()
            loss = _smart_loss(score, label_tensor)
            loss.backward()
            opt.step()

        last_attn = attn

        gstd = float(attn["film_d_gamma"].detach().std().item())
        bstd = float(attn["film_d_beta"].detach().std().item())
        gw = attn["gate_weights"].detach().mean(dim=0)
        cur_mdrp = float(gw[2].item())
        if (best_mdrp is None) or (cur_mdrp > best_mdrp):
            best_mdrp = cur_mdrp
            with torch.no_grad():
                Mod_d_cur = _forward_eval_Mod_d(model, bg, p_idx, smiles_arg, orig_h).detach()
            best_Mod_d = Mod_d_cur
            best_step = step
            best_gstd = gstd
            best_bstd = bstd

        if (print_every is not None) and (step % print_every == 0 or step == steps):
            print(f"[FiLM] step={step}/{steps} loss={float(loss):.6f} | gamma_std={gstd:.4f} beta_std={bstd:.4f}")
            _print_gate(attn["gate_weights"])

    model.eval()
    if best_Mod_d is not None:
        Mod_d_last = best_Mod_d
    else:
        Mod_d_last = _forward_eval_Mod_d(model, bg, p_idx, smiles_arg, orig_h).detach()

    if best_mdrp is not None and best_step is not None:
        print(f"[Select] use step={best_step} with Md_Rp={best_mdrp:.6f} | gamma_std={best_gstd:.4f} beta_std={best_bstd:.4f}")
    elif last_attn is not None:
        gw = last_attn["gate_weights"].detach().mean(dim=0)
        best_mdrp = float(gw[2].item())
        best_gstd = float(last_attn["film_d_gamma"].detach().std().item())
        best_bstd = float(last_attn["film_d_beta"].detach().std().item())
        print(f"[Select] use step={steps} with Md_Rp={best_mdrp:.6f} | gamma_std={best_gstd:.4f} beta_std={best_bstd:.4f}")

    return Mod_d_last


@torch.no_grad()
def _forward_eval_Mod_d(model, bg, p_idx, smiles_arg, orig_h):
    with bg.local_scope():
        bg.ndata['h'] = orig_h
        Mod_d, score, _ = model(bg, p_idx, smiles_list=smiles_arg, mode="eval")
        return Mod_d


def export_Mod_d_from_csv(pth_path: str, cfg_yaml: str, one_row_csv: str, sample_idx: int = 1,
                           use_rule_teacher: bool = True,
                           out_npy: str = os.path.join(BASE_OUT, "Mod_d.npy"),
                           warmup_steps: int = 200, warmup_lr: float = 5e-4,
                           heatmap_tau: float = 0.2, heatmap_path: str = os.path.join(BASE_OUT, "heatmap.png"),
                           seed: int = 42, deterministic: bool = True):
    set_seed(seed, deterministic)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_yaml)
    model = SymfuseX(**cfg).to(device).eval()
    _inject_fuser(model, cfg, device)
    _prebuild_rule_encoder_if_needed(model, device)

    df = pd.read_csv(one_row_csv)
    ds = LoadDataset(list_IDs=list(range(len(df))), df=df)
    g, p, y, s = ds[sample_idx]
    bg, p_idx, label, smiles_list = graph_collate_func([(g, p, y, s)])
    bg = bg.to(device)
    p_idx = p_idx.to(device)
    label = label.to(device)
    orig_h = bg.ndata['h']

    state = torch.load(pth_path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("[Warn] Strict load failed, falling back to non-strict:\n", e)
        res = model.load_state_dict(state, strict=False)
        print("  Missing keys:", res.missing_keys)
        print("  Unexpected keys:", res.unexpected_keys)
    _set_bn_eval(model)

    smiles_arg = [s] if (use_rule_teacher and isinstance(s, str) and len(s) > 0) else None
    Mod_d_best = _warmup_fuser_on_single_sample(
        model, bg, p_idx, label, orig_h,
        smiles_arg=smiles_arg, steps=warmup_steps, lr=warmup_lr, print_every=10
    )

    if Mod_d_best is None:
        Mod_d = _forward_eval_Mod_d(model, bg, p_idx, smiles_arg, orig_h)
    else:
        Mod_d = Mod_d_best
    Mod_d_np = Mod_d.detach().cpu().numpy()
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, Mod_d_np)
    print("[OK] Mod_d L1-norm:", float(np.abs(Mod_d_np).sum()))
    print("[OK] saved to:", out_npy)

    with torch.no_grad():
        with bg.local_scope():
            bg.ndata['h'] = orig_h
            v_d_feat = model.drug_extractor(bg)
            v_p_feat = model.protein_extractor(p_idx)
        delta = _reconstruct_delta(model.fuser, v_d_feat, v_p_feat, smiles_list=smiles_arg, tau=heatmap_tau)
        _plot_delta_heatmap(delta.detach().cpu().numpy(), heatmap_path)
        print("[OK] saved heatmap:", heatmap_path)


# ===================== Fragment selection and growth (generator) =====================
def rdkit_canon(smi: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except Exception:
        return None

def summarize(m: Chem.Mol) -> dict:
    return dict(
        heavy_atoms=m.GetNumHeavyAtoms(),
        rings=rdmd.CalcNumRings(m),
        arom_rings=rdmd.CalcNumAromaticRings(m),
        hetero_atoms=sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6)),
        mw=round(Descriptors.MolWt(m), 1),
        logp=round(Descriptors.MolLogP(m), 2),
        hbd=rdmd.CalcNumHBD(m),
        hba=rdmd.CalcNumHBA(m),
        tpsa=round(rdmd.CalcTPSA(m), 1),
    )

def seed_from_latent(latent_vec: np.ndarray, extra: int = 0) -> int:
    h = hashlib.sha1(latent_vec.tobytes()).hexdigest()
    base = int(h[:8], 16)
    return (base + int(extra)) & 0xFFFFFFFF


def _attachable_atoms(mol: Chem.Mol) -> List[int]:
    cands = []
    for a in mol.GetAtoms():
        if a.GetNumImplicitHs() > 0:
            cands.append(a.GetIdx())
    def score(idx):
        a = mol.GetAtomWithIdx(idx); arom = a.GetIsAromatic(); anum = a.GetAtomicNum()
        if anum == 6 and arom: return (0, idx)
        if anum == 6:         return (1, idx)
        if anum in (7, 8):    return (2, idx)
        return (3, idx)
    return sorted(cands, key=score)

def _add_single_bond(a: Chem.Mol, b: Chem.Mol, ia: int, ib: int) -> Optional[Chem.Mol]:
    combo = Chem.CombineMols(a, b); em = Chem.EditableMol(combo); offset = a.GetNumAtoms()
    try:
        em.AddBond(int(ia), int(offset + ib), Chem.rdchem.BondType.SINGLE)
        nm = em.GetMol(); Chem.SanitizeMol(nm); return nm
    except Exception:
        return None

def _attach_one(a: Chem.Mol, b: Chem.Mol, rng: random.Random) -> Optional[Chem.Mol]:
    cA = _attachable_atoms(a); cB = _attachable_atoms(b)
    if not cA or not cB: return None
    rng.shuffle(cA); rng.shuffle(cB)
    for ia in cA[:10]:
        for ib in cB[:10]:
            nm = _add_single_bond(a, b, ia, ib)
            if nm is not None: return nm
    return None

def _one_core_growth(core: Chem.Mol, subs: List[Chem.Mol], rng: random.Random, steps: int) -> Optional[Chem.Mol]:
    cur = Chem.Mol(core)
    for _ in range(max(1, steps)):
        sub = rng.choice(subs)
        tmp = _attach_one(cur, sub, rng)
        if tmp is None: return None
        cur = tmp
    return cur

def _small_two_core(coreA: Chem.Mol, coreB: Chem.Mol, linker: Chem.Mol, rng: random.Random) -> Optional[Chem.Mol]:
    mid = _attach_one(coreA, linker, rng)
    return _attach_one(mid, coreB, rng) if mid is not None else None

def _latent_targets_mid(latent: np.ndarray) -> dict:
    import math
    v = latent.ravel(); sig = lambda x: 1.0/(1.0+math.exp(-float(x)))
    x0 = sig(v[0%len(v)])
    return dict(base_steps=2 + int(x0*2))

def assemble_mid(latents: np.ndarray, need: int = 10) -> List[str]:
    rng = random.Random(20251020)
    cores = [Chem.MolFromSmiles(s) for s in CORES if Chem.MolFromSmiles(s)]
    subs  = [Chem.MolFromSmiles(s) for s in SUBS  if Chem.MolFromSmiles(s)]
    links = [Chem.MolFromSmiles(s) for s in LINKERS if Chem.MolFromSmiles(s)]
    res, seen = [], set(); tries, max_tries = 0, need*400
    while len(res) < need and tries < max_tries:
        lv = latents[tries % len(latents)]
        rng.seed(seed_from_latent(lv, extra=tries))
        tgt = _latent_targets_mid(lv)

        if rng.random() < 0.7:
            core = rng.choice(cores)
            steps = max(1, min(4, tgt["base_steps"] + rng.choice([-1,0,1])))
            nm = _one_core_growth(core, subs, rng, steps)
        else:
            coreA, coreB, linker = rng.choice(cores), rng.choice(cores), rng.choice(links)
            nm = _small_two_core(coreA, coreB, linker, rng)
            if nm is not None:
                for _ in range(rng.choice([0,1,1,2])):
                    tmp = _one_core_growth(nm, subs, rng, 1)
                    if tmp is not None: nm = tmp

        tries += 1
        if nm is None: continue

        m = Chem.Mol(nm)
        smi = Chem.MolToSmiles(m, canonical=True)
        if smi in seen: continue
        seen.add(smi); res.append(smi)
    return res


# ===================== Synthesis feasibility filters =====================
def _lazy_build_filter_catalog():
    from rdkit.Chem import FilterCatalog
    params = FilterCatalog.FilterCatalogParams()
    FC = FilterCatalog.FilterCatalogParams.FilterCatalogs
    loaded = []

    if hasattr(FC, "PAINS"):
        params.AddCatalog(getattr(FC, "PAINS")); loaded.append("PAINS")
    else:
        for sub in ("PAINS_A", "PAINS_B", "PAINS_C"):
            if hasattr(FC, sub):
                params.AddCatalog(getattr(FC, sub)); loaded.append(sub)

    for name in ("BRENK", "NIH", "ZINC"):
        if hasattr(FC, name):
            params.AddCatalog(getattr(FC, name)); loaded.append(name)

    print(f"[Synth] catalogs loaded: {', '.join(loaded) if loaded else 'none'}")
    return FilterCatalog.FilterCatalog(params)

def _violates_catalog(m: Chem.Mol, catalog) -> bool:
    try:
        if catalog is None:
            return False
        return catalog.HasMatch(m)
    except Exception:
        return False

def _try_sa_score(m: Chem.Mol) -> float:
    try:
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem import SA_Score
        return float(SA_Score.sascorer.calculateScore(m))
    except Exception:
        return -1.0

def apply_synthesis_filters(smiles_list: List[str], sa_thr: float = 6.0) -> List[str]:
    try:
        from rdkit.Chem import FilterCatalog
        catalog = _lazy_build_filter_catalog()
    except Exception:
        catalog = None
        print("[Synth] FilterCatalog not available; skipping PAINS/BRENK/NIH/ZINC checks.")

    kept = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if _violates_catalog(m, catalog):
            continue
        sa = _try_sa_score(m)
        if sa < 0.0 or sa <= sa_thr:
            kept.append(s)
    print(f"[OK] kept {len(kept)}/{len(smiles_list)} after PAINS/Brenk/Nih/Zinc")
    return kept


# ===================== Analyze =====================
def analyze_main(Mod_d_paths: Optional[List[str]] = None,
                 target_n: int = 10,
                 summary_txt_path: str = os.path.join(BASE_OUT, "summary.txt")):
    latent_path = (Mod_d_paths or [os.path.join(BASE_OUT, "Mod_d.npy")])[0]
    lat = np.load(latent_path, allow_pickle=True)
    if lat.ndim == 1:
        lat = lat.reshape(1, -1)
    elif lat.ndim > 2:
        lat = lat.reshape(lat.shape[0], -1)

    smiles = assemble_mid(lat, need=target_n)

    uniq, seen, lines = [], set(), []
    for s in smiles:
        can = rdkit_canon(s)
        if can and (can not in seen):
            uniq.append(can); seen.add(can)
            m = Chem.MolFromSmiles(can)
            lines.append(f"{can} | {summarize(m)}")

    os.makedirs(os.path.dirname(summary_txt_path), exist_ok=True)
    with open(summary_txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] Done! Obtain high-quality drug candidate molecules")
    return uniq


# ===================== Scoring (common loader) =====================
def _infer_smiles_col(df: pd.DataFrame) -> str:
    for c in ["SMILES", "smiles", "Smiles", "drug", "Drug", "molecule", "MOL", "mol"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise KeyError("SMILES column not found")

def _reload_model(cfg_yaml: str, ckpt_path: str):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_yaml)
    model = SymfuseX(**cfg).to(device).eval()
    _inject_fuser(model, cfg, device)
    _prebuild_rule_encoder_if_needed(model, device)
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("[Warn] Strict load failed, falling back to non-strict:\n", e)
        res = model.load_state_dict(state, strict=False)
        print("  Missing keys:", res.missing_keys)
        print("  Unexpected keys:", res.unexpected_keys)
    _set_bn_eval(model)
    return model


# ===================== DTI pre-screen =====================
@torch.no_grad()
def screen_smiles_by_DTI(smiles_list: List[str], base_csv: str, sample_idx: int,
                                cfg_yaml: str, ckpt_path: str, th: float,
                                out_scores_txt: str = os.path.join(BASE_OUT, "scores_generated_smiles_DTI.txt")) -> List[str]:
    if not smiles_list:
        return []
    df0 = pd.read_csv(base_csv)
    row = df0.iloc[sample_idx].copy()
    smiles_col = _infer_smiles_col(df0)

    rows = [row.to_dict() for _ in smiles_list]
    new_df = pd.DataFrame(rows)
    new_df[smiles_col] = smiles_list
    ds_new = LoadDataset(list_IDs=list(range(len(new_df))), df=new_df)

    model = _reload_model(cfg_yaml, ckpt_path)

    scores = []
    for i in range(len(new_df)):
        g, p, y, s = ds_new[i]
        bg, p_idx, _, _ = graph_collate_func([(g, p, y, s)])
        bg = bg.to(device)
        p_idx = p_idx.to(device)
        with bg.local_scope():
            _, score, _ = model(bg, p_idx, smiles_list=[smiles_list[i]], mode="eval")
            if (score.dim() == 1) or (score.shape[-1] == 1):
                val = torch.sigmoid(score.view(-1)).mean().item()
            else:
                val = torch.softmax(score, dim=-1)[..., 1].mean().item()
            scores.append(float(val))

    os.makedirs(os.path.dirname(out_scores_txt), exist_ok=True)
    with open(out_scores_txt, "w") as f:
        f.write("smiles\tscore_DTI\n")
        for s, sc in zip(smiles_list, scores):
            f.write(f"{s}\t{sc:.6f}\n")
    print(f"[DTI] saved to {out_scores_txt}")

    kept = [s for s, sc in zip(smiles_list, scores) if sc > th]
    print(f"[OK] kept {len(kept)}/{len(smiles_list)} with TH>{th}")
    return kept

# ===================== Classification scoring =====================
@torch.no_grad()
def score_smiles_list(smiles_list: List[str], base_csv: str, sample_idx: int,
                      cfg_yaml: str, ckpt_path: str,
                      out_scores_txt: str = os.path.join(BASE_OUT, "scores_generated_smiles.txt"),
                      best_txt_path: str = os.path.join(BASE_OUT, "best_smiles.txt"),
                      best_img_path: str = os.path.join(BASE_OUT, "best_smiles.png"),
                      print_orig_pair_score: bool = True):
    df0 = pd.read_csv(base_csv)
    row = df0.iloc[sample_idx].copy()
    smiles_col = _infer_smiles_col(df0)
    orig_smiles = str(row[smiles_col])

    rows = [row.to_dict() for _ in smiles_list]
    new_df = pd.DataFrame(rows)
    new_df[smiles_col] = smiles_list
    ds_new = LoadDataset(list_IDs=list(range(len(new_df))), df=new_df)

    model = _reload_model(cfg_yaml, ckpt_path)

    scores = []
    for i in range(len(new_df)):
        g, p, y, s = ds_new[i]
        bg, p_idx, _, _ = graph_collate_func([(g, p, y, s)])
        bg = bg.to(device)
        p_idx = p_idx.to(device)
        with bg.local_scope():
            _, score, _ = model(bg, p_idx, smiles_list=[smiles_list[i]], mode="eval")
            prob = torch.sigmoid(score.view(-1)).mean().item() if (score.dim()==1 or score.shape[-1]==1) \
                   else torch.softmax(score, dim=-1)[..., 1].mean().item()
            scores.append(float(prob))

    best_idx = int(np.argmax(scores))
    best_smi = smiles_list[best_idx]

    qed_raw = None
    qed_best = None
    try:
        from rdkit.Chem.QED import qed
        m_raw = Chem.MolFromSmiles(orig_smiles) if isinstance(orig_smiles, str) else None
        m_best = Chem.MolFromSmiles(best_smi) if isinstance(best_smi, str) else None
        if m_raw is not None:
            props_raw = summarize(m_raw)
            qed_raw = float(qed(m_raw))
            print(f"[Props][Raw] {orig_smiles} | {props_raw} | QED={qed_raw:.3f}")
        else:
            print(f"[Props][Raw] {orig_smiles} | invalid molecule")
        if m_best is not None:
            props_best = summarize(m_best)
            qed_best = float(qed(m_best))
            print(f"[Props][Best] {best_smi} | {props_best} | QED={qed_best:.3f}")
        else:
            print(f"[Props][Best] {best_smi} | invalid molecule")
    except Exception:
        pass


    prob0 = None
    if print_orig_pair_score:
        ds_orig = LoadDataset(list_IDs=list(range(len(df0))), df=df0)
        g0, p0, y0, s0 = ds_orig[sample_idx]
        bg0, p0_idx, _, _ = graph_collate_func([(g0, p0, y0, s0)])
        bg0 = bg0.to(device)
        p0_idx = p0_idx.to(device)
        with bg0.local_scope():
            _, score0, _ = model(bg0, p0_idx, smiles_list=[s0], mode="eval")
            prob0 = torch.sigmoid(score0.view(-1)).mean().item() if (score0.dim()==1 or score0.shape[-1]==1) \
                    else torch.softmax(score0, dim=-1)[..., 1].mean().item()

    print(f"[Best] {best_smi} | score={scores[best_idx]:.6f}")
    if print_orig_pair_score and prob0 is not None:
        print(f"[Raw] {orig_smiles} | score={prob0:.6f}")
        score_diff = scores[best_idx] - prob0
        percent_increase = (score_diff / prob0) * 100 if prob0 != 0 else float('inf')
        if (qed_raw is not None) and (qed_best is not None):
            qed_diff = qed_best - qed_raw
            qed_percent = (qed_diff / qed_raw * 100) if qed_raw > 0 else float('inf')
            print(f"[NOTE] The affinity score of the new drug to the target is {score_diff:.6f} higher than that of the raw drug, an increase of {percent_increase:.2f}%.") 
            print(f"[NOTE] QED improves by {qed_diff:.3f} (from {qed_raw:.3f} to {qed_best:.3f}, increase {qed_percent:.2f}%).")

    os.makedirs(os.path.dirname(out_scores_txt), exist_ok=True)
    with open(out_scores_txt, "w") as f:
        f.write("smiles\tscore\n")
        for s, sc in zip(smiles_list, scores):
            f.write(f"{s}\t{sc:.6f}\n")
    print(f"[OK] Scores saved to {out_scores_txt}")

    with open(best_txt_path, "w") as f:
        f.write(best_smi + "\n")
    print(f"[Best] saved to {best_txt_path}")

    try:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(best_smi)
        if mol:
            Draw.MolToFile(mol, best_img_path, size=(640, 480))
            print(f"[Best] molecule image saved to {best_img_path}")
    except Exception:
        pass

    return best_smi, scores[best_idx]


# ===================== Orchestrator =====================
def run_pipeline(ckpt_path=CKPT_PATH,
                 cfg_yaml=CFG_YAML,
                 base_csv=BASE_CSV,
                 sample_idx=SAMPLE_IDX):
    print("[Main] Step1: Start single sample symbiotic modulation......")
    export_Mod_d_from_csv(
        pth_path=ckpt_path, cfg_yaml=cfg_yaml, one_row_csv=base_csv, sample_idx=sample_idx,
        use_rule_teacher=True, out_npy=os.path.join(BASE_OUT, "Mod_d.npy"),
        warmup_steps=200, warmup_lr=1e-4,
        heatmap_tau=0.2, heatmap_path=os.path.join(BASE_OUT, "heatmap.png"),
        seed=42, deterministic=True
    )

    print("[Main] Step2: Latent Target-Preference Guided Molecule Generation......")
    smiles = analyze_main(
        Mod_d_paths=[os.path.join(BASE_OUT, "Mod_d.npy")],
        target_n=100000,
        summary_txt_path=os.path.join(BASE_OUT, "summary.txt")
    )
    if not smiles:
        raise RuntimeError("No SMILES generated.")

    print("[Main] Step3: Synthesis feasibility filtering......")
    smiles = apply_synthesis_filters(smiles, sa_thr=6.0)
    if not smiles:
        raise RuntimeError("No SMILES after synthesis feasibility filtering.")

    print("[Main] Step4: DTI pre-screening......")
    smiles = screen_smiles_by_DTI(
        smiles_list=smiles,
        base_csv=base_csv,
        sample_idx=sample_idx,
        cfg_yaml=cfg_yaml,
        ckpt_path=REGR_CKPT_PATH,
        th=REGR_TH,
        out_scores_txt=os.path.join(BASE_OUT, "scores_generated_smiles_DTI.txt"),
    )
    if not smiles:
        raise RuntimeError("No SMILES passed DTI threshold.")

    print("[Main] Step5: Biochemical attribute scoring......")
    _ = score_smiles_list(
        smiles_list=smiles,
        base_csv=base_csv,
        sample_idx=sample_idx,
        cfg_yaml=cfg_yaml,
        ckpt_path=ckpt_path,
        out_scores_txt=os.path.join(BASE_OUT, "scores_generated_smiles.txt"),
        best_txt_path=os.path.join(BASE_OUT, "best_smiles.txt"),
        best_img_path=os.path.join(BASE_OUT, "best_smiles.png"),
        print_orig_pair_score=True
    )


if __name__ == "__main__":
    print("[Main] The program for generating high-affinity drug molecules has begun!")
    run_pipeline()

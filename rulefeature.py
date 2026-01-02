# rulefeature.py
# -*- coding: utf-8 -*-
from typing import List, Optional
import torch
import torch.nn as nn

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False


class _LinearReLULinear(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RuleFeature(nn.Module):

    def __init__(
        self,
        out_dim: int,
        nBits: int = 1024,
        radius: int = 2,
        mid_dim: int = 512,
        dropout: float = 0.00,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.nBits = int(nBits)
        self.radius = int(radius)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.project = _LinearReLULinear(self.nBits, mid_dim=mid_dim, out_dim=self.out_dim, dropout=dropout).to(self.device)
        self.post_ln = nn.LayerNorm(self.out_dim)
        self.scale = nn.Parameter(torch.ones(1, self.out_dim))

        if not _HAS_RDKIT:
            vocab = [chr(i) for i in range(32, 127)]
            self._stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  
            self._char_emb = nn.Embedding(len(vocab) + 1, self.nBits, padding_idx=0).to(self.device)

    @torch.no_grad()
    def _ecfp_bits(self, smiles_list: List[str]) -> torch.Tensor:

        B = len(smiles_list)
        X = torch.zeros((B, self.nBits), dtype=torch.float32, device=self.device)

        if _HAS_RDKIT:
            for i, smi in enumerate(smiles_list):
                smi = (smi or "").strip()
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                bv = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
                for bit in bv.GetOnBits():
                    if 0 <= bit < self.nBits:
                        X[i, bit] = 1.0
        else:
            for i, smi in enumerate(smiles_list):
                s = (smi or "").strip()
                if not s:
                    continue
                idxs = [self._stoi.get(ch, 0) for ch in s]
                t = torch.tensor(idxs, dtype=torch.long, device=self.device)
                emb = self._char_emb(t)  
                X[i] = emb.mean(dim=0)
        return X

    @torch.no_grad()
    def forward(self, smiles_list: List[str]) -> torch.Tensor:

        bits = self._ecfp_bits(smiles_list)            
        rule = self.project(bits)                     
        rule = self.post_ln(rule) * self.scale        
        return rule

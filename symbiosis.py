# symbiosis.py
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from rulefeature import RuleFeature


class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(dims[i], dims[i + 1]), dim=None))
            if act:
                layers.append(getattr(nn, act)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def _zero_last_linear(module: nn.Sequential):
    for m in reversed(module):
        if isinstance(m, nn.Linear) or (
                hasattr(m, 'weight') and isinstance(m, nn.modules.linear.NonDynamicallyQuantizableLinear)):
            nn.init.zeros_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
            break


class BilinearVecFuse(nn.Module):
    def __init__(self, in_v: int, in_q: int, h_dim: int, k: int = 4, dropout: float = 0.1, act='ReLU'):
        super().__init__()
        self.h_dim = h_dim
        self.k = max(1, int(k))
        self.v_net = FCNet([in_v, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([in_q, h_dim * self.k], act=act, dropout=dropout)
        self.bn = nn.BatchNorm1d(h_dim)
        if self.k > 1:
            self.pool = nn.AvgPool1d(self.k, stride=self.k)

    def forward(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        x = v_ * q_
        if self.k > 1:
            x = x.unsqueeze(1)
            x = self.pool(x).squeeze(1) * self.k
        x = self.bn(x)
        return x


class SYMBRIA_XFusion(nn.Module):
    def __init__(self,
                 drug_feat_dim: int,
                 target_feat_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.H = int(hidden_dim)
        self.k = max(1, int(num_heads))
        self.L = int(num_layers)
        self.drop = nn.Dropout(dropout)

        self.proj_d = nn.Linear(drug_feat_dim, self.H)
        self.proj_p = nn.Linear(target_feat_dim, self.H)

        self.film_d = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Linear(self.H, 2 * self.H)
        )
        self.film_p = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Linear(self.H, 2 * self.H)
        )
        _zero_last_linear(self.film_d[-1:])
        _zero_last_linear(self.film_p[-1:])

        self.fuse_layers = nn.ModuleList([
            BilinearVecFuse(self.H, self.H, self.H, k=self.k, dropout=dropout, act='ReLU')
            for _ in range(self.L)
        ])

        self.gate = nn.Sequential(
            nn.Linear(4 * self.H, self.H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.H, 4)
        )

        self.post_ln = nn.LayerNorm(self.H)
        self.post_ffn = nn.Sequential(
            nn.Linear(self.H, self.H * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.H * 2, self.H),
        )

        self._rule_encoder: Optional[RuleFeature] = None
        self._last_attn: Dict[str, torch.Tensor] = {}

    def _ensure_rule_encoder(self, device):
        if self._rule_encoder is None:
            self._rule_encoder = RuleFeature(out_dim=self.H, device=device).to(device)

    def get_last_attn(self) -> Dict[str, torch.Tensor]:
        return self._last_attn

    @staticmethod
    def _mean_pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def _make_fake_vectors(self, z_d: torch.Tensor, z_p: torch.Tensor):
        gd = self.film_d(z_p)
        gp = self.film_p(z_d)
        gamma_d, beta_d = torch.chunk(gd, 2, dim=-1)
        gamma_p, beta_p = torch.chunk(gp, 2, dim=-1)
        Mod_d = gamma_d * z_d + beta_d
        Mod_p = gamma_p * z_p + beta_p
        return Mod_d, Mod_p, gamma_d, beta_d, gamma_p, beta_p

    def forward(
            self,
            drug_seq: torch.Tensor,
            target_seq: torch.Tensor,
            smiles_list: Optional[List[str]] = None,
            tau: float = 0.2,
    ) -> torch.Tensor:

        # 1) Projection + mean-pool
        h_d = self.proj_d(drug_seq)
        h_p = self.proj_p(target_seq)
        z_d = self._mean_pool(h_d)
        Raw_p = self._mean_pool(h_p)

        # 2) Rule vector injection
        if smiles_list is not None:
            self._ensure_rule_encoder(device=drug_seq.device)
            with torch.no_grad():
                rule_d = self._rule_encoder(smiles_list)
            Raw_d = (1.0 - float(tau)) * z_d + float(tau) * rule_d.detach()
        else:
            rule_d = None
            Raw_d = z_d

        # 3) Generate modified vectors
        Mod_d, Mod_p, g_d, b_d, g_p, b_p = self._make_fake_vectors(Raw_d, Raw_p)

        # 4) Four-way fusion
        fused_sum = 0.0
        pair_logits_list: List[torch.Tensor] = []

        gate_w = F.softmax(
            self.gate(torch.cat([Raw_d, Mod_d, Raw_p, Mod_p], dim=-1)),
            dim=-1
        )

        for layer in self.fuse_layers:
            Rd_Rp = layer(Raw_d, Raw_p)
            Rd_Mp = layer(Raw_d, Mod_p)
            Md_Rp = layer(Mod_d, Raw_p)
            Md_Mp = layer(Mod_d, Mod_p)
            pair_logits = torch.stack([Rd_Rp, Rd_Mp, Md_Rp, Md_Mp], dim=1)
            fused_layer = torch.einsum('bf,bfh->bh', gate_w, pair_logits)
            fused_sum = fused_sum + self.drop(fused_layer)
            pair_logits_list.append(pair_logits)

        fused = fused_sum / float(self.L)

        # 5) Post-processing
        fused = self.post_ln(fused)
        fused = fused + self.post_ffn(fused)
        fused = self.post_ln(fused)

        # 6) Explanation cache
        if len(pair_logits_list) > 0:
            last_pairs = pair_logits_list[-1]
        else:
            last_pairs = torch.stack([Rd_Rp, Rd_Mp, Md_Rp, Md_Mp], dim=1)

        self._last_attn = {
            "film_d_gamma": g_d, "film_d_beta": b_d,
            "film_p_gamma": g_p, "film_p_beta": b_p,
            "gate_weights": gate_w,
            "Rd_Rp": last_pairs[:, 0, :],
            "Rd_Mp": last_pairs[:, 1, :],
            "Md_Rp": last_pairs[:, 2, :],
            "Md_Mp": last_pairs[:, 3, :],
        }
        if rule_d is not None:
            self._last_attn["rule_d"] = rule_d
            self._last_attn["tau"] = torch.tensor(float(tau), device=fused.device)

        return fused, Mod_d

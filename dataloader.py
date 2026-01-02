# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from functools import partial
from dgllife.utils import (
    smiles_to_bigraph,
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
)
from Integerization import integer_label_protein

class LoadDataset(data.Dataset):
    """
    Two modes:
      1) Online: build DGL graph from SMILES on the fly.
      2) Offline: read from cache dict built by main.py.
    """
    def __init__(self, list_IDs, df=None, max_drug_nodes=290, cache_dict=None):
        self.use_cache = cache_dict is not None
        self.cache = cache_dict

        if not self.use_cache:
            self.list_IDs = list_IDs
            self.df = df
            self.max_drug_nodes = int(max_drug_nodes)
            self.atom_featurizer = CanonicalAtomFeaturizer()
            # IMPORTANT: add self-loop features because we will add self-loop edges.
            self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
            self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        else:
            self.graphs   = self.cache["graphs"]       # list of DGLGraph
            self.proteins = self.cache["proteins"]     # list of LongTensor
            self.labels   = self.cache["labels"]       # FloatTensor [N]
            self.smiles   = self.cache.get("smiles", [])
            self.seqs     = self.cache.get("seqs", [])

    def __len__(self):
        if self.use_cache:
            return len(self.graphs)
        return len(self.list_IDs)

    def __getitem__(self, index):
        if self.use_cache:
            g = self.graphs[index]
            p = self.proteins[index]
            y = float(self.labels[index])
            s = self.smiles[index] if self.smiles else ""
            return g, p, y, s

        # Online path
        index = self.list_IDs[index]
        row = self.df.iloc[index]
        smiles_str = row["SMILES"]
        g = self.fc(
            smiles=smiles_str,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )

        # Add a virtual-node indicator channel at the last dim.
        node_feats = g.ndata["h"]                # [N_real, A_dim]
        n_real = node_feats.shape[0]
        A_dim = node_feats.shape[1]
        n_virtual = self.max_drug_nodes - n_real
        if n_virtual < 0:
            raise ValueError(
                f"Graph has {n_real} nodes, exceeds DRUG.MAX_NODES={self.max_drug_nodes}."
            )

        # Mark real nodes with indicator 0
        real_indicator = torch.zeros((n_real, 1), dtype=node_feats.dtype)
        g.ndata["h"] = torch.cat((node_feats, real_indicator), dim=1)

        # Add virtual nodes if needed: zeros in original dims + indicator 1
        if n_virtual > 0:
            virtual_feat = torch.cat(
                (
                    torch.zeros((n_virtual, A_dim), dtype=node_feats.dtype),
                    torch.ones((n_virtual, 1), dtype=node_feats.dtype),
                ),
                dim=1,
            )
            g.add_nodes(n_virtual, {"h": virtual_feat})

        p_seq = row["Protein"]
        p_idx = integer_label_protein(p_seq)  # LongTensor (seq indices)
        y = float(row["Y"])
        return g, p_idx, y, smiles_str

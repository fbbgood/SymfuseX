# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv
from symbiosis import SYMBRIA_XFusion


class SymfuseX(nn.Module):
    def __init__(self, **config):
        super(SymfuseX, self).__init__()

        # DRUG
        d_node_in = config["DRUG"]["NODE_IN_FEATS"]
        d_emb     = config["DRUG"]["NODE_IN_EMBEDDING"]
        d_hid     = config["DRUG"]["HIDDEN_LAYERS"]
        d_pad     = config["DRUG"]["PADDING"]

        # PROTEIN
        p_emb     = config["PROTEIN"]["EMBEDDING_DIM"]
        p_filters = config["PROTEIN"]["NUM_FILTERS"]
        p_kernel  = config["PROTEIN"]["KERNEL_SIZE"]
        p_pad     = config["PROTEIN"]["PADDING"]

        # SYMBIOSIS
        f_hidden  = config["SYMBIOSIS"]["HIDDEN_DIM"]
        f_heads   = config["SYMBIOSIS"]["NUM_HEADS"]
        f_layers  = config["SYMBIOSIS"]["NUM_LAYERS"]
        f_drop    = config["SYMBIOSIS"]["DROPOUT"]

        # DECODER
        mlp_in    = config["DECODER"]["IN_DIM"]   
        mlp_h     = config["DECODER"]["HIDDEN_DIM"]
        mlp_out   = config["DECODER"]["OUT_DIM"]
        binary    = config["DECODER"]["BINARY"]

        self.drug_extractor    = DrugGIN(d_node_in, d_emb, d_pad, d_hid) 
        self.protein_extractor = ProteinRS(p_emb, p_filters, p_kernel, p_pad) 

      
        self.fuser: SYMBRIA_XFusion = None
        self._f_hidden = f_hidden
        self._f_heads  = f_heads
        self._f_layers = f_layers
        self._f_drop   = f_drop

        self.mlp_classifier = MLPDecoder(f_hidden, mlp_h, mlp_out, binary=binary)

    def _maybe_build_fuser(self, d_dim: int, t_dim: int, device):
        if self.fuser is None:
            self.fuser = SYMBRIA_XFusion(
                drug_feat_dim=d_dim,
                target_feat_dim=t_dim,
                hidden_dim=self._f_hidden,
                num_heads=self._f_heads,
                num_layers=self._f_layers,
                dropout=self._f_drop
            ).to(device)

    def forward(self, bg_d, v_p, smiles_list=None, seq_list=None, mode="train"):
        
        v_d     = self.drug_extractor(bg_d)   
        v_p_emb = self.protein_extractor(v_p) 

        d_dim = v_d.size(-1)
        t_dim = v_p_emb.size(-1)
        self._maybe_build_fuser(d_dim, t_dim, v_d.device)

        fused_vec, Mod_d = self.fuser(v_d, v_p_emb, smiles_list=smiles_list, tau=0.1)

        score = self.mlp_classifier(fused_vec) 

        if mode == "train":
            f_stub = fused_vec.unsqueeze(1)
            return v_d, v_p_emb, f_stub, score
        else:
            att_stub = None
            return Mod_d, score, att_stub


# ====== Backbone blocks ======
class DrugGIN(nn.Module):
    def __init__(self, in_feats, dim_embedding, padding, hidden_feats):
        super(DrugGIN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0.0)
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i, out_dim in enumerate(hidden_feats):
            in_dim = dim_embedding if i == 0 else hidden_feats[i - 1]
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            self.gin_layers.append(GINConv(mlp, learn_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
        self.jk_lin = nn.Linear(sum(hidden_feats), hidden_feats[-1])
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h') 
        h = self.init_transform(node_feats)
        h = F.relu(h)

        layer_feats = []
        for gin, bn in zip(self.gin_layers, self.batch_norms):
            h = gin(batch_graph, h)
            h = bn(h)
            h = F.relu(h)
            layer_feats.append(h)

        h_cat = torch.cat(layer_feats, dim=1)  
        h = self.jk_lin(h_cat)                 

        batch_size = batch_graph.batch_size
        num_nodes = h.size(0) // batch_size
        node_feats = h.view(batch_size, num_nodes, self.output_feats) 
        return node_feats


class ProteinRS(nn.Module):

    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinRS, self).__init__()
        # Embedding
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)

      
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

        c1 = num_filters[0]
        k1 = kernel_size[0]
        p1 = k1 // 2
        self.conv1_1 = nn.Conv1d(embedding_dim, c1, k1, stride=1, padding=p1)
        self.bn1_1   = nn.BatchNorm1d(c1)
        self.conv1_2 = nn.Conv1d(c1, c1, k1, stride=1, padding=p1)
        self.bn1_2   = nn.BatchNorm1d(c1)
        self.conv1_3 = nn.Conv1d(c1, c1, k1, stride=1, padding=p1)
        self.bn1_3   = nn.BatchNorm1d(c1)
        self.match1  = nn.Conv1d(embedding_dim, c1, kernel_size=1, padding=0)

        c2 = num_filters[1]
        k2 = kernel_size[1]
        p2 = k2 // 2
        self.conv2_1 = nn.Conv1d(c1, c2, k2, stride=1, padding=p2)
        self.bn2_1   = nn.BatchNorm1d(c2)
        self.conv2_2 = nn.Conv1d(c2, c2, k2, stride=1, padding=p2)
        self.bn2_2   = nn.BatchNorm1d(c2)
        self.conv2_3 = nn.Conv1d(c2, c2, k2, stride=1, padding=p2)
        self.bn2_3   = nn.BatchNorm1d(c2)
        self.match2  = nn.Conv1d(c1, c2, kernel_size=1, padding=0)

    def _resblock1(self, x):
        residual = x
        out = self.conv1_1(x); out = self.bn1_1(out); out = self.relu(out); out = self.dropout(out)
        out = self.conv1_2(out); out = self.bn1_2(out); out = self.relu(out); out = self.dropout(out)
        out = self.conv1_3(out); out = self.bn1_3(out)

        residual = self.match1(residual)
        if residual.shape[-1] != out.shape[-1]:
            diff = out.shape[-1] - residual.shape[-1]
            residual = F.pad(residual, (0, diff))

        out = self.relu(out + residual)
        return out

    def _resblock2(self, x):
        residual = x
        out = self.conv2_1(x); out = self.bn2_1(out); out = self.relu(out); out = self.dropout(out)
        out = self.conv2_2(out); out = self.bn2_2(out); out = self.relu(out); out = self.dropout(out)
        out = self.conv2_3(out); out = self.bn2_3(out)

        residual = self.match2(residual)
        if residual.shape[-1] != out.shape[-1]:
            diff = out.shape[-1] - residual.shape[-1]
            residual = F.pad(residual, (0, diff))

        out = self.relu(out + residual)
        return out

    def forward(self, v):
        v = self.embedding(v.long())   # (B, L, E)
        v = v.transpose(2, 1)          # (B, E, L)
        v = self._resblock1(v)
        v = self._resblock2(v)
        v = v.view(v.size(0), v.size(2), -1)  # (B, L, C)
        return v



class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


def binary_cross_entropy(pred_output, labels):
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = nn.BCELoss()(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    y_hat = class_output.max(1)[1]
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

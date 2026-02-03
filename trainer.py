# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, precision_score, mean_squared_error, r2_score,
    f1_score
)

from modules import binary_cross_entropy, cross_entropy_logits, entropy_logits

try:
    from yacs.config import CfgNode
except Exception:
    CfgNode = None

def _to_pydict(obj):
    if CfgNode is not None and isinstance(obj, CfgNode):
        return {k: _to_pydict(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _to_pydict(v) for k, v in obj.items()}
    return obj


def calculate_ci(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = len(y_true)
    concordant, comparable = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            comparable += 1
            if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                concordant += 1
    return concordant / comparable if comparable > 0 else 0.5


def calculate_rm2(y_true, y_pred):
    try:
        r2 = r2_score(y_true, y_pred)
        if np.isnan(r2):
            r2 = 0.0
    except Exception:
        r2 = 0.0
    eps = 1e-12
    denom = np.sum(np.array(y_pred) ** 2)
    if denom <= eps:
        return float(r2 * (1 - np.sqrt(max(0.0, abs(r2)))))

    k = np.sum(np.array(y_true) * np.array(y_pred)) / (denom + eps)
    y_pred_0 = k * np.array(y_pred)
    try:
        r0_squared = r2_score(y_true, y_pred_0)
        if np.isnan(r0_squared):
            r0_squared = 0.0
    except Exception:
        r0_squared = 0.0

    inner = max(0.0, abs(r2 - r0_squared))
    rm2 = r2 * (1 - np.sqrt(inner))
    return float(rm2) if np.isfinite(rm2) else 0.0


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, task="DTI", **config):
        cfgd = _to_pydict(config)

        self.model = model
        self.optim = optim
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.epochs = cfgd.get("SOLVER", {}).get("MAX_EPOCH", 1)
        self.current_epoch = 0
        self.step = 0
        self.batch_size = cfgd.get("SOLVER", {}).get("BATCH_SIZE", 32)

        self.n_class = cfgd.get("DECODER", {}).get("BINARY", 1)
        self.task = str(task).upper()
        self.is_regression = (self.task == "DTA")

        train_cfg = cfgd.get("TRAIN", {})
        self.reg_normalize = bool(train_cfg.get("REG_NORMALIZE", True if self.is_regression else False))
        self.reg_loss_name = str(train_cfg.get("REG_LOSS", "MSE" if self.is_regression else "SmoothL1")).upper()

        if self.is_regression:
            self.criterion = torch.nn.MSELoss() if self.reg_loss_name == "MSE" else torch.nn.SmoothL1Loss()
            self.best_state_dict = None
            self.best_epoch = None
            self.best_mse = float('inf')
            self.best_ci = 0.0
            self.best_rm2 = 0.0

            self.reg_mean = None
            self.reg_std = None
            if self.reg_normalize:
                try:
                    ds = self.train_dataloader.dataset
                    labels = [float(ds[i][2]) for i in range(len(ds))]
                    labels = np.asarray(labels, dtype=float)
                    self.reg_mean = float(labels.mean())
                    self.reg_std = float(labels.std() + 1e-8)
                except Exception:
                    self.reg_normalize = False
                    self.reg_mean = 0.0
                    self.reg_std = 1.0
        else:
            self.criterion = None
            self.best_state_dict = None
            self.best_epoch = None
            self.best_auroc = 0.0

        self.config = cfgd
        self.output_dir = cfgd.get("RESULT", {}).get("OUTPUT_DIR", "./result")
        os.makedirs(self.output_dir, exist_ok=True)
        self.clip_grad_norm = cfgd.get("TRAIN", {}).get("CLIP_GRAD_NORM", None)

        self.test_table = PrettyTable(["# Best Epoch", "Split", "Metrics"])

        if self.is_regression:
            self.val_table = PrettyTable(["Epoch", "Val_MSE", "Val_CI", "Val_RM2", "Val_Loss"])
        else:
            self.val_table = PrettyTable(["Epoch", "Val_AUROC", "Val_AUPRC", "Val_Loss"])

    def _normalize_label(self, label_tensor):
        if not self.reg_normalize:
            return label_tensor
        return (label_tensor - self.reg_mean) / self.reg_std

    def _denormalize_pred(self, pred_tensor):
        if not self.reg_normalize:
            return pred_tensor
        return pred_tensor * self.reg_std + self.reg_mean

    def train(self):
        for _ in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()

            if not self.is_regression:
                auroc, auprc, val_loss = self.test(dataloader="val")
                print(f'[Val] Epoch {self.current_epoch}: loss {val_loss:.4f}, AUROC {auroc:.4f}, AUPRC {auprc:.4f}')
                self.val_table.add_row([
                    self.current_epoch,
                    f"{auroc:.4f}",
                    f"{auprc:.4f}",
                    f"{val_loss:.4f}",
                ])
                if auroc >= getattr(self, "best_auroc", 0.0):
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_auroc = auroc
                    self.best_epoch = self.current_epoch
            else:
                val_mse, val_ci, val_rm2, val_loss = self.test(dataloader="val")
                print(
                    f'[Val] Epoch {self.current_epoch}: loss {val_loss:.4f}, '
                    f'MSE {val_mse:.4f}, CI {val_ci:.4f}, RM2 {val_rm2:.4f}'
                )
                self.val_table.add_row([
                    self.current_epoch,
                    f"{val_mse:.4f}",
                    f"{val_ci:.4f}",
                    f"{val_rm2:.4f}",
                    f"{val_loss:.4f}",
                ])
                if val_mse <= getattr(self, "best_mse", float('inf')):
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_mse = val_mse
                    self.best_ci = val_ci
                    self.best_rm2 = val_rm2
                    self.best_epoch = self.current_epoch

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "val_markdowntable.txt"), "w") as fp:
            fp.write(self.val_table.get_string())

        if self.is_regression:
            metrics = {
                "Split": "VAL",
                "Best_epoch": self.best_epoch,
                "MSE": getattr(self, "best_mse", 0.0),
                "CI": getattr(self, "best_ci", 0.0),
                "RM2": getattr(self, "best_rm2", 0.0),
            }
            brief = (
                f"MSE={metrics['MSE']:.4f}, CI={metrics['CI']:.4f}, RM2={metrics['RM2']:.4f}"
            )
            print(f'[Best @ epoch {self.best_epoch}] VAL: {brief}')
            self.test_table.add_row([f"epoch {self.best_epoch}", "VAL", brief])
            self.save_result(metrics)
            return metrics

        if getattr(self, "best_state_dict", None) is not None:
            self._temp_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_state_dict)

        if self.test_dataloader is None:
            auroc, auprc, val_loss = self.test(dataloader="val")
            brief = f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}, LOSS={val_loss:.4f}"
            print(f'[Best @ epoch {self.best_epoch}] VAL(as final): {brief}')
            metrics = {
                "Split": "VAL",
                "Best_epoch": self.best_epoch,
                "AUROC": auroc, "AUPRC": auprc, "Test_loss": val_loss
            }
            self.test_table.add_row([f"epoch {self.best_epoch}", "VAL", brief])
            self.save_result(metrics)

            if hasattr(self, "_temp_state"):
                try:
                    self.model.load_state_dict(self._temp_state); del self._temp_state
                except Exception:
                    pass
            return metrics

        auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim, test_loss, precision_val = self.test(
            dataloader="test")
        metrics = {
            "Split": "TEST",
            "Best_epoch": self.best_epoch,
            "AUROC": auroc, "AUPRC": auprc, "F1": f1, "Sensitivity": sensitivity, "Specificity": specificity,
            "Accuracy": accuracy, "Threshold": thred_optim, "Precision": precision_val, "Test_loss": test_loss
        }
        brief = (
            f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={f1:.4f}, ACC={accuracy:.4f}, "
            f"SEN={sensitivity:.4f}, SPE={specificity:.4f}, TH={thred_optim:.4f}, LOSS={test_loss:.4f}"
        )
        print(f'[Best @ epoch {self.best_epoch}] TEST: {brief}')
        self.test_table.add_row([f"epoch {self.best_epoch}", "TEST", brief])

        self.save_result(metrics)

        if hasattr(self, "_temp_state"):
            try:
                self.model.load_state_dict(self._temp_state); del self._temp_state
            except Exception:
                pass

        return metrics

    def save_result(self, test_metrics):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.config.get("RESULT", {}).get("SAVE_MODEL", False):
            if getattr(self, "best_state_dict", None) is not None and getattr(self, "best_epoch", None) is not None:
                best_path = os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth")
                torch.save(self.best_state_dict, best_path)

        with open(os.path.join(self.output_dir, "test_markdowntable.txt"), 'w') as fp:
            fp.write(self.test_table.get_string())

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        real_batches = 0
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}", dynamic_ncols=True, leave=True)
        for batch in pbar:
            self.step += 1
            real_batches += 1
            try:
                v_d, v_p, labels, smiles_list = batch
            except Exception:
                v_d, v_p, labels = batch
                smiles_list = None

            v_d = v_d.to(self.device)
            v_p = v_p.to(self.device)
            if not self.is_regression:
                labels = (labels.float() if self.n_class == 1 else labels.long().view(-1)).to(self.device)
            else:
                labels = labels.float().to(self.device)
                if self.reg_normalize:
                    labels = self._normalize_label(labels)

            self.optim.zero_grad()
            outputs = self.model(v_d, v_p, smiles_list=smiles_list)
            score = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs

            if not self.is_regression:
                if self.n_class == 1:
                    _, loss_tensor = binary_cross_entropy(score, labels)
                else:
                    _, loss_tensor = cross_entropy_logits(score, labels)
            else:
                score_view = score if (torch.is_tensor(score) and score.dim() == 2 and score.size(1) == 1) else score.view(-1, 1)
                loss_tensor = self.criterion(score_view, labels.view(-1, 1))

            loss_tensor.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optim.step()

            batch_loss = float(loss_tensor.item())
            total_loss += batch_loss
            avg_loss = total_loss / real_batches
            pbar.set_postfix({"b_loss": f"{batch_loss:.4f}", "avg_loss": f"{avg_loss:.4f}"})
        pbar.close()

        avg_loss = total_loss / max(real_batches, 1)
        print(f'[Train] Epoch {self.current_epoch}: loss {avg_loss:.4f}')
        return avg_loss

    def test(self, dataloader="test"):
        y_label, y_score = [], []
        test_loss = 0.0
        real_batches = 0

        if dataloader == "test":
            data_loader = self.test_dataloader
            if data_loader is None:
                data_loader = self.val_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError("dataloader must be one of: 'val', 'test'.")

        model_to_use = self.model
        model_to_use.eval()
        with torch.no_grad():
            for batch in data_loader:
                real_batches += 1
                try:
                    v_d, v_p, labels, smiles_list = batch
                except Exception:
                    v_d, v_p, labels = batch
                    smiles_list = None

                v_d = v_d.to(self.device)
                v_p = v_p.to(self.device)
                if not self.is_regression:
                    labels = (labels.float() if self.n_class == 1 else labels.long().view(-1)).to(self.device)
                else:
                    labels = labels.float().to(self.device)
                    labels_norm = self._normalize_label(labels) if self.reg_normalize else labels

                outputs = model_to_use(v_d, v_p, smiles_list=smiles_list)
                score = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs

                if not self.is_regression:
                    if self.n_class == 1:
                        preds_for_metrics, loss_tensor = binary_cross_entropy(score, labels)
                    else:
                        preds_for_metrics, loss_tensor = cross_entropy_logits(score, labels)
                    test_loss += float(loss_tensor.item())

                    if preds_for_metrics is not None and torch.is_tensor(preds_for_metrics):
                        y_score.extend(preds_for_metrics.detach().cpu().view(-1).tolist())
                    else:
                        probs = torch.sigmoid(score.view(-1)) if score.dim() != 2 or score.size(1) == 1 \
                            else torch.softmax(score, dim=1)[:, 1]
                        y_score.extend(probs.detach().cpu().tolist())
                    y_label.extend(labels.view(-1).cpu().tolist())
                else:
                    score_view = score if (torch.is_tensor(score) and score.dim() == 2 and score.size(1) == 1) else score.view(-1, 1)
                    loss_tensor = self.criterion(score_view, labels_norm.view(-1, 1))
                    test_loss += float(loss_tensor.item())
                    y_label.extend(labels.view(-1).cpu().tolist())
                    if self.reg_normalize:
                        preds = self._denormalize_pred(score_view.view(-1)).detach().cpu().tolist()
                        y_score.extend(preds)
                    else:
                        y_score.extend(score_view.view(-1).detach().cpu().tolist())

        test_loss = test_loss / max(real_batches, 1)
        if not self.is_regression:
            if len(y_label) == 0:
                return (0.0, 0.0, test_loss) if dataloader == "val" \
                    else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, test_loss, 0.0)

            auroc = self._safe_auc(y_label, y_score)
            auprc = self._safe_average_precision(y_label, y_score)

            if dataloader == "val":
                return auroc, auprc, test_loss
            else:
                precisions, recalls, thresholds = precision_recall_curve(y_label, y_score)
                if thresholds is None or thresholds.size == 0:
                    thred_optim = 0.5
                    y_pred_s = [1 if s >= thred_optim else 0 for s in y_score]
                else:
                    f1_scores = []
                    for i in range(len(thresholds)):
                        p = precisions[i + 1]
                        r = recalls[i + 1]
                        denom = (p + r)
                        f1_scores.append(2 * p * r / (denom + 1e-12) if denom > 0 else 0.0)
                    best_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else 0
                    thred_optim = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
                    y_pred_s = [1 if s >= thred_optim else 0 for s in y_score]

                cm = confusion_matrix(y_label, y_pred_s)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                accuracy = (tn + tp) / (tn + fp + fn + tp + 1e-8)
                sensitivity = tp / (tp + fn + 1e-8)
                specificity = tn / (tn + fp + 1e-8)
                precision_score_val = precision_score(y_label, y_pred_s) if len(y_label) > 0 else 0.0
                f1_val_final = f1_score(y_label, y_pred_s) if len(y_label) > 0 else 0.0
                return auroc, auprc, f1_val_final, sensitivity, specificity, accuracy, thred_optim, test_loss, precision_score_val
        else:
            mse = mean_squared_error(y_label, y_score) if len(y_label) > 0 else 0.0
            ci = calculate_ci(y_label, y_score) if len(y_label) > 0 else 0.0
            rm2 = calculate_rm2(y_label, y_score) if len(y_label) > 0 else 0.0
            return mse, ci, rm2, test_loss

    def _safe_auc(self, y_true, y_score):
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return 0.0

    def _safe_average_precision(self, y_true, y_score):
        try:
            return float(average_precision_score(y_true, y_score))
        except Exception:
            return 0.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset_builder import DataframeCombiner
import pandas as pd



class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss      
        
class AmseLoss(nn.Module):
    """
    Attention-Adaptive MSE Loss (positive regions only).
    """
    def __init__(self):
        super(AmseLoss, self).__init__()

    def forward(self, T_Mk, Gk, target):
        # Ensure that target is a boolean tensor
        target_mask = (target != 0)

        # Flatten the spatial dimensions and remove the channel dimension
        T_Mk_flat = T_Mk.view(T_Mk.size(0), -1)
        Gk_flat = Gk.view(Gk.size(0), -1)

        # Calculate the squared differences and then sum across spatial dimensions
        numerator = torch.sum((T_Mk_flat - Gk_flat) ** 2, dim=1)
        denominator = torch.sum(T_Mk_flat + Gk_flat, dim=1) + 1e-6

        # Compute the AMSE loss for each abnormality
        lamse_values = numerator / denominator

        # Convert MetaTensor to Torch Tensor if needed
        if hasattr(lamse_values, 'as_tensor'):
            lamse_values = lamse_values.as_tensor()
        # Apply the target mask
        masked_lamse_values = lamse_values[target_mask]
        # Calculate mean of masked lamse values
        lamse = torch.mean(masked_lamse_values) if masked_lamse_values.numel() > 0 else torch.tensor(0.0).to(T_Mk.device)

        return lamse        
  
class AMSELossFull(nn.Module):
    """
    AMSE loss used for full region (positive + negative).
    """
    def __init__(self):
        super(AMSELossFull, self).__init__()

    def forward(self, T_Mk, Gk):

        # Flatten the spatial dimensions and remove the channel dimension
        T_Mk_flat = T_Mk.view(T_Mk.size(0), -1)
        Gk_flat = Gk.view(Gk.size(0), -1)

        # Calculate the squared differences and then sum across spatial dimensions
        numerator = torch.sum((T_Mk_flat - Gk_flat) ** 2, dim=1)
        denominator = torch.sum(T_Mk_flat + Gk_flat, dim=1) + 1e-6

        # Compute the AMSE loss
        lamse_values = numerator / denominator

        return torch.mean(lamse_values)
     
class SegLoss(nn.Module):
    """Classification + positive-only AMSE loss"""
    def __init__(self, class_weights, amse_weight=1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.amse = AMSELoss()
        self.amse_weight = amse_weight

    def forward(self, out_cls, target_cls, out_seg, target_seg):
        out_seg = torch.sigmoid(out_seg)
        ce_loss = self.ce(out_cls, target_cls)
        amse_loss = self.amse(out_seg, target_seg, target_cls)
        total_loss = ce_loss + amse_loss * self.amse_weight
        return total_loss, amse_loss, ce_loss

class SegLossFull(nn.Module):
    """Classification + full-volume AMSE loss"""
    def __init__(self, class_weights, amse_weight=1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.amse = AMSELossFull()
        self.amse_weight = amse_weight

    def forward(self, out_cls, target_cls, out_seg, target_seg):
        out_seg = torch.sigmoid(out_seg)
        ce_loss = self.ce(out_cls, target_cls.long())
        amse_loss = self.amse(out_seg, target_seg)
        total_loss = ce_loss + amse_loss * self.amse_weight
        return total_loss, amse_loss, ce_loss
    
class MultiSegLoss(nn.Module):
    """Multi-organ classification + per-organ segmentation + positive-only AMSE"""
    def __init__(self, amse_weight=1, **class_weights):
        super().__init__()
        self.loss_funcs = {
            organ: SegLoss(w, amse_weight) for organ, w in class_weights.items()
        }
        self.amse = AMSELoss()

    def forward(self, out_cls, target_cls, out_seg, target_seg):
        out_seg = torch.sigmoid(out_seg)
        total_loss, total_amse, total_ce = 0, 0, 0
        for i, (organ, loss_func) in enumerate(self.loss_funcs.items()):
            cls_i = out_cls[:, i, :]
            lbl_i = target_cls[..., i]
            seg_out_i = out_seg[:, i:i+1, ...]
            seg_tgt_i = target_seg[:, i:i+1, ...]
            loss, amse, ce = loss_func(cls_i, lbl_i, seg_out_i, seg_tgt_i)
            total_loss += loss
            total_amse += amse
            total_ce += ce
        loss_all_mask = AMSELoss(out_seg, target_seg, (target_cls.sum(1) > 0).int())
        total_amse+= loss_all_mask
        total_loss += loss_all_mask

        return total_loss, total_amse, total_ce
    
class FocalSegLoss(nn.Module):
    """Focal loss for classification + AMSE loss for segmentation."""
    def __init__(self, alpha, gamma=2.0, reduction='mean', amse_weight=1):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.lamse = AmseLoss()
        self.amse_weight = amse_weight

    def forward(self, outputs, targets, masks_outputs, masks_targets):
        loss1 = self.focal(outputs, targets)
        loss2 = self.lamse(masks_outputs, masks_targets, targets)
        return loss1 + (loss2 * self.amse_weight)


class MultiSegLossFull(nn.Module):

    """
    Multi-organ classification + per-organ segmentation + full-volume AMSE
    """

    def __init__(self, amse_weight=1, **class_weights):
        super().__init__()
        self.loss_funcs = {
            organ: SegLossFull(w, amse_weight) for organ, w in class_weights.items()
        }
        self.amse = AMSELossFull()

    def forward(self, out_cls, target_cls, out_seg, target_seg):
        out_seg = torch.sigmoid(out_seg)
        total_loss, total_amse, total_ce = 0, 0, 0
        for i, (organ, loss_func) in enumerate(self.loss_funcs.items()):
            cls_i = out_cls[:, i, :]
            lbl_i = target_cls[..., i]
            seg_out_i = out_seg[:, i * 2: i * 2 + 2, ...]
            seg_tgt_i = target_seg[:, i * 2: i * 2 + 2, ...]
            loss, amse, ce = loss_func(cls_i, lbl_i, seg_out_i, seg_tgt_i)
            total_loss += loss
            total_amse += amse
            total_ce += ce

        # Optionally add full-image AMSE
        amse_all = self.amse(out_seg, target_seg)
        total_amse += amse_all
        total_loss += amse_all

        return total_loss, total_amse, total_ce

def compute_loss_weights(grouped, device):
    """
    Compute class weights based on sample count. Supports multi-class.
    
    Args:
        grouped: Grouped data by class.
        device: Target device for the weight tensor (CPU/GPU).
    
    Returns:
        torch.Tensor: Inverse frequency weights for each class.
    """
    # Initialize a dictionary to store sample count per class
    group_dict = {}
    for name, group in grouped:
        group_dict[name] = len(group)

    # Ensure weight covers all class indices
    weights = torch.tensor(
        [1 / group_dict.get(cls, 1) for cls in range(len(group_dict))]
    ).to(device)

    return weights


class LossBuilder:
    """Loss function builder for multi-organ classification + segmentation training."""
    def __init__(self, cfg, df_cgmh=None, df_rsna=None, class_type="multiple", device="cuda"):
        self.cfg = cfg
        self.device = device
        self.df_cgmh = df_cgmh if df_cgmh is not None else pd.DataFrame()
        self.df_rsna = df_rsna if df_rsna is not None else pd.DataFrame()
        self.n_classes = cfg.data_setting.n_classes
        self.loss_type = cfg.data_setting.loss
        self.class_type = class_type
        self.df_all = DataframeCombiner(self.df_cgmh, self.df_rsna, self.n_classes).combine()

    def compute_weights(self, grouped):
        return compute_loss_weights(grouped, device=self.device)

    def get_loss_function(self):
        if self.class_type == "multiple":
            weights_liv = self.compute_weights(self.df_all.groupby("liver_label"))
            weights_spl = self.compute_weights(self.df_all.groupby("spleen_label"))
            weights_kid = self.compute_weights(self.df_all.groupby("kidney_label"))
        else:
            col_map = {
                "all": "inj_solid",
                "liver": "liver_label",
                "spleen": "spleen_label",
                "kidney": "kidney_label"
            }
            grouped = self.df_all.groupby(col_map[self.class_type])
            weights = self.compute_weights(grouped)

        if self.loss_type == "crossentropy":
            return torch.nn.CrossEntropyLoss(weight=weights)

        elif self.loss_type == "focal":
            return FocalLoss(alpha=weights, gamma=2.0, reduction='mean')

        elif self.loss_type == "focal_amse":
            return FocalSegLoss(alpha=weights, gamma=2.0, reduction='mean', amse_weight=1)

        elif self.loss_type == "amse":
            if self.class_type == "multiple":
                return MultiSegLoss(
                    amse_weight=1,
                    liver=weights_liv,
                    spleen=weights_spl,
                    kidney=weights_kid
                )
            else:
                return SegLoss(class_weights=weights, amse_weight=1)

        elif self.loss_type == "amse_full":
            if self.class_type == "multiple":
                return MultiSegLossFull(
                    amse_weight=1,
                    liver=weights_liv,
                    spleen=weights_spl,
                    kidney=weights_kid
                )
            else:
                return SegLossFull(class_weights=weights, amse_weight=1)

        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

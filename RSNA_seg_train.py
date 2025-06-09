import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from datetime import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import csv
import nibabel as nib
import matplotlib.pyplot as plt
import sys

# 路徑要根據你的docker路徑來設定
sys.path.append("/tf/yilian618/ABD_classification/model/")
from efficientnet_3d.model_3d import EfficientNet3D
from efficientnet_3d import efficientnet_fpn_3d
from resnet_3d import resnet_3d, resnet, MedicalNet_resnet
# from resnet_3d import resnet
# from resnet_3d import MedicalNet_resnet

# 此架構參考這篇
# https://github.com/fei-aiart/NAS-Lung
sys.path.append("/tf/yilian618/ABD_classification/model/NAS-Lung/")
from models.cnn_res import ConvRes
from models.net_sphere import AngleLoss

import utils.config as config
import configparser
import gc
import math
import json
from utils.training_torch_utils import (
    train_seg,
    validation_seg,
    plot_loss_metric,
    FocalLoss,
    ConcatMasksd,
    TargetedChannelMasksd,
    AttentionModel,
    AttentionModel_new,
    Dulicated_new,
)
import pickle
from utils.loss import FocalSegLoss, SegLoss, Muti_SegLoss, TotalSegLoss

# Data augmnetation module (based on MONAI)
from monai.networks.nets import UNet, densenet, SENet, ViT
from monai.apps import download_and_extract
from monai.data import CacheDataset, DataLoader, Dataset
from monai.config import print_config
from monai.utils import first, set_determinism
from monai.transforms import (
    LoadImage,
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    Rand3DElasticd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    FillHoles,
    Resized,
    RepeatChanneld,
)
import functools

# let all of print can be flush = ture
print = functools.partial(print, flush=True)


def get_parser():
    parser = argparse.ArgumentParser(description="spleen classification")
    parser.add_argument("-f", "--file", help=" The config file name. ", type=str)
    parser.add_argument(
        "-c",
        "--class_type",
        help=" The class of data. (liver, kidney, spleen, all) ",
        type=str,
    )
    return parser


# 對應不同資料有不同路徑與處理
# 目前以data_progress_all為主
# 依照不同 class_type 來區分資料
# 目前都是以 bbox 為主
# 不同器官的mask又有區分
# whole : 整張影像
# cropping_normal : Totalsegmatator出來的結果，沒有做其他處理
# cropping_convex : 對Totalsegmatator出來的結果進行dilation與convex處理
# cropping_dilation : 對Totalsegmatator出來的結果進行dilation
# bbox : 對Totalsegmatator出來的結果進行dilation並且轉換成bounding box


# 這個部分也要看你docker路徑去改對應路徑
def data_progress_all(file, dicts, class_type):
    dicts = []
#     dirs = "/SSD/TotalSegmentator/rsna_selected_crop_gaussian"
    dirs = "/tf/TotalSegmentator/rsna_select_z_whole_image"
    mask_dirs = "/tf/SharedFolder/TotalSegmentator/gaussian_mask"
    for index, row in file.iterrows():
        output = os.path.basename(row['file_paths'])[:-7]
        image = os.path.join(dirs,output)+".nii.gz"
        if class_type =="spleen":
            mask = os.path.join(mask_dirs,"spl",output)+".nii.gz"
            if n_classes>2:
                label = np.argmax([int(row["spleen_healthy"]),int(row["spleen_low"]), int(row["spleen_high"])], axis=0)
            else:
                label = int(np.where(row["spleen_healthy"]==1,0,1))   
            dicts.append({"image": image, "label": label, "mask": mask})
        elif class_type=="kidney":
            mask = os.path.join(mask_dirs,"kid_all",output)+".nii.gz"
            label = int(np.where(row["kidney_healthy"]==1,0,1))
            dicts.append({"image": image, "label": label, "mask": mask})
            
        elif class_type=="liver":
            mask = os.path.join(mask_dirs,"liv",output)+".nii.gz"
            label = int(np.where(row["liver_healthy"]==1,0,1))
            dicts.append({"image": image, "label": label, "mask": mask})
            
        elif class_type=="all":
            mask = os.path.join(mask_dirs,"any_inj",output)+".nii.gz"
            label = 0 if (row["liver_healthy"] == 1 and row["spleen_healthy"] == 1 and row["kidney_healthy"] == 1) else 1
            dicts.append({"image": image, "label": label, "mask": mask})
        elif class_type=="multiple":
            mask_liv = os.path.join(mask_dirs,"liv",output)+".nii.gz"
            mask_spl = os.path.join(mask_dirs,"spl",output)+".nii.gz"
            mask_kid = os.path.join(mask_dirs,"kid_all",output)+".nii.gz"
            label_liv = int(np.where(row["liver_healthy"]==1,0,1))
            label_spl= int(np.where(row["spleen_healthy"]==1,0,1))
            label_kid= int(np.where(row["kidney_healthy"]==1,0,1))
#             label = np.array([label_liv, label_spl, label_kid])
            label = np.stack([label_liv, label_spl, label_kid], axis=0)
            dicts.append({"image": image, "label": label, "mask_liv": mask_liv, "mask_spl": mask_spl, "mask_kid": mask_kid})
        
    return dicts


# 判斷是否為injury
def inj_check(row):
    kid_inj_tmp = 0 if row["kid_inj_rt"] == row["kid_inj_lt"] == 0 else 1
    liv_inj_tmp = 0 if row["liv_inj"] == 0 else 1
    spl_inj_tmp = 0 if row["spl_inj"] == 0 else 1
    return pd.Series([kid_inj_tmp, liv_inj_tmp, spl_inj_tmp])


# 日期判斷並轉換
def convert_date(x):
    if pd.isna(x):  # Check if the value is NaN
        return x  # If it's NaN, return it as-is
    else:
        return pd.to_datetime(int(x), format="%Y%m%d")


# 將positive進行複製
def duplicate(df, col_name, num_sample, pos_sel=True):
    if pos_sel:
        df_inj_tmp = df[df[col_name] == 1]
    else:
        df_inj_tmp = df

    # 進行重複
    df_inj_tmp_duplicated = pd.concat([df_inj_tmp] * num_sample, ignore_index=True)

    # 將原始的df和複製後的df結合
    df_new = pd.concat([df, df_inj_tmp_duplicated], ignore_index=True)

    return df_new

def compute_loss_weights(grouped, num_samples, device):
    """
    計算權重函數，支援多於兩個類別。
    grouped: 分組後的數據。
    num_samples: 樣本總數。
    device: 權重張量的設備（CPU/GPU）。
    """
    # 初始化字典以存儲每個類別的樣本數量
    group_dict = {}
    for name, group in grouped:
        group_dict[name] = len(group) * num_samples

    # 確保權重對應到所有類別
    weights = torch.tensor(
        [1 / group_dict.get(cls, 1) for cls in range(len(group_dict))]
    ).to(device)
    return weights


# 依據positive情況進行資料切分
def train_valid_test_split(df, test_data, ratio=(0.8, 0.2, 0.2), seed=0):
    # set key for df
    df['group_key'] = df.apply(
        lambda row: (
            f"{row['liver_low']}_"
            f"{row['liver_high']}_"
            f"{row['spleen_low']}_"
            f"{row['spleen_high']}_"
            f"{row['kidney_low']}_"
            f"{row['kidney_high']}_"
            f"{row['bowel_injury']}_"
            f"{row['extravasation_injury']}"
        ),axis=1)

    df = df.reset_index()
    test_data = test_data.reset_index()
    
    test_df = test_data
    train_df = df.groupby("group_key", group_keys=False).sample(
            frac=ratio[0], random_state=seed)
    valid_df = df.drop(train_df.index.to_list())

    return train_df, valid_df, test_df



class AmseLoss(nn.Module):
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
        denominator = torch.sum(T_Mk_flat + Gk_flat, dim=1)

        # Compute the AMSE loss for each abnormality
        lamse_values = numerator / denominator
#         print(type(lamse_values))

        # Convert MetaTensor to Torch Tensor if needed
        if hasattr(lamse_values, 'as_tensor'):
#             print(type(lamse_values))
            lamse_values = lamse_values.as_tensor()

        # Apply the target mask
        masked_lamse_values = lamse_values[target_mask]

        # Calculate mean of masked lamse values
        if masked_lamse_values.numel() > 0:
            lamse = torch.mean(masked_lamse_values)
        else:
            lamse = torch.tensor(0.0).to(T_Mk.device)

        return lamse


# class SegLoss(nn.Module):
#     def __init__(self, loss_weights, weight=1):
#         super(SegLoss, self).__init__()
#         self.ce = torch.nn.CrossEntropyLoss(weight=loss_weights)
# #         self.bce = nn.BCELoss()
#         self.lamse = AmseLoss()
#         self.weight=weight
        
#     def forward(self, outputs, targets, masks_outputs,masks_targets):
        
#         loss1 = self.ce(outputs, targets)
#         loss2 = self.lamse(masks_outputs, masks_targets, targets)
#         loss = loss1 + (loss2 * self.weight) 
        
#         return loss
    
# class Muti_SegLoss(nn.Module):
#     def __init__(self, loss_weights_liv, loss_weights_spl, loss_weights_kid, weight=1):
#         super(Muti_SegLoss, self).__init__()
        
#         self.SegLoss_liv = SegLoss(loss_weights_liv, weight)
#         self.SegLoss_spl = SegLoss(loss_weights_spl, weight)
#         self.SegLoss_kid = SegLoss(loss_weights_kid, weight)
        
#     def forward(self, outputs, targets, masks_outputs, masks_targets):
        
#         loss_liv = self.SegLoss_liv(outputs[:, 0, :], targets[...,0], masks_outputs[:, 0:1, :, :, :], masks_targets[:, 0:1, :, :, :])
#         loss_spl = self.SegLoss_spl(outputs[:, 1, :], targets[...,1], masks_outputs[:, 1:2, :, :, :], masks_targets[:, 1:2, :, :, :])
#         loss_kid = self.SegLoss_kid(outputs[:, 2, :], targets[...,2], masks_outputs[:, 2:3, :, :, :] , masks_targets[:, 2:3, :, :, :] )
#         loss = loss_liv+loss_spl+loss_kid
        
#         return loss


# 進行完整一次預測
def run_once(times=0):
    # reset config parameter
    config.initialize()

    train_df, valid_df, test_df = train_valid_test_split(
        df_all,test_data, ratio=data_split_ratio, seed=seed
    )
#     train_df = train_df[:50]
#     valid_df = valid_df[:25]
#     test_df = test_df[:100]
    
    if num_samples != 1:
        if class_type == "all":
            train_df = duplicate(train_df, "inj_solid", num_samples)
        elif class_type == "liver":
            train_df = duplicate(train_df, "liv_inj_tmp", num_samples)
        elif class_type == "spleen":
            train_df = duplicate(train_df, "spl_inj_tmp", num_samples)
        elif class_type == "kidney":
            train_df = duplicate(train_df, "kid_inj_tmp", num_samples)

    train_data_dicts = data_progress_all(train_df, "train_data_dict", class_type)
    valid_data_dicts = data_progress_all(valid_df, "valid_data_dict", class_type)
    test_data_dicts = data_progress_all(test_df, "test_data_dict", class_type)
    # with open('/tf/jacky831006/ABD_data/train.pickle', 'wb') as f:
    #    pickle.dump(train_data_dicts, f)

    set_determinism(seed=0)
    train_ds = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        cache_rate=1,
        num_workers=dataloader_num_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=traning_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
    )
    valid_ds = CacheDataset(
        data=valid_data_dicts,
        transform=valid_transforms,
        cache_rate=1,
        num_workers=dataloader_num_workers,
    )
    val_loader = DataLoader(
        valid_ds, batch_size=valid_batch_size, num_workers=dataloader_num_workers
    )

    device = torch.device("cuda:0")

    # Model setting
    # normal_structure :
    # True 為沒有修改原始架構(深度較深，最後的影像解析度較低)
    # False 則為修改原始架構(深度較淺，最後的影像解析度較高)
    # bbox 則代表input除了原始的影像外，還加入bounding box影像藉由channel增維
    if architecture == "densenet":
        if normal_structure:
            # Normal DenseNet121
            if bbox:
                model = densenet.densenet121(
                    spatial_dims=3, in_channels=2, out_channels=2
                ).to(device)
            else:
                model = densenet.densenet121(
                    spatial_dims=3, in_channels=1, out_channels=2
                ).to(device)
        else:
            if bbox:
                # Delete last dense block
                model = densenet.DenseNet(
                    spatial_dims=3,
                    in_channels=2,
                    out_channels=2,
                    block_config=(6, 12, 40),
                ).to(device)
            else:
                model = densenet.DenseNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=2,
                    block_config=(6, 12, 40),
                ).to(device)

    elif architecture == "resnet":
        if bbox:
            model = resnet_3d.generate_model(
                101, normal=normal_structure, n_input_channels=2
            ).to(device)
        else:
            model = resnet_3d.generate_model(101, normal=normal_structure).to(device)

    elif architecture == "efficientnet":
        if bbox:
            model = EfficientNet3D.from_name(
                f"efficientnet-{structure_num}",
                in_channels=2,
                num_classes=2,
                image_size=size,
                normal=normal_structure,
            ).to(device)
        else:
            model = EfficientNet3D.from_name(
                f"efficientnet-b0",
                in_channels=1,
                num_classes=3,
                image_size=size,
                normal=normal_structure,
            ).to(device)

    elif architecture == "CBAM":
        if size[0] == size[1] == size[2]:
            if bbox:
                model = ConvRes(
                    size[0],
                    [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]],
                    input_channel=2,
                    normal=normal_structure,
                ).to(device)
            else:
                model = ConvRes(
                    size[0],
                    [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]],
                    normal=normal_structure,
                    num_classes=3
                ).to(device)
        else:
            raise RuntimeError("CBAM model need same size in x,y,z axis")
    
    elif architecture == "pretrain":
        model = resnet.generate_model(model_depth=50,n_classes=1139,n_input_channels=3)
        pretrain = torch.load("/tf/yilian618/ABD_classification/r3d50_KMS_200ep.pth", map_location='cpu')
        model.load_state_dict(pretrain['state_dict'])
        model.fc = nn.Linear(model.fc.in_features, 3)
        tmp_model = resnet.modify_first_conv_layer(model, new_in_channels=1, pretrained=True)
        model = tmp_model.to(device)
        
    elif architecture == "pretrain_seg":
        if class_type == "multiple":
            model = resnet.generate_model(model_depth="seg_muti",n_classes=n_classes,n_input_channels=1, mask_classes=3).to(device)
        else:
#             print(n_classes)
            model = resnet.generate_model(model_depth="seg",n_classes=n_classes,n_input_channels=1, mask_classes=mask_classes).to(device)     
#         model = resnet.ResNetUNet(block=Bottleneck, layers=[3, 4, 6, 3], block_inplanes=[64, 128, 256, 512], n_input_channels=1, n_classes=1).to(device)     
        net_dict = model.state_dict()

        # Load the pretrained model
        pretrain = torch.load('/tf/yilian618/ABD_classification/r3d50_KMS_200ep.pth')

        # Process the pretrained weights
        pretrain_dict = {new_key: v for k, v in pretrain['state_dict'].items() 
                         if (new_key := k.replace("module.","")) in net_dict.keys() and not new_key.startswith('fc.')}
        # Handle conv1 input channel size mismatch
        if 'conv1.weight' in pretrain_dict:
            pretrain_conv1_weight = pretrain_dict['conv1.weight']

            # The pretrained model has 3 input channels, so we need to adapt it for 1 channel
            if pretrain_conv1_weight.shape[1] == 3 and model.conv1.weight.shape[1] == 1:
                # Average over the RGB channels to fit a single channel input
                pretrain_conv1_weight = pretrain_conv1_weight.mean(dim=1, keepdim=True)
                pretrain_dict['conv1.weight'] = pretrain_conv1_weight

        # Update model state dict with the pre-trained model, ignoring size mismatches for fc layers
        net_dict.update(pretrain_dict)

        # Load the updated state dict into the model, but ignore the fully connected (fc) layer
        model.load_state_dict(net_dict, strict=False)  # strict=False ignores any layers not found in the state_dict

        # Optionally, reinitialize the fully connected layer if needed
#         model.fc.weight.data.normal_(0, 0.01)  # Reinitialize fc weights
#         model.fc.bias.data.zero_()  # Reinitialize fc bias
    elif architecture == "efficientnet_fpn":
#         model = efficientnet_fpn_3d.EfficientNet3D_FPN(model_name= f"efficientnet-b0", in_channels=1,class_num=3,image_size=size, normal=False, dropout=0.2).to(device)
        model = efficientnet_fpn_3d.EfficientNet3D_FPN(model_name= f"efficientnet-b0", in_channels=1,class_num=3,image_size=size, normal=True, dropout=0.2).to(device)
    
    elif architecture == "MedicalNet":
        model = MedicalNet_resnet.resnet50(sample_input_D=size[2], sample_input_H=size[0], sample_input_W=size[1], num_seg_classes=3, shortcut_type='B').to(device)
        net_dict = model.state_dict()
        pretrain = torch.load('/tf/jacky831006/ABD_classification/pretrain_weight/resnet_50_23dataset.pth')
        pretrain_dict = {new_key: v for k, v in pretrain['state_dict'].items() if (new_key := k.replace("module.","")) in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
        
    elif architecture == "MedicalNet_seg":
            model = MedicalNet_resnet.resnet50(sample_input_D=size[2], sample_input_H=size[0], sample_input_W=size[1], num_seg_classes=1, shortcut_type='B').to(device)
            net_dict = model.state_dict()
            pretrain = torch.load('/tf/jacky831006/ABD_classification/pretrain_weight/resnet_50_23dataset.pth')
            pretrain_dict = {new_key: v for k, v in pretrain['state_dict'].items() if (new_key := k.replace("module.","")) in net_dict.keys()}
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)

    # Mask attention block in CNN
    if attention_mask:
        dense = densenet.DenseNet(
            spatial_dims=3, in_channels=1, out_channels=2, block_config=(6, 12, 20)
        ).to(device)
        model = AttentionModel_new(2, size, model, dense, architecture).to(device)
        # model = AttentionModel(2, size, model, architecture).to(device)

    # Imbalance loss
    # 根據不同資料的比例做推測
    # 目前kidney都以單邊受傷為主
#     df_all["spleen_label"] = df_all.apply(lambda row: np.argmax([int(row["spleen_healthy"]), int(row["spleen_low"]), int(row["spleen_high"])]), axis=1)
    if n_classes > 2:
        df_all["kidney_label"] = df_all.apply(lambda row: np.argmax([int(row["kidney_healthy"]), int(row["kidney_low"]), int(row["kidney_high"])]), axis=1)
        df_all["spleen_label"] = df_all.apply(lambda row: np.argmax([int(row["spleen_healthy"]), int(row["spleen_low"]), int(row["spleen_high"])]), axis=1)
        df_all["liver_label"] = df_all.apply( lambda row: np.argmax([int(row["liver_healthy"]), int(row["liver_low"]), int(row["liver_high"])]), axis=1)
        df_all["inj_solid"] = np.where((df_all["liver_healthy"] == 1) & (df_all["spleen_healthy"] == 1) & (df_all["kidney_healthy"] == 1),0,1)
    else:
        df_all["spleen_label"] = np.where(df_all["spleen_healthy"]==1,0,1)
        df_all["kidney_label"] = np.where(df_all["kidney_healthy"]==1,0,1)
        df_all["liver_label"] = np.where(df_all["liver_healthy"]==1,0,1)
        df_all["inj_solid"] = np.where((df_all["liver_healthy"] == 1) & (df_all["spleen_healthy"] == 1) & (df_all["kidney_healthy"] == 1),0,1)
        
        
    if class_type == "multiple":
        # 分別計算 liver, spleen, kidney 的 weights
        loss_weights_liv = compute_loss_weights(df_all.groupby("liver_label"), num_samples, device)
        loss_weights_spl = compute_loss_weights(df_all.groupby("spleen_label"), num_samples, device)
        loss_weights_kid = compute_loss_weights(df_all.groupby("kidney_label"), num_samples, device)
    else:
        # 根據 class_type 設置 grouped
        group_by_mapping = {
            "all": "inj_solid",
            "liver": "liver_label",
            "spleen": "spleen_label",
            "kidney": "kidney_label",
        }
        grouped = df_all.groupby(group_by_mapping[class_type])
        weights = compute_loss_weights(grouped, num_samples, device)


    # CBAM有自己的loss function
    if architecture == "CBAM" and not normal_structure:
        loss_function = AngleLoss(weight=weights)
    elif loss_type == "crossentropy":
        loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    elif loss_type == "focal":
        loss_function = FocalLoss(class_num=2, alpha=0.25, weight=weights)
    elif loss_type == "focal_amse":
        loss_function = FocalSegLoss(alpha=weights, gamma=2.0, reduction='mean', weight=1)
    elif loss_type == "amse":
        if class_type=="multiple":
            loss_function = Muti_SegLoss(loss_weights_liv, loss_weights_spl, loss_weights_kid)
        else:
            loss_function = SegLoss(loss_weights=weights, weight=1)
    elif loss_type == "total_amse":
        loss_function = TotalSegLoss(loss_weights=weights, weight=1)
        

    # Grid search
    if len(init_lr) == 1:
        optimizer = torch.optim.Adam(model.parameters(), init_lr[0])
    else:
        optimizer = torch.optim.Adam(model.parameters(), init_lr[times])

    if lr_decay_epoch == 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=lr_decay_rate, patience=epochs, verbose=True
        )
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs,gamma=lr_decay_rate, verbose=True )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_decay_rate,
            patience=lr_decay_epoch,
            verbose=True,
        )

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # check_point path
    check_path = (
        f"/tf/yilian618/classification_torch/training_checkpoints/{classification_type}/{class_type}/{now}"
    )
    if not os.path.isdir(check_path):
        os.makedirs(check_path)
    print(f"\n Weight location:{check_path}", flush=True)
    if cross_kfold == 1:
        print(f"\n Processing begining", flush=True)
    else:
        print(f"\n Processing fold #{times}", flush=True)

    data_num = len(train_ds)
    # test_model = train(model, device, data_num, epochs, optimizer, loss_function, train_loader, \
    #                    val_loader, early_stop, init_lr, lr_decay_rate, lr_decay_epoch, check_path)

    test_model = train_seg(
        model,
        device,
        data_num,
        epochs,
        optimizer,
        loss_function,
        train_loader,
        val_loader,
        early_stop,
        scheduler,
        check_path,
    )

    # plot train loss and metric
    plot_loss_metric(config.epoch_loss_values, config.metric_values, check_path)
    # remove dataloader to free memory
    del train_ds
    del train_loader
    del valid_ds
    del val_loader
    gc.collect()

    # Avoid ram out of memory
    test_ds = CacheDataset(
        data=test_data_dicts,
        transform=valid_transforms,
        cache_rate=1,
        num_workers=dataloader_num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers
    )
    # validation is same as testing
    print(f"Best accuracy:{config.best_metric}")
    if config.best_metric != 0:
        load_weight = f"{check_path}/{config.best_metric}.pth"
        model.load_state_dict(torch.load(load_weight))

    # record paramter
    accuracy_list.append(config.best_metric)
    file_list.append(now)
    epoch_list.append(config.best_metric_epoch)

    test_acc = validation_seg(model, test_loader, device)
    test_accuracy_list.append(test_acc)
    del test_ds
    del test_loader
    gc.collect()

    print(f"\n Best accuracy:{config.best_metric}, Best test accuracy:{test_acc}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    class_type = args.class_type
    # 讀檔路徑，之後可自己微調
    if args.file.endswith("ini"):
        cfgpath = f"/tf/yilian618/ABD_classification/config/{class_type}/{args.file}"
    else:
        cfgpath = (
            f"/tf/yilian618/ABD_classification/config/{class_type}/{args.file}.ini"
        )

    conf = configparser.ConfigParser()
    conf.read(cfgpath)

    # Augmentation
    num_samples = conf.getint("Augmentation", "num_sample")
    size = eval(conf.get("Augmentation", "size"))
    prob = conf.getfloat("Rand3DElasticd", "prob")
    sigma_range = eval(conf.get("Rand3DElasticd", "sigma_range"))
    magnitude_range = eval(conf.get("Rand3DElasticd", "magnitude_range"))
    translate_range = eval(conf.get("Rand3DElasticd", "translate_range"))
    rotate_range = eval(conf.get("Rand3DElasticd", "rotate_range"))
    scale_range = eval(conf.get("Rand3DElasticd", "scale_range"))

    # Data_setting
    architecture = conf.get("Data_Setting", "architecture")
    if architecture == "efficientnet":
        structure_num = conf.get("Data_Setting", "structure_num")
    gpu_num = conf.getint("Data_Setting", "gpu")
    seed = conf.getint("Data_Setting", "seed")
    cross_kfold = conf.getint("Data_Setting", "cross_kfold")
    normal_structure = conf.getboolean("Data_Setting", "normal_structure")
    data_split_ratio = eval(conf.get("Data_Setting", "data_split_ratio"))
    # imbalance_data_ratio = conf.getint('Data_Setting','imbalance_data_ratio')
    epochs = conf.getint("Data_Setting", "epochs")
    # early_stop = 0 means None
    early_stop = conf.getint("Data_Setting", "early_stop")
    traning_batch_size = conf.getint("Data_Setting", "traning_batch_size")
    valid_batch_size = conf.getint("Data_Setting", "valid_batch_size")
    testing_batch_size = conf.getint("Data_Setting", "testing_batch_size")
    dataloader_num_workers = conf.getint("Data_Setting", "dataloader_num_workers")
    # init_lr = conf.getfloat('Data_Setting','init_lr')
    init_lr = json.loads(conf.get("Data_Setting", "init_lr"))
    # optimizer = conf.get('Data_Setting','optimizer')
    lr_decay_rate = conf.getfloat("Data_Setting", "lr_decay_rate")
    lr_decay_epoch = conf.getint("Data_Setting", "lr_decay_epoch")
    # whole, cropping_normal, cropping_convex, cropping_dilation
    loss_type = conf.get("Data_Setting", "loss")
    bbox = conf.getboolean("Data_Setting", "bbox")
    attention_mask = conf.getboolean("Data_Setting", "attention_mask")
    # HU range: ex 0,100
    img_hu = eval(conf.get("Data_Setting", "img_hu"))
    if conf.has_option("Data_Setting", "n_classes"):
        n_classes = conf.getint("Data_Setting", "n_classes")
        classification_type = "Multiclass"
    else:
        n_classes = 2
        if class_type=="multiple":
            classification_type = "Multilabel"
        else:
            classification_type = "Binary"
    if conf.has_option("Data_Setting", "mask_classes"):
        mask_classes = conf.getint("Data_Setting", "mask_classes")
    else:
        mask_classes = 1
    # Setting cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    # Data progressing
    All_data = pd.read_csv("/tf/yilian618/rsna_train_new_v2.csv")
    no_seg = pd.read_csv("/tf/yilian618/nosegmentation.csv")
    All_data = All_data[~All_data['file_paths'].isin(no_seg['file_paths'])]
    # Assuming All_data is a pandas DataFrame and 'file_paths' is one of its columns
    All_data = All_data[~All_data['file_paths'].isin(["/SSD/rsna-2023/train_images_new/63501/SDY00001/1.2.123.12345.1.2.3.63501.7194.nii.gz"])]
    test_data = pd.read_csv("/tf/jacky831006/ABD_classification/rsna_test_20240531.csv")
    df_all = All_data[~All_data['file_paths'].isin(test_data['file_paths'].values)]
    
    pos_data = df_all[df_all['any_injury']==1]
    neg_data = df_all[df_all['any_injury']==0].sample(n=800, random_state=seed)
    df_all = pd.concat([pos_data, neg_data])

    if class_type=="multiple":
#         dicts.append({"image": image, "label": label, "mask_liv": mask_liv, "mask_spl": mask_spl, "mask_kid": mask_kid})
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "mask_liv", "mask_spl", "mask_kid"]),
                EnsureChannelFirstd(keys=["image", "mask_liv", "mask_spl", "mask_kid"]),
                ScaleIntensityRanged(
                    # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    keys=["image"],
                    a_min=-50,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                Spacingd(
                    keys=["image", "mask_liv", "mask_spl", "mask_kid"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"
                ),
                Orientationd(keys=["image", "mask_liv", "mask_spl", "mask_kid"], axcodes="RAS"),
                CropForegroundd(keys=["image", "mask_liv", "mask_spl", "mask_kid"], source_key="image"),
                Resized(keys=["image", "mask_liv", "mask_spl", "mask_kid"], spatial_size=size, mode="trilinear"),
                Rand3DElasticd(
                    keys=["image", "mask_liv", "mask_spl", "mask_kid"],
                    mode="bilinear",
                    prob=prob,
                    sigma_range=sigma_range,
                    magnitude_range=magnitude_range,
                    spatial_size=size,
                    translate_range=translate_range,
                    rotate_range=rotate_range,
                    scale_range=scale_range,
                    padding_mode="border",
                ),
                ConcatMasksd(keys=["mask_liv", "mask_spl", "mask_kid"], output_key="mask"),  # 合併 masks
            ]
        )
        valid_transforms = Compose(
            [
                LoadImaged(keys=["image", "mask_liv", "mask_spl", "mask_kid"]),
                EnsureChannelFirstd(keys=["image", "mask_liv", "mask_spl", "mask_kid"]),
                ScaleIntensityRanged(
                    # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    keys=["image"],
                    a_min=-50,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                Spacingd(
                    keys=["image", "mask_liv", "mask_spl", "mask_kid"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"
                ),
                Orientationd(keys=["image", "mask_liv", "mask_spl", "mask_kid"], axcodes="RAS"),
                CropForegroundd(keys=["image", "mask_liv", "mask_spl", "mask_kid"], source_key="image"),
                Resized(keys=["image", "mask_liv", "mask_spl", "mask_kid"], spatial_size=size, mode="trilinear"),
                ConcatMasksd(keys=["mask_liv", "mask_spl", "mask_kid"], output_key="mask"),  # 合併 masks
            ]
        )
    else:
        if mask_classes > 2:
            train_transforms = Compose(
                [
                    LoadImaged(keys=["image","mask"]),
                    EnsureChannelFirstd(keys=["image","mask"]),
                    # RepeatChanneld(keys=["image","label"], repeats = num_sample),
                    ScaleIntensityRanged(
                        # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                        keys=["image"],
                        a_min=-50,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    # Dulicated_new(keys=["image"], num_samples=num_samples, pos_sel=True),
                    Spacingd(keys=["image","mask"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                    Orientationd(keys=["image","mask"], axcodes="RAS"),
                    CropForegroundd(keys=["image","mask"], source_key="image"),
                    Resized(keys=["image","mask"], spatial_size=size, mode=("trilinear")),
                    Rand3DElasticd(
                        keys=["image","mask"],
                        mode=("bilinear"),
                        prob=prob,
                        sigma_range=sigma_range,
                        magnitude_range=magnitude_range,
                        spatial_size=size,
                        translate_range=translate_range,
                        rotate_range=rotate_range,
                        scale_range=scale_range,
                        padding_mode="border",
                    ),
                    TargetedChannelMasksd(keys=["mask"], target_key="label", output_key="mask")
                ]
            )
            valid_transforms = Compose(
                [
                    LoadImaged(keys=["image","mask"]),
                    EnsureChannelFirstd(keys=["image","mask"]),
                    ScaleIntensityRanged(
                        # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                        keys=["image"],
                        a_min=-50,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    Spacingd(keys=["image","mask"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                    Orientationd(keys=["image","mask"], axcodes="RAS"),
                    CropForegroundd(keys=["image","mask"], source_key="image"),
                    Resized(keys=["image","mask"], spatial_size=size, mode=("trilinear")),
                    TargetedChannelMasksd(keys=["mask"], target_key="label", output_key="mask")
                ]
            )
        else:
            train_transforms = Compose(
                [
                    LoadImaged(keys=["image","mask"]),
                    EnsureChannelFirstd(keys=["image","mask"]),
                    # RepeatChanneld(keys=["image","label"], repeats = num_sample),
                    ScaleIntensityRanged(
                        # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                        keys=["image"],
                        a_min=-50,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    # Dulicated_new(keys=["image"], num_samples=num_samples, pos_sel=True),
                    Spacingd(keys=["image","mask"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                    Orientationd(keys=["image","mask"], axcodes="RAS"),
                    CropForegroundd(keys=["image","mask"], source_key="image"),
                    Resized(keys=["image","mask"], spatial_size=size, mode=("trilinear")),
                    Rand3DElasticd(
                        keys=["image","mask"],
                        mode=("bilinear"),
                        prob=prob,
                        sigma_range=sigma_range,
                        magnitude_range=magnitude_range,
                        spatial_size=size,
                        translate_range=translate_range,
                        rotate_range=rotate_range,
                        scale_range=scale_range,
                        padding_mode="border",
                    ),
                ]
            )
            valid_transforms = Compose(
                [
                    LoadImaged(keys=["image","mask"]),
                    EnsureChannelFirstd(keys=["image","mask"]),
                    ScaleIntensityRanged(
                        # keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                        keys=["image"],
                        a_min=-50,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    Spacingd(keys=["image","mask"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                    Orientationd(keys=["image","mask"], axcodes="RAS"),
                    CropForegroundd(keys=["image","mask"], source_key="image"),
                    Resized(keys=["image","mask"], spatial_size=size, mode=("trilinear")),
                ]
            )

    # Training by cross validation
    accuracy_list = []
    test_accuracy_list = []
    file_list = []
    epoch_list = []

    if cross_kfold * data_split_ratio[2] != 1 and cross_kfold != 1:
        raise RuntimeError("Kfold number is not match test data ratio")

    first_start_time = time.time()
    # kfold
    if cross_kfold != 1:
        for k in range(cross_kfold):
            run_once(k)
    # grid search
    elif len(init_lr) != 1:
        for k in range(len(init_lr)):
            run_once(k)
    else:
        run_once()

    if cross_kfold != 1:
        print(f"\n Mean accuracy:{sum(accuracy_list)/len(accuracy_list)}")

    final_end_time = time.time()
    hours, rem = divmod(final_end_time - first_start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    all_time = "All time:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds
    )
    print(all_time)
    # write some output information in ori ini
    conf["Data output"] = {}
    conf["Data output"]["Running time"] = all_time
    conf["Data output"]["Data file name"] = str(file_list)
    # ini write in type need str type
    conf["Data output"]["Best accuracy"] = str(accuracy_list)
    conf["Data output"]["Best Test accuracy"] = str(test_accuracy_list)
    conf["Data output"]["Best epoch"] = str(epoch_list)

    with open(cfgpath, "w") as f:
        conf.write(f)

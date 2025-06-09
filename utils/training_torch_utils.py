import os
import time
import sys
from datetime import datetime
import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import utils.config as config
import matplotlib.pyplot as plt
import os, psutil
import functools
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
from skimage.transform import resize
from tqdm import tqdm, trange
# Based on MONAI 1.1
from monai.transforms.transform import MapTransform
from monai.utils import ensure_tuple_rep
from monai.config import KeysCollection
from typing import Optional
from monai.utils.enums import PostFix
import scipy.ndimage
# let all of print can be flush = ture
print = functools.partial(print, flush=True)

#-------- Dataloder --------
# Based on MONAI 0.4.0
# After augmnetation with resize, crop spleen area and than transofermer 

def train(model, device, data_num, epochs, optimizer, loss_function, train_loader, valid_loader, early_stop, scheduler, check_path):
    # Let ini config file can be writted
    #global best_metric
    #global best_metric_epoch
    #val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    trigger_times = 0
    if early_stop == 0:
        early_stop = None
    #epoch_loss_values = list()
    
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        # record ram memory used
        process = psutil.Process(os.getpid())
        print(f'RAM used:{process.memory_info().rss/ 1024 ** 3} GB')
        model.train()
        epoch_loss = 0
        step = 0
        first_start_time = time.time()
        for batch_data in train_loader:
            step += 1
            if batch_data['label'].dim() == 1:
                labels = batch_data['label'].long().to(device)
            else:
                labels = batch_data['label'].float().to(device)
            if "image_r" in batch_data:
                image_r, image_l = batch_data['image_r'].to(device), batch_data['image_l'].to(device)
                # z axis
                inputs = torch.cat((image_r,image_l), dim=-1)
            else:
#                 inputs = batch_data['image'].permute(0, 1, 4, 2, 3)
                inputs = batch_data['image']
                inputs = inputs.to(device)

            if "bbox" in batch_data:
                bboxs = batch_data['bbox'].to(device)

            optimizer.zero_grad()
            #inputs, labels = Variable(inputs), Variable(labels)
            if "bbox" in batch_data:
                outputs = model(bboxs, inputs)
            else:
                outputs = model(inputs)
            # print(f'outputs:{outputs.size()}')
            # print(f'labels:{labels.size()}')
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = data_num // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        config.epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        final_end_time = time.time()
        hours, rem = divmod(final_end_time-first_start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'one epoch runtime:{int(minutes)}:{seconds}')
        # Early stopping & save best weights by using validation
        metric = validation(model, valid_loader, device)
        scheduler.step(metric)

        # checkpoint setting
        if metric > best_metric:
            # reset trigger_times
            trigger_times = 0
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"{check_path}/{best_metric}.pth")
            print('trigger times:', trigger_times)
            print("saved new best metric model")
        else:
            trigger_times += 1
            print('trigger times:', trigger_times)
            # Save last 3 epoch weight
            if early_stop and early_stop - trigger_times <= 3 or epochs - epoch <= 3:
                torch.save(model.state_dict(), f"{check_path}/{metric}_last.pth")
                print("save last metric model")
        print(
            "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )
        writer.add_scalar("val_accuracy", metric, epoch + 1)

        # early stop 
        if early_stop and trigger_times >= early_stop:
            print('Early stopping!\nStart to test process.')
            break
        
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    config.best_metric = best_metric
    config.best_metric_epoch = best_metric_epoch
    writer.close()
    #print(f'training_torch best_metric:{best_metric}',flush =True)
    #print(f'training_torch config.best_metric:{config.best_metric}',flush =True)
    return model


class AngleLoss_predict(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss_predict, self).__init__()
        self.gamma = gamma
        self.it = 1
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input):
        cos_theta, phi_theta = input
        cos_theta = cos_theta.as_tensor()
        phi_theta = phi_theta.as_tensor()
        #cos_theta = torch.tensor(cos_theta,  requires_grad=True)
        #phi_theta = torch.tensor(phi_theta,  requires_grad=True)
        #target = target.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B, Classnum)
        # index = index.scatter(1, target.data.view(-1, 1).long(), 1)
        #index = index.byte()
        index = index.bool()  
        index = Variable(index)
        # index = Variable(torch.randn(1,2)).byte()

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output1 = output.clone()
        # output1[index1] = output[index] - cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        # output1[index1] = output[index] + phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] = output1[index]- cos_theta[index] * (1.0 + 0) / (1 + self.lamb)+ phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        return(output)


def validation(model, val_loader, device):
    #metric_values = list()
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        total_labels = 0
        num_correct_labels = 0.0
        for val_data in val_loader:
            
            if "image_r" in val_data:
                image_r, image_l, val_labels = val_data['image_r'].to(device), val_data['image_l'].to(device), val_data['label'].to(device)
                # z axis
                val_images = torch.cat((image_r,image_l), dim=-1)
            else:
#                 val_images, val_labels = val_data['image'].permute(0, 1, 4, 2, 3), val_data['label'].to(device)
                val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
#                 val_images = val_images.to(device)
            
            
            if "bbox" in val_data:
                bboxs = val_data['bbox'].to(device)
                val_outputs = model(bboxs, val_images)
            else:
                val_outputs = model(val_images)
            # print(val_outputs.size())
            # base on AngleLoss
            if isinstance(val_outputs, tuple):
                val_outputs = AngleLoss_predict()(val_outputs)
            if val_outputs.size(1) >= 2:
                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                num_correct += value.sum().item()
            else:
                # 将预测分数转换为二进制标签，例如，通过应用阈值
                binary_predictions = (val_outputs > 0.5).float()
                correct_predictions = torch.eq(binary_predictions, val_labels)
                # 计算每个样本的所有标签是否都被正确预测
                all_correct_per_sample = torch.all(correct_predictions, dim=1)
                # 计算正确分类的标签数
                num_correct_labels = correct_predictions.sum().item()
                total_labels = torch.numel(val_labels)
                # 计算正确分类的样本数
                num_correct += all_correct_per_sample.sum().item()
            metric_count += val_outputs.size(0)
        # if 'total_labels' in locals():
        if total_labels !=0:
            label_accuracy = num_correct_labels / total_labels
            print(f'validation num_correct_labels:{label_accuracy}',flush =True)
        metric = num_correct / metric_count
        config.metric_values.append(metric)
        #print(f'validation metric:{config.metric_values}',flush =True)
    return metric

    
def plot_loss_metric(epoch_loss_values, metric_values, save_path, 
                     epoch_ce_loss_values=None, epoch_amse_loss_values=None):
    """
    绘制训练损失和验证指标曲线，如果额外的损失值存在，则一同绘制。
    
    Args:
        epoch_loss_values (list): 训练平均损失值
        metric_values (list): 验证指标（例如准确率）
        save_path (str): 保存图像的路径
        epoch_ce_loss_values (list, optional): 交叉熵损失值
        epoch_amse_loss_values (list, optional): AMSE损失值
    """
    plt.figure("train", (12, 6))
    
    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    plt.plot(x, epoch_loss_values, label="Train Loss", color="blue")
    
    # 如果存在交叉熵损失，绘制它
    if epoch_ce_loss_values:
        plt.plot(x, epoch_ce_loss_values, label="CE Loss", color="green")
        
    # 如果存在 AMSE 损失，绘制它
    if epoch_amse_loss_values:
        plt.plot(x, epoch_amse_loss_values, label="AMSE Loss", color="orange")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 绘制验证指标曲线
    plt.subplot(1, 2, 2)
    plt.title("Validation Accuracy")
    x = [i + 1 for i in range(len(metric_values))]
    plt.plot(x, metric_values, label="Val Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_path}/train_loss_metric.png")
    plt.close()

def kfold_split(file, kfold, seed, type, fold):
    if type == 'pos':
        d = {}
        file_list = ['file']
        file_list.extend([f'pos_split_df_{i}' for i in range(kfold)])
        d['file'] = file
        for i in range(kfold):
            d[f'test_pos_df_{i}'] = d[file_list[i]].groupby(["gender","age_range","spleen_injury_class"],group_keys=False).apply(lambda x: x.sample(frac=1/(kfold-i),random_state=1))
            d[f'pos_split_df_{i}'] = d[file_list[i]].drop(d[f'test_pos_df_{i}'].index.to_list())
        output_file = d[f'test_pos_df_{fold}']

    elif type == 'neg':
        file_list = [f'neg_split_df_{i}' for i in range(kfold)]
        file_list = np.array_split(file.sample(frac=1,random_state=seed), kfold)
        output_file = file_list[fold]
        
    return output_file

            
def plot_loss_metric_new(epoch_loss_values,epoch_metric, metric_loss, metric_values,save_path):
    plt.figure("loss", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(y)
    plt.plot(metric_loss)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    x = [i + 1 for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(epoch_metric)
    plt.plot(y)
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(f'{save_path}/train_loss_metric.png')
    

def dice_score(preds, labels):
    """
    Compute the Dice Score.
    
    Parameters:
    preds (torch.Tensor): Predicted segmentation masks.
    labels (torch.Tensor): Ground truth segmentation masks.
    
    Returns:
    float: Dice Score.
    """
    # Ensure binary prediction
    preds = preds > 0.5
    labels = labels > 0.5
    
    intersection = (preds & labels).float().sum()  # Intersection points
    union = preds.float().sum() + labels.float().sum()  # Union points
    
    if union == 0:
        return torch.tensor(1.0)  # If both are zero, return perfect similarity
    else:
        dice = 2. * intersection / union
        return dice
    

def resize_tensor(input_tensor, target_size):
    """
    Resize the input tensor to the target size using scipy's zoom function.
    
    Parameters:
    input_tensor (torch.Tensor): The input tensor, shape (n, c, d, h, w)
    target_size (tuple): The target size, format (n, c, d, h, w)
    
    Returns:
    torch.Tensor: Resized tensor.
    """
    # Ensure tensor is on CPU and convert to numpy
    input_numpy = input_tensor.cpu().numpy()
    
    # Calculate the zoom factors
    zoom_factors = [n / o for n, o in zip(target_size, input_numpy.shape)]
    
    # Apply zoom
    resized_numpy = scipy.ndimage.zoom(input_numpy, zoom_factors, order=1)  # Use linear interpolation
    
    # Convert back to tensor
    resized_tensor = torch.from_numpy(resized_numpy).to(input_tensor.device)
    
    return resized_tensor    
    
def train_seg(model, device, data_num, epochs, optimizer, loss_function, train_loader, valid_loader, early_stop, scheduler, check_path):
    # Let ini config file can be writted
    #global best_metric
    #global best_metric_epoch
    #val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    trigger_times = 0
    if early_stop == 0:
        early_stop = None
    #epoch_loss_values = list()
    
#     writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        # record ram memory used
        process = psutil.Process(os.getpid())
        print(f'RAM used:{process.memory_info().rss/ 1024 ** 3} GB')
        model.train()
        epoch_loss = 0
        epoch_amse_loss = 0
        epoch_ce_loss = 0
        step = 0
        first_start_time = time.time()
        for batch_data in train_loader:
            step += 1
            if batch_data['label'].dim() == 1:
                labels = batch_data['label'].long().to(device)
            else:
                labels = batch_data['label'].float().to(device)
            labels = batch_data['label'].long().to(device)
            inputs, mask = batch_data['image'].permute(0, 1, 4, 2, 3), batch_data['mask'].permute(0, 1, 4, 2, 3)
            inputs, mask = inputs.to(device), mask.to(device)
            if "bbox" in batch_data:
                bboxs = batch_data['bbox'].to(device)

            optimizer.zero_grad()
            #inputs, labels = Variable(inputs), Variable(labels)
            if "bbox" in batch_data:
                outputs = model(bboxs, inputs)
            else:
                outputs, pred_mask = model(inputs)
            # print(f'outputs:{outputs.size()}')
            # print(f'labels:{labels.size()}')
            gt_mask = resize_tensor(mask,pred_mask.shape)
            loss, amse_loss, ce_loss = loss_function(outputs, labels,pred_mask, gt_mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_amse_loss += amse_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_len = data_num // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
#             writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_amse_loss /= step
        epoch_ce_loss /= step
        config.epoch_loss_values.append(epoch_loss)
        config.epoch_amse_loss_values.append(epoch_amse_loss)
        config.epoch_ce_loss_values.append(epoch_ce_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, average AMSE loss: {epoch_amse_loss:.4f}, average CE loss: {epoch_ce_loss:.4f}")
        final_end_time = time.time()
        hours, rem = divmod(final_end_time-first_start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'one epoch runtime:{int(minutes)}:{seconds}')
        # Early stopping & save best weights by using validation
        metric = validation_seg(model, valid_loader, device)
        scheduler.step(metric)

        # checkpoint setting
        if metric >= best_metric:
            # reset trigger_times
            trigger_times = 0
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"{check_path}/{best_metric}.pth")
            print('trigger times:', trigger_times)
            print("saved new best metric model")
        else:
            trigger_times += 1
            print('trigger times:', trigger_times)
            # Save last 3 epoch weight
            if early_stop and early_stop - trigger_times <= 3 or epochs - epoch <= 3:
                torch.save(model.state_dict(), f"{check_path}/{metric}_last.pth")
                print("save last metric model")
        print(
            "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )
#         writer.add_scalar("val_accuracy", metric, epoch + 1)

        # early stop 
        if early_stop and trigger_times >= early_stop:
            print('Early stopping!\nStart to test process.')
            break
        
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    config.best_metric = best_metric
    config.best_metric_epoch = best_metric_epoch
#     writer.close()
    #print(f'training_torch best_metric:{best_metric}',flush =True)
    #print(f'training_torch config.best_metric:{config.best_metric}',flush =True)
    return model

    
def validation_seg(model, val_loader, device):
    #metric_values = list()
    model.eval()
    pre_first = True
    with torch.no_grad():
        metric_count = 0
        total_dice = 0
        num_correct_labels = 0.0
        sigmoid = nn.Sigmoid()
        for val_data in val_loader:
            val_images, val_labels, val_mask = val_data['image'].permute(0, 1, 4, 2, 3), val_data['label'].to(device), val_data['mask'].permute(0, 1, 4, 2, 3)
            val_images, val_mask = val_images.to(device), val_mask.to(device)
            if "bbox" in val_data:
                bboxs = val_data['bbox'].to(device)
                val_outputs = model(bboxs, val_images)
            else:
                val_outputs, val_pred_mask = model(val_images)
            # print(val_outputs.size())
            # base on AngleLoss
            if isinstance(val_outputs, tuple):
                val_outputs = AngleLoss_predict()(val_outputs)
            if val_outputs.size(1) >= 2:
#                 print(val_pred_mask.shape)
                if val_pred_mask.size(1) >= 2 and val_outputs.dim() == 3:
                    val_outputs = nn.functional.softmax(val_outputs,dim=2)
                    val_pred_mask = nn.functional.softmax(val_pred_mask,dim=1)
                    gt_mask = resize_tensor(val_mask,val_pred_mask.shape)
                    dice = dice_score(val_pred_mask,gt_mask)
                    total_dice+=dice
                elif val_pred_mask.size(1) >= 2:
                    val_outputs = nn.functional.softmax(val_outputs,dim=1)
                    val_pred_mask = nn.functional.softmax(val_pred_mask,dim=1)
                    gt_mask = resize_tensor(val_mask,val_pred_mask.shape)
                    dice = dice_score(val_pred_mask,gt_mask)
                    total_dice+=dice
                    
                else:
                    val_outputs = nn.functional.softmax(val_outputs,dim=1)
                    val_pred_mask = sigmoid(val_pred_mask)
                    gt_mask = resize_tensor(val_mask,val_pred_mask.shape)
                    dice = dice_score(val_pred_mask,gt_mask)
                    total_dice+=dice
#                 value = torch.eq(val_outputs.argmax(dim=1), val_labels)
#                 num_correct += value.sum().item()
            else:
                val_outputs = nn.functional.softmax(val_outputs,dim=1).cpu().detach().numpy()
#                 val_outputs = sigmoid(val_outputs)
                val_pred_mask = sigmoid(val_pred_mask)
                gt_mask = resize_tensor(val_mask,val_pred_mask.shape)
                dice = dice_score(val_pred_mask,gt_mask)
                total_dice+=dice
                
            if pre_first:
                pre_first = None
                predict_values = val_outputs.cpu().detach().numpy()
                gt = val_labels.cpu().detach().numpy()
            else:
                predict_values = np.concatenate((predict_values,val_outputs),axis=0)
                gt = np.concatenate((gt,val_labels.cpu().detach().numpy()),axis=0)
#             print(predict_values)
#             print(gt)
            metric_count += 1.0
        if val_outputs.dim() > 2:
            precision_liv, recall_liv, f1_liv = f1_score(predict_values[:, 0, :], gt[...,0])
            print(f"liver precision: {precision_liv}, recall: {recall_liv}, f1: {f1_liv}")
            precision_spl, recall_spl, f1_spl = f1_score(predict_values[:, 1, :], gt[...,1])
            print(f"spleen precision: {precision_spl}, recall: {recall_spl}, f1: {f1_spl}")
            precision_kid, recall_kid, f1_kid = f1_score(predict_values[:, 2, :], gt[...,2])
            print(f"kidney precision: {precision_kid}, recall: {recall_kid}, f1: {f1_kid}")
            predict_values_total = predict_values.reshape(-1, 2)
            gt_total = gt.flatten()
            precision, recall, f1 = f1_score(predict_values_total, gt_total)
        else:
            precision, recall, f1 = f1_score(predict_values, gt)
        metric_value = f1
        metric = metric_value.item() if isinstance(metric_value, torch.Tensor) else float(metric_value)

        dice_scores_valu = total_dice / metric_count
        dice_scores = dice_scores_valu.item() if isinstance(dice_scores_valu, torch.Tensor) else float(dice_scores_valu)
        config.metric_values.append(metric)
        print(f'validation dice:{dice_scores}',flush =True)
        print(f"precision: {precision}, recall: {recall}, f1: {f1}")
        print(f'validation metric:{metric}',flush =True)
        
    return metric

import torch
import torch.nn.functional as F
import time
import numpy as np
from torch import nn

from utils.data_utils import resize_tensor, process_mask_with_thresholding
from utils.metrics import f1_score, dice_score, mean_and_confidence_interval

class InferenceRunner:
    def __init__(self, model, device, dataloader, mode="classification", mask_threshold = 0.5):
        self.model = model.to(device)
        self.device = device
        self.dataloader = dataloader
        self.mode = mode.lower()  # 確保大小寫不影響判斷
        self.threshold = mask_threshold

    def run(self):
        if self.mode == 'classification':
            return self.inference_classification(self.dataloader)
        elif self.mode == 'segmentation':
            return self.inference_segmentation(self.dataloader, self.threshold)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def inference_classification(self, dataloader):
        self.model.eval()
        num_correct = 0
        total_samples = 0
        predict_values = []
        gt_values = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = self._prepare_input(batch)
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)
                pre = nn.functional.softmax(output,dim=1).cpu().detach().numpy()

                predict_values.append(pre)
                gt_values.append(labels.cpu().detach().numpy())

                

                if outputs.size(1) >= 2:
                    preds = outputs.argmax(dim=1)
                    correct = torch.eq(preds, labels).sum().item()
                else:
                    preds = (outputs > 0.5).float()
                    correct = torch.all(preds == labels, dim=1).sum().item()

                num_correct += correct
                total_samples += labels.size(0)

        predict_values = np.concatenate(predict_values, axis=0)
        gt_values = np.concatenate(gt_values, axis=0)

        accuracy = num_correct / total_samples
        prec, rec, f1 = f1_score(predict_values, gt_values)
        print(f"Classification Accuracy: {accuracy:.4f}, f1: {f1:.4f}")
        return (predict_values)

    def inference_segmentation(self, dataloader, threshold):
        self.model.eval()
        total_dice = 0.0
        num_batches = 0
        predict_values = []
        gt_values = []
        sigmoid = nn.Sigmoid()

        image_list, mask_list, dice_data = [], [], []
        dice_liv, dice_spl, dice_kid = [], [], []
        dice_liv_neg, dice_spl_neg, dice_kid_neg = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                images = self._prepare_input(batch)
                masks = batch["mask"].permute(0, 1, 4, 2, 3).to(self.device)
                labels = batch['label'].to(self.device)

                outputs, pred_masks = self.model(images)
                outputs_softmax = F.softmax(outputs, dim=2 if outputs.dim() >= 3 else 1)

                predict_values.append(outputs_softmax.cpu().numpy())
                gt_values.append(labels.cpu().numpy())

                pred_masks = sigmoid(pred_masks)
                pred_mask_resized  = resize_tensor(pred_masks,masks.shape)

                for i in range(labels.size(0)):
                    gt_mask = batch["mask"][i].numpy().squeeze()
                    image = batch["image"][i].numpy().squeeze()
                    mask_np = pred_mask_resized[i].permute(0, 2, 3, 1).cpu().numpy()

                    if pred_mask_resized.size(1) == 6:
                        dice_liv_neg.append(dice_score(mask_np[0], gt_mask[0]))
                        dice_liv.append(dice_score(mask_np[1], gt_mask[1]))
                        dice_spl_neg.append(dice_score(mask_np[2], gt_mask[2]))
                        dice_spl.append(dice_score(mask_np[3], gt_mask[3]))
                        dice_kid_neg.append(dice_score(mask_np[4], gt_mask[4]))
                        dice_kid.append(dice_score(mask_np[5], gt_mask[5]))
                        if torch.any(labels[i] == 1):
                            mask_components = [
                                process_mask_with_thresholding(mask_np[1].squeeze(), threshold),
                                process_mask_with_thresholding(mask_np[3].squeeze(), threshold),
                                process_mask_with_thresholding(mask_np[5].squeeze(), threshold),
                            ]
                            mask_list.append(np.maximum.reduce(mask_components))
                            image_list.append(image)

                    elif pred_mask_resized.size(1) == 3:
                        if any(labels[i].cpu().numpy()):
                            components = []
                            for j in range(pred_mask_resized.size(1)):
                                if labels[i][j]:
                                    components.append(process_mask_with_thresholding(mask_np[j]))
                                    if j == 0:
                                        dice_liv.append(dice_score(mask_np[j], gt_mask[j]))
                                    elif j == 1:
                                        dice_spl.append(dice_score(mask_np[j], gt_mask[j]))
                                    elif j == 2:
                                        dice_kid.append(dice_score(mask_np[j], gt_mask[j]))
                            mask_list.append(np.maximum.reduce(components))
                            image_list.append(image)
                    else:
                        if labels[i] == 1:
                            mask_bin = process_mask_with_thresholding(mask_np.squeeze())
                            dice_data.append(dice_score(mask_np.squeeze(), gt_mask.squeeze()))
                            mask_list.append(mask_bin)
                            image_list.append(image)

            predict_values = np.concatenate(predict_values, axis=0)
            gt_values = np.concatenate(gt_values, axis=0)

            if pred_masks.size(1) > 2 and outputs_softmax.size(1)==3:
                organ_list = ["Liver","Spleen","Kidney"]
                for i in range(len(organ_list)):
                    precision, recall, f1 = f1_score(predict_values[:, i, :], gt_values[...,i])
                    print(f"{organ_list[i]} precision: {precision}, recall: {recall}, f1: {f1}")

                    liver_dice, liver_ci_low, liver_ci_high = mean_and_confidence_interval(dice_liv)
                    spleen_dice, spleen_ci_low, spleen_ci_high = mean_and_confidence_interval(dice_spl)
                    kidney_dice, kidney_ci_low, kidney_ci_high = mean_and_confidence_interval(dice_kid)
                    
                    print(f"Liver dice: {liver_dice:.4f} ({liver_ci_low:.4f} to {liver_ci_high:.4f}), "
                        f"Spleen dice: {spleen_dice:.4f} ({spleen_ci_low:.4f} to {spleen_ci_high:.4f}), "
                        f"Kidney dice: {kidney_dice:.4f} ({kidney_ci_low:.4f} to {kidney_ci_high:.4f})")
                    

                    if dice_liv_neg:
                        liver_dice, liver_ci_low, liver_ci_high = mean_and_confidence_interval(dice_liv_neg)
                        spleen_dice, spleen_ci_low, spleen_ci_high = mean_and_confidence_interval(dice_spl_neg)
                        kidney_dice, kidney_ci_low, kidney_ci_high = mean_and_confidence_interval(dice_kid_neg)
                        print(f"Liver negative dice: {liver_dice:.4f} ({liver_ci_low:.4f} to {liver_ci_high:.4f}), "
                            f"Spleen negative dice: {spleen_dice:.4f} ({spleen_ci_low:.4f} to {spleen_ci_high:.4f}), "
                            f"Kidney negative dice: {kidney_dice:.4f} ({kidney_ci_low:.4f} to {kidney_ci_high:.4f})") 
                precision, recall, f1 = f1_score(predict_values.reshape(-1, 2), gt_values.flatten()) 
                print("precision: {precision}, recall: {recall}, f1: {f1}")      
            else:
                precision, recall, f1 = f1_score(predict_values, gt_values)
                print("precision: {precision}, recall: {recall}, f1: {f1}")
                dice, ci_low, ci_high = mean_and_confidence_interval(dice_data)
                print(f"Segmentation Dice: {dice:.4f} ({ci_low:.4f} to {ci_high:.4f})")

        return (predict_values), image_list, mask_list

    def _prepare_input(self, batch):
        if "image_r" in batch and "image_l" in batch:
            images = torch.cat(
                [batch["image_r"].to(self.device), batch["image_l"].to(self.device)],
                dim=-1
            )
            return images.permute(0, 1, 4, 2, 3)  # BCHWD → BCDHW
        else:
            return batch["image"].permute(0, 1, 4, 2, 3).to(self.device)
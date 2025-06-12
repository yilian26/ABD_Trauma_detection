import torch
import os
import time
import numpy as np
import psutil
import torch.nn.functional as F
from torch import nn
from utils.data_utils import resize_tensor
from utils.metrics import f1_score, dice_score
from config.config_loader import TrainingProgressTracker

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, best_metric_epoch, trigger_times, check_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "trigger_times": trigger_times,
    }
    torch.save(checkpoint, os.path.join(check_path, "latest_checkpoint.pth"))

class Trainer:
    def __init__(
        self,
        conf,
        model,
        loss_function,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        checkpoint_path,
        mode="classification",
    ):
        self.conf = conf
        self.model = model
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mode = mode.lower()
        self.checkpoint_path = checkpoint_path
        self.epochs = self.conf.data_setting.epochs
        self.early_stop = self.conf.data_setting.early_stop or None
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.trigger_times = 0
        self.tracker = TrainingProgressTracker()

    def run(self, times=0):
        if self.mode == 'classification':
            self._run_classification()
        elif self.mode == 'segmentation':
            self._run_segmentation()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _prepare_input(self, batch):
        if "image_r" in batch and "image_l" in batch:
            images = torch.cat(
                [batch["image_r"].to(self.device), batch["image_l"].to(self.device)],
                dim=-1
            )
            return images.permute(0, 1, 4, 2, 3)  # BCHWD → BCDHW
        else:
            return batch["image"].permute(0, 1, 4, 2, 3).to(self.device)

    def _run_classification(self):
        for epoch in range(self.epochs):
            process = psutil.Process(os.getpid())
            print(f"Epoch {epoch + 1}/{self.epochs}, RAM used:{process.memory_info().rss/ 1024 ** 3} GB")
            self.model.train()
            epoch_loss = 0
            step = 0
            first_start_time = time.time()

            for batch in self.train_loader:
                inputs = self._prepare_input(batch)
                if batch['label'].dim() == 1:
                    labels = batch['label'].long().to(self.device)
                else:
                    labels = batch['label'].float().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                step += 1

            epoch_loss /= step
            self.tracker.epoch_loss_values.append(epoch_loss)
            epoch_len = len(self.train_loader) * self.train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            
            final_end_time = time.time()
            hours, rem = divmod(final_end_time-first_start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f'one epoch runtime:{int(minutes)}:{seconds}')
            

            evaluator = Evaluator(device=self.device, tracker=self.tracker)
            metric = evaluator.evaluate_classification(self.model, self.val_loader)
            self.scheduler.step(metric)
            self._checkpoint(metric, epoch)
            save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, self.best_metric, self.best_metric_epoch, self.trigger_times, self.checkpoint_path)
            if self._early_stop():
                break

    def _run_segmentation(self):
        for epoch in range(self.epochs):
            process = psutil.Process(os.getpid())
            print(f"Epoch {epoch + 1}/{self.epochs}, RAM used:{process.memory_info().rss/ 1024 ** 3} GB")
            self.model.train()
            epoch_loss, epoch_ce, epoch_amse = 0, 0, 0
            step = 0
            first_start_time = time.time()
            for batch in self.train_loader:
                inputs = batch['image'].permute(0, 1, 4, 2, 3).to(self.device)
                masks = batch['mask'].permute(0, 1, 4, 2, 3).to(self.device)
                if batch['label'].dim() == 1:
                    labels = batch['label'].long().to(self.device)
                else:
                    labels = batch['label'].float().to(self.device)

                self.optimizer.zero_grad()
                outputs, pred_masks = self.model(inputs)
                gt_masks = resize_tensor(masks, pred_masks.shape)

                loss, amse_loss, ce_loss = self.loss_function(outputs, labels, pred_masks, gt_masks)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_ce += ce_loss.item()
                epoch_amse += amse_loss.item()
                step += 1
                epoch_len = len(self.train_loader) * self.train_loader.batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

            epoch_loss /= step
            epoch_ce /= step
            epoch_amse /= step

            self.tracker.epoch_loss_values.append(epoch_loss)
            self.tracker.epoch_ce_loss_values.append(epoch_ce)
            self.tracker.epoch_amse_loss_values.append(epoch_amse)

            print(f"epoch {epoch + 1} Loss: {epoch_loss:.4f}, CE: {epoch_ce:.4f}, AMSE: {epoch_amse:.4f}")
            
            final_end_time = time.time()
            hours, rem = divmod(final_end_time-first_start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f'one epoch runtime:{int(minutes)}:{seconds}')

            evaluator = Evaluator(device=self.device, tracker=self.tracker)
            metric = evaluator.evaluate_segmentation(self.model, self.val_loader)
            
            self.scheduler.step(metric)
            self._checkpoint(metric, epoch)
            save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, self.best_metric, self.best_metric_epoch, self.trigger_times, self.checkpoint_path)
            
            if self._early_stop():
                break
            
        print(f"train completed, best_metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}")
        self.tracker.best_metric = self.best_metric
        self.tracker.best_metric_epoch = self.best_metric_epoch

    def _checkpoint(self, metric, epoch):
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_metric_epoch = epoch + 1
            self.trigger_times = 0
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f"{self.best_metric}.pth"))
            print("Saved new best model")
        else:
            self.trigger_times += 1
            print(f"Trigger times: {self.trigger_times}")
            if self.early_stop and self.early_stop - self.trigger_times <= 3:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f"{metric}_last.pth"))
                print("save last metric model")

    def _early_stop(self):
        if self.early_stop and self.trigger_times >= self.early_stop:
            print("Early stopping triggered")
            return True
        return False


class Evaluator:
    def __init__(self, device: torch.device, tracker: TrainingProgressTracker = None):
        self.device = device
        self.tracker = tracker if tracker is not None else TrainingProgressTracker()

    def evaluate_classification(self, model, dataloader) -> float:
        first_start_time = time.time()
        model.eval()
        num_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = self._prepare_input(batch)
                labels = batch["label"].to(self.device)

                outputs = model(inputs)

                if outputs.size(1) >= 2:
                    preds = outputs.argmax(dim=1)
                    correct = torch.eq(preds, labels).sum().item()
                else:
                    preds = (outputs > 0.5).float()
                    correct = torch.all(preds == labels, dim=1).sum().item()

                num_correct += correct
                total_samples += labels.size(0)

        metric = num_correct / total_samples
        self.tracker.metric_values.append(metric)
        print(f'validation metric:{config.metric_values}',flush =True)
        return metric

    def evaluate_segmentation(self, model, dataloader) -> float:
        first_start_time = time.time()
        model.eval()
        total_dice = 0.0
        num_batches = 0
        predict_values = []
        gt_values = []
        sigmoid = nn.Sigmoid()
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].permute(0, 1, 4, 2, 3).to(self.device)
                masks = batch["mask"].permute(0, 1, 4, 2, 3).to(self.device)
                labels = batch['label'].to(self.device)

                outputs, pred_masks = model(images)

                if outputs.dim() >= 3:
                    outputs = F.softmax(outputs,dim=2)
                else:
                    outputs = F.softmax(outputs,dim=1)

                predict_values.append(outputs.cpu().detach().numpy())
                gt_values.append(labels.cpu().detach().numpy())

                pred_masks = sigmoid(pred_masks)
                resized_gt = resize_tensor(masks, pred_masks.shape)
                dice = dice_score(pred_masks, resized_gt)
                total_dice += dice.item()
                num_batches += 1

        avg_dice = total_dice / num_batches
        predict_values = np.concatenate(predict_values, axis=0)
        gt_values = np.concatenate(gt_values, axis=0)

        if predict_values.ndim > 2:  # multi-label classification
            if predict_values.shape[1] == 3:
                organ_list = ["Liver", "Spleen", "Kidney"]
            elif predict_values.shape[1] == 5:
                organ_list = ["Liver", "Spleen", "Kidney", "Bowel", "Extravasation"]
            else:
                organ_list = [f"Class_{i}" for i in range(predict_values.shape[1])]

            for i in range(len(organ_list)):
                prec, rec, f1 = f1_score(predict_values[:, i], gt_values[..., i])
                print(f"{organ_list[i]} precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

            # flattened version for macro metric
            predict_flat = predict_values.reshape(-1, predict_values.shape[-1])
            gt_flat = gt_values.flatten()
            precision, recall, f1 = f1_score(predict_flat, gt_flat)
        else:
            precision, recall, f1 = f1_score(predict_values, gt_values)

        metric = float(f1)
        self.tracker.metric_values.append(metric)

        final_end_time = time.time()
        minutes, seconds = divmod(final_end_time - first_start_time, 60)
        print(f'validation one epoch runtime: {int(minutes)}:{seconds:.2f}')
        print(f'validation dice: {avg_dice:.4f}')
        print(f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
        print(f'validation metric: {metric:.4f}')

        return metric


    def _prepare_input(self, batch):
        if "image_r" in batch and "image_l" in batch:
            images = torch.cat(
                [batch["image_r"].to(self.device), batch["image_l"].to(self.device)],
                dim=-1
            )
            return images.permute(0, 1, 4, 2, 3)  # BCHWD → BCDHW
        else:
            return batch["image"].permute(0, 1, 4, 2, 3).to(self.device)


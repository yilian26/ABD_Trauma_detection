import subprocess
import sys
import gc
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("adjustText")

import os
import cv2
import numpy as np
from tqdm.auto import tqdm
import concurrent.futures
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from utils.metrics import Find_Optimal_Cutoff, get_roc_CI


def plot_loss_metric(epoch_loss_values, metric_values, save_path, 
                     epoch_ce_loss_values=None, epoch_amse_loss_values=None):
    """
    Plot training loss and validation metrics. Optionally include CE and AMSE losses.
    
    Args:
        epoch_loss_values (list): Average training loss per epoch.
        metric_values (list): Validation metrics (e.g., accuracy) per epoch.
        save_path (str): Path to save the generated plot.
        epoch_ce_loss_values (list, optional): Cross-entropy loss values per epoch.
        epoch_amse_loss_values (list, optional): AMSE loss values per epoch.
    """
    plt.figure("train", (12, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    plt.plot(x, epoch_loss_values, label="Train Loss", color="blue")
    
    # Plot CE loss if provided
    if epoch_ce_loss_values:
        plt.plot(x, epoch_ce_loss_values, label="CE Loss", color="green")
        
    # Plot AMSE loss if provided
    if epoch_amse_loss_values:
        plt.plot(x, epoch_amse_loss_values, label="AMSE Loss", color="orange")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot validation metric
    plt.subplot(1, 2, 2)
    plt.title("Validation Accuracy")
    x = [i + 1 for i in range(len(metric_values))]
    plt.plot(x, metric_values, label="Val Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{save_path}/train_loss_val_metric.png")
    plt.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     fontsize=14,
                     color="black")
    for i in range (cm.shape[0]):
        for j in range (cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize =14,
                 color="white" if (i == 0 and j == 0) else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_confusion_matrix_multi(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     fontsize=14,
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



def plot_multiple_confusion_matrices(cms, classes_list, normalize=False, title=['Confusion matrix'], cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrices for multiple classes.
    `cms` is an array of confusion matrices.
    `classes_list` is an array of class labels for each matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Set up the subplots grid: 3 rows, ceil(7/3) columns
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the 2D array to easily iterate over it

    for idx, (cm, classes) in enumerate(zip(cms, classes_list)):
        ax = axes[idx]  # Get the subplot for this confusion matrix

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            ax.set_title("Normalized confusion matrix\n" + title + f" {idx + 1}")
        else:
            ax.set_title(title[idx])

        cax = ax.matshow(cm, interpolation='nearest', cmap=cmap)
        fig.colorbar(cax, ax=ax)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticklabels(classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        fontsize=14,
                        color="white" if cm[i, j] > thresh else "black")

    # Turn off unused subplots
    for idx in range(len(cms), len(axes)):
        axes[idx].axis('off')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_roc(y_pre, y_label, title, dir_path, file_name):
    scores = list()
    # 正樣本的數值輸出
    for i in range(y_pre.shape[0]):
        scores.append(y_pre[i][1])
    scores = np.array(scores)
    y_label = y_label.astype(int)
    fpr, tpr, _ = roc_curve(y_label, scores)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(y_label, scores)
    roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(y_label, scores)
    fig = plt.figure(figsize=(6, 6))
    lw = 2
    conf_int = ' ({:.3f}-{:.3f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
    test = f'AUC:{roc_auc:.3f}\n95% CI, {conf_int}'
    plt.plot(fpr, tpr, lw=lw, color='k', label=test)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.plot(optimal_point[0], optimal_point[1], marker = 'o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.3f}')
    ticks = np.linspace(0, 1, 11)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1, color='b')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid() # 網格
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title+' ROC-AUC')
    plt.legend(loc="lower right")
    fig.savefig(f"{dir_path}/{file_name}_roc.png")
    plt.close()

    return optimal_th 

def plot_multi_class_roc(y_pre, y_label, n_classes, cls_type, dir_path, file_name):
    fig = plt.figure(figsize=(6, 6))
    lw = 2
    y_label = np.array(y_label)
    y_pre = np.array(y_pre)
    optimal_th_list = []
    texts = []

    for i in range(n_classes):
        y_true_binary = np.where(y_label == i, 1, 0)
        y_pre_binary = y_pre[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_pre_binary)
        roc_auc = auc(fpr, tpr)
        optimal_th, optimal_point = Find_Optimal_Cutoff(y_true_binary, y_pre_binary)
        optimal_th_list.append(optimal_th)
        roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(y_true_binary, y_pre_binary)

        conf_int = ' ({:.3f}-{:.3f})'.format(np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
        plot_label = f'Class {i} AUC:{roc_auc:.3f}\n95% CI, {conf_int}'
        plt.plot(fpr, tpr, lw=lw, label=plot_label)
        plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
        text = plt.text(optimal_point[0], optimal_point[1], f'Class {i} Threshold:{optimal_th:.3f}')
        texts.append(text)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1, color='b')

    ticks = np.linspace(0, 1, 11)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC of {cls_type} for each class')
    plt.legend(loc="lower right")

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    fig.savefig(f"{dir_path}/{file_name}_{cls_type}_roc.png")
    plt.close()

    return optimal_th_list

def plot_dis(pos_list, neg_list, dir_path, file_name):      
    plt.hist(pos_list, alpha=.5, label='Pos')
    plt.hist(neg_list, alpha=.5, label='Neg')
    plt.title("Data distributions")
    plt.legend(loc="upper right")
    plt.savefig(f"{dir_path}/{file_name}_dis.png")
    plt.close()

def plot_heatmap_detail(heatmap, img, save_path):
    
    fig, ax = plt.subplots(1, 2, figsize = (10,20))
    plt.axis('off') 
    # 水平翻轉跟順時鐘旋轉 (原本為RAS)
    img = cv2.flip(img, 1)
    heatmap = cv2.flip(heatmap, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
    heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE) 
#     heatmap = np.uint8(255 * heatmap)
    
    norm_heatmap_show = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_heatmap_show = np.uint8(norm_heatmap_show)  # 确保类型为uint8
    
    # 以 0.6 透明度繪製原始影像
    ax[0].imshow(img, cmap ='bone')
    ax[0].set_axis_off()
    # 以 0.4 透明度繪製熱力圖
    ax[1].imshow(img, cmap ='bone')
    ax[1].imshow(norm_heatmap_show, cmap ='jet', alpha=0.4)
    ax[1].set_axis_off()
    #plt.title(pred_class_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()

def plot_heatmap_one_picture(heatmap, img, save_path, fig_size=(5,100)):
    fig, ax = plt.subplots(heatmap.shape[2], 2, figsize=fig_size, constrained_layout=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    
    for i in range(heatmap.shape[2]):
        # 水平翻转和顺时针旋转 (原本为RAS)
        img_show = cv2.flip(img[:, :, i], 1)
        heatmap_show = cv2.flip(heatmap[:, :, i], 1)
        
        img_show = cv2.rotate(img_show, cv2.ROTATE_90_CLOCKWISE)
        heatmap_show = cv2.rotate(heatmap_show, cv2.ROTATE_90_CLOCKWISE)
        
        # 归一化处理，确保值在0-255范围内
        norm_heatmap_show = cv2.normalize(heatmap_show, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        norm_heatmap_show = np.uint8(norm_heatmap_show)  # 确保类型为uint8
        
        ax[i, 0].imshow(img_show, cmap='bone')
        ax[i, 0].axis('off')  # 使用 `axis('off')` 替代 `set_axis_off()` 以保持代码风格一致
        
        ax[i, 1].imshow(img_show, cmap='bone')
        ax[i, 1].imshow(norm_heatmap_show, cmap='jet', alpha=0.4)  # 使用归一化后的热图
        ax[i, 1].axis('off')
    
    fig.savefig(save_path)
    plt.close(fig)

def plot_vedio(path):
    """
    將指定目錄下的 PNG 檔案依序讀入並寫入為 AVI 影片
    """
    # 取得檔案清單（僅限 png）
    files = sorted([f for f in os.listdir(path) if f.endswith(".png")])

    if not files:
        print(f"No PNG files found in {path}, skipping video generation.")
        return

    # 讀取第一張圖像確認尺寸
    first_img_path = os.path.join(path, files[0])
    img = cv2.imread(first_img_path)

    if img is None:
        print(f"Failed to read first image: {first_img_path}")
        return

    size = (img.shape[1], img.shape[0])  # (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 設定影片輸出路徑
    vedio_path_list = path.split('/')
    vedio_path_list.insert(-1, 'video')
    video_path = f'{"/".join(vedio_path_list)}.avi'

    dir_path = '/'.join(vedio_path_list[:-1])
    os.makedirs(dir_path, exist_ok=True)

    # 建立影片寫入器
    video = cv2.VideoWriter(video_path, fourcc, 20, size)

    for fname in files:
        file_path = os.path.join(path, fname)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Skipping unreadable file: {file_path}")
            continue
        if img.shape[:2] != size[::-1]:  # 高寬不一致會報錯
            img = cv2.resize(img, size)
        video.write(img)

    video.release()
    print(f"Video saved to {video_path}")


def generate_segmentation_heatmaps(
    test_df,
    image_list,
    mask_list,
    class_type,
    dir_path,
    model_type="segmentation"
):
    """
    Generate heatmap visualizations from segmentation model outputs.

    Args:
        test_df (pd.DataFrame): DataFrame containing test metadata.
        image_list (List[np.ndarray]): List of predicted image volumes.
        mask_list (List[np.ndarray]): List of predicted mask volumes.
        class_type (str): Type of class (e.g., liver, spleen, kidney, all, multiple).
        dir_path (str): Path to output directory.
        model_type (str): Model type, should be "segmentation" to trigger heatmap generation.
    """
    if model_type != "segmentation":
        print("Model type is not segmentation, skipping heatmap generation.")
        return

    if class_type in ["liver", "kidney", "spleen"]:
        filter_col = f"{class_type}_label"
        df = test_df[test_df[filter_col] > 0]
    else:
        df = test_df[test_df["inj_solid"] > 0]

    assert len(df) == len(image_list), "Mismatch between df and image_list lengths"
    print(f"Generating heatmaps for {len(image_list)} samples...")

    name_list = df["file_id"].values
    for i in range(len(image_list)):
        total_final_path = os.path.join(dir_path, "seg_heatmap", name_list[i])
        os.makedirs(total_final_path, exist_ok=True)

        # Save total view heatmap
        plot_heatmap_one_picture(
            mask_list[i], image_list[i], f"{total_final_path}/total_view.png"
        )

        # Save each slice
        for j in range(image_list[i].shape[-1]):
            plot_heatmap_detail(
                mask_list[i][:, :, j],
                image_list[i][:, :, j],
                f"{total_final_path}/{j:03}.png"
            )

        # Generate video
        plot_vedio(total_final_path)

    print("Heatmap generation finished.")

def process_plot_detail(j, heatmap_total, image, final_path):
    plot_heatmap_detail(
        heatmap_total[:, :, j], image[:, :, j], f"{final_path}/{j:03}.png"
    )


def process_plot_multiprocess(heatmap_total, image, final_path, num_cores=10):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_cores
    ) as executor:  # 創建線程池
        futures = [
            executor.submit(process_plot_detail, j, heatmap_total, image, final_path)
            for j in range(image.shape[-1])
        ]  # 將函數和相應的參數提交給線程池執行
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="GradCam plot progressing",
        ):  # 確認函數迭代已完成的任務並用tqdm進行進度調顯示
            pass
    plot_vedio(final_path)
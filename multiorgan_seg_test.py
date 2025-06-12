import os
import gc
import math
import time
import torch
import argparse
import functools
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from models import prepare_model
from utils.loss import LossBuilder
from utils.visualizer import plot_loss_metric
from model_engine.inference import InferenceRunner
from config.config_loader import ConfigLoader
from data.dataset_builder import data_split, load_cgmh_data, load_rsna_data, DataframeCombiner
from data.transforms import get_valid_transforms
from data.data_loaders import test_loaders
from utils.data_utils import str2num_or_false, prepare_labels, str2mode
from utils.metrics import confusion_matrix_CI, confusion_matrix_CI_multi

from utils.visualizer import (
    plot_confusion_matrix, 
    plot_heatmap_one_picture, 
    plot_roc,
    plot_dis,
    plot_multi_class_roc,
    plot_confusion_matrix_multi,
    generate_segmentation_heatmaps,

)


# let all of print can be flush = ture
print = functools.partial(print, flush=True)


def get_parser():
    parser = argparse.ArgumentParser(description="ABD Trauma detection")
    parser.add_argument("-c", "--class_type", type=str, help="The class of data. (liver, kidney, spleen, all, multiple)")
    parser.add_argument("-f", "--file", type=str, help="The config file name")
    parser.add_argument("-s", "--select", type=str2num_or_false, default=False, help="The selection of data file number")
    parser.add_argument("-l", "--label", action="store_const", const=True, default=False, help="The Cam map show as label")
    parser.add_argument("-m", "--mode", type=str2mode, default="segmentation", help="Model mode: 0/cls=classification, 1/seg=segmentation")
    return parser




def run_evaluation(times=0):
    setting = conf.data_setting
    result_info = conf.read_output_section()

    # Split
    _, _, test_df_rsna = data_split(
        df_all_rsna, test_data=test_data, test_fix=None, source="RSNA", ratio=setting.data_split_ratio, seed=setting.seed
    )
    _, _, test_df_cgmh = data_split(
        df_all_cgmh,test_data=None, test_fix=2016, source="CGMH", ratio=setting.data_split_ratio, seed=setting.seed
    )

    # test_df_rsna = test_df_rsna[:50]
    # test_df_cgmh = test_df_cgmh[:50]

    combiner = DataframeCombiner(df_cgmh=test_df_cgmh, df_rsna=test_df_rsna, n_classes = setting.n_classes)
    test_df = combiner.combine()

    # Transforms
    test_transforms = get_valid_transforms(class_type=class_type, cfg = conf)
    test_loader =  test_loaders(conf, test_df_cgmh, test_df_rsna, data_source="all", class_type=class_type, test_transforms=test_transforms)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prepare_model(conf, device, resnet_depth=50, n_input_channels=1, class_type=class_type, pretrain=True)
    
    weight_path = f"/tf/yilian618/ABD_Trauma_detection/weights/{classification_type}/{class_type}/{data_file_name[times]}/{data_acc[times]}.pth"
    model.load_state_dict(torch.load(weight_path))
    model.to(device)

    print(f"times:{times}, file:{data_file_name}, acc:{data_acc}")
    print("Collecting:", datetime.now(), flush=True)

    # Predict
    Inference = InferenceRunner(model = model, device = device, dataloader=test_loader, mode=mode)
    y_pred, images, masks = Inference.run()
    print("Evaluation complete.")

    return y_pred, images, masks, test_df


    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    class_type = args.class_type.lower()
    mode = args.mode
    if args.file.endswith("ini"):
        cfgpath = f"/tf/yilian618/ABD_Trauma_detection/config/{class_type}/{args.file}"
    else:
        cfgpath = (f"/tf/yilian618/ABD_Trauma_detection/config/{class_type}/{args.file}.ini")

    conf = ConfigLoader(cfgpath)
    setting = conf.data_setting
    n_classes = setting.n_classes
    result_info = conf.read_output_section()

    file_name = cfgpath.split("/")[-1][:-4]
    file_name = f"{file_name}_test"


    classification_type = "Multilabel" if class_type == "multiple" else "Binary" if n_classes <= 2 else "Multiclass"

    # Load DataFrame
    df_all_rsna, test_data = load_rsna_data(
        rsna_path="/tf/yilian618/rsna_train_new_v2.csv",
        noseg_path="/tf/yilian618/nosegmentation.csv",
        test_path="/tf/jacky831006/ABD_classification/rsna_test_20240531.csv",
        seed=setting.seed,
        neg_sample=800,
    )

    df_all_cgmh = load_cgmh_data(
        path="/tf/yilian618/ABD_classification/ABD_venous_all_20230709_for_label_new.csv",
        rm_list=[21410269, 3687455, 21816625, 21410022, 39010142, 20430081]
    )

        # best model output
    if args.select is not False:
        data_file_name = [result_info["data_file_name"][[args.select]]]
        data_acc = [result_info["best_accuracy"][args.select]]
    else:
        data_file_name = result_info["data_file_name"]
        data_acc = result_info["best_accuracy"]

    # Visualization & Evaluation Placeholder
    for k in range(len(data_file_name)):

        if len(data_file_name) == 1:
            dir_path = f"/tf/yilian618/classification_torch/{class_type}/{file_name}/"
        else:
            dir_path = f"/tf/yilian618/classification_torch/{class_type}/{file_name}/{k}"

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        y_pred, images, masks, test_df = run_evaluation(times=k)
        y_label_data, y_pre_data, organ_list = prepare_labels(test_df, class_type, y_pred)

        for i in range(len(organ_list)):
            if len(organ_list) > 1:
                y_pre, y_label, organ = y_pre_data[:, i, :], y_label_data[i], organ_list[i]
            else:
                y_pre, y_label, organ = y_pred, y_label_data, organ_list[i]

            if n_classes <=2:
                optimal_th = plot_roc(y_pre, y_label, organ, dir_path, f"{file_name}_{k}_{organ}")
                print(f"{organ} cutoff value:{optimal_th}")

                pos_list = []
                neg_list = []
                for y_label_value, y_pre_value in zip(y_label, y_pre):
                    if y_label_value == 1:
                        pos_list.append(y_pre_value[1])
                    else:
                        neg_list.append(y_pre_value[1])
                plot_dis(pos_list, neg_list, dir_path, f"{file_name}_{k}_{organ}") 

                y_pre_n = list(np.where(y_pre[..., 1]> optimal_th,1,0))
                y_list = list(y_label)

                test_df[f"pre_label_{organ}"] = np.array(y_pre_n)
                test_df[f"ori_pre_{organ}"] = np.array(y_pre[..., 1])
                test_df = test_df[test_df.columns.drop(list(test_df.filter(regex="Unnamed")))]
                test_df_path = f"{dir_path}/{file_name}_{k}.csv"
                test_df.to_csv(test_df_path, index=False)

                perd = test_df[f"ori_pre_{organ}"].values
                perd_label = list(np.where(perd>0.5,1,0))
                result = confusion_matrix(y_label, perd_label)
                (tn, fp, fn, tp) = confusion_matrix(y_list, perd_label).ravel()
                plot_confusion_matrix(result, classes=[0, 1], title=f"Confusion matrix_{organ}")
                plt.savefig(f"{dir_path}/{file_name}_{k}_{organ}.png")
                plt.close()

                ACC, PPV, NPV, Sensitivity, Specificity = confusion_matrix_CI(tn, fp, fn, tp)
                print(f"{organ} Test Accuracy: {ACC}")
                print("PPV:", PPV, "NPV:", NPV, "Sensitivity:", Sensitivity, "Specificity:", Specificity)

                result = confusion_matrix(y_label, y_pre_n)
                (tn, fp, fn, tp) = confusion_matrix(y_list, y_pre_n).ravel()
                plot_confusion_matrix(result, classes=[0, 1], title=f"Confusion matrix_{organ}")
                plt.savefig(f"{dir_path}/{file_name}_{k}_{organ}_Modifed.png")
                plt.close()
                ACC, PPV, NPV, Sensitivity, Specificity = confusion_matrix_CI(tn, fp, fn, tp)
                print(f"Modifed : {organ} Test Accuracy: {ACC}")
                print("PPV:", PPV, "NPV:", NPV, "Sensitivity:", Sensitivity, "Specificity:", Specificity)

            
            else:
                y_pre_n = np.argmax(y_pre, axis=1)
                test_df[f"pre_label_{organ}"] = np.array(y_pre_n)
                test_df[f"ori_pre_health_{organ}"] = np.array(y_pre[...,0])
                test_df[f"ori_pre_low_{organ}"] = np.array(y_pre[...,1])
                test_df[f"ori_pre_high_{organ}"] = np.array(y_pre[...,2])
                test_df = test_df[test_df.columns.drop(list(test_df.filter(regex="Unnamed")))]
                test_df_path = f"{dir_path}/{file_name}_{k}.csv"
                test_df.to_csv(test_df_path, index=False)
                
                
                optimal_th = plot_multi_class_roc(test_df[[f"ori_pre_health_{organ}", f"ori_pre_low_{organ}", f"ori_pre_high_{organ}"]].values, y_label, n_classes, organ, dir_path, f"{file_name}_{k}_{organ}")
                print(f"{organ}: cutoff value:{optimal_th}")

                y_list = list(y_label)
                result = confusion_matrix(y_list, y_pre_n)
                plot_confusion_matrix_multi(result, classes=[0, 1,2], title=organ)
                plt.savefig(f"{dir_path}/{file_name}_{k}_{organ}.png")
                plt.close()
                metrics = confusion_matrix_CI_multi(result)
                print(metrics)

                print("Modifed Test Accuracy: ")

                adj_testing = np.where(
                    test_df[f"ori_pre_high_{organ}"].values > optimal_th[2], 2,
                    np.where(test_df[f"ori_pre_low_{organ}"].values > optimal_th[1], 1, 0)
                )
                
                result = confusion_matrix(y_list, adj_testing)
                plot_confusion_matrix_multi(result, classes=[0, 1,2], title=organ)
                plt.savefig(f"{dir_path}/{file_name}_{k}_{organ}_Modifed.png")
                plt.close()
                metrics = confusion_matrix_CI_multi(result)
                print(metrics)
    
    generate_segmentation_heatmaps(
        test_df=test_df,
        image_list=images,
        mask_list=masks,
        class_type=class_type,
        dir_path=dir_path,
        model_type=mode
    )
            
    print(f"testing is finish!")

    

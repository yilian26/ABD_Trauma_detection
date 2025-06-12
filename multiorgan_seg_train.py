import os
import gc
import time
import torch
import argparse
import functools
import numpy as np
import pandas as pd
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from models import prepare_model
from utils.loss import LossBuilder
from utils.visualizer import plot_loss_metric
from utils.data_utils import str2mode, str2dataset
from model_engine.trainer_and_evaluator import Trainer, Evaluator
from config.config_loader import ConfigLoader, TrainingProgressTracker
from data.dataset_builder import data_split, load_cgmh_data, load_rsna_data
from data.transforms import get_train_transforms,  get_valid_transforms
from data.data_loaders import train_loaders,  valid_loaders, test_loaders

# let all of print can be flush = ture
print = functools.partial(print, flush=True)

def get_parser():
    parser = argparse.ArgumentParser(description="ABD Trauma detection")
    parser.add_argument("-f", "--file", help=" The config file name. ", type=str)
    parser.add_argument(
        "-c",
        "--class_type",
        help=" The class of data. (liver, kidney, spleen, all) ",
        type=str,
    parser.add_argument("-m", "--mode", type=str2mode, default="segmentation", help="Model mode: 0/cls=classification, 1/seg=segmentation")
    parser.add_argument("-d", "--dataset", type=str2dataset, default="cgmh", help="Dataset source: 0/cgmh, 1/rsna, 2/multiple")
    )
    return parser



def run_once(times=0):
    # reset config parameter
    tracker = TrainingProgressTracker()
    setting = conf.data_setting



    train_df_rsna, valid_df_rsna, test_df_rsna = data_split(
        df_all_rsna, test_data=test_data, test_fix=None, source="RSNA", ratio=setting.data_split_ratio, seed=setting.seed
    )
    train_df_cgmh, valid_df_cgmh, test_df_cgmh = data_split(
        df_all_cgmh,test_data=None, test_fix=2016, source="CGMH", ratio=setting.data_split_ratio, seed=setting.seed
    )

    # train_df_rsna = train_df_rsna[:20]
    # valid_df_rsna = valid_df_rsna[:10]
    # test_df_rsna = test_df_rsna[:10]

    # train_df_cgmh = train_df_cgmh[:20]
    # valid_df_cgmh = valid_df_cgmh[:10]
    # test_df_cgmh = test_df_cgmh[:10]

    train_transforms = get_train_transforms(class_type=class_type, cfg = conf)
    valid_transforms = get_valid_transforms(class_type=class_type, cfg = conf)
    
    train_loader = train_loaders(conf, train_df_cgmh, train_df_rsna, data_source="all", class_type=class_type, train_transforms=train_transforms)
    valid_loader = valid_loaders(conf, valid_df_cgmh, valid_df_rsna, data_source="all", class_type=class_type, valid_transforms=valid_transforms)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0")

    # Model setting
    model = prepare_model(conf, device, resnet_depth=50, n_input_channels=1, class_type=class_type, pretrain=True)
    loss_function = LossBuilder(conf, df_cgmh = df_all_cgmh, df_rsna = df_all_rsna, class_type=class_type, device=device).get_loss_function()


    optimizer = torch.optim.Adam(model.parameters(), setting.init_lr[times])

    patience = epochs if setting.lr_decay_epoch == 0 else setting.lr_decay_epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=setting.lr_decay_rate, patience=patience, verbose=True
    )
    
    if len(setting.init_lr) == 1:
        print(f"\n Processing begining", flush=True)
    else:
        print(f"\n Processing grid search #{times}, learning rate:{setting.init_lr[times]}", flush=True)

    data_num = len(train_df_cgmh+train_df_rsna)
    # test_model = train(model, device, data_num, epochs, optimizer, loss_function, train_loader, \
    #                    val_loader, early_stop, init_lr, lr_decay_rate, lr_decay_epoch, check_path)
    trainer = Trainer(
        conf=conf,
        model=model,
        loss_function=loss_function,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_path=check_path,
        mode="segmentation",  
    )

    trainer.checkpoint_path = check_path
    trainer.tracker = tracker
    trainer.run()

    # plot train loss and metric
    plot_loss_metric(tracker.epoch_loss_values, tracker.metric_values, check_path)
    # remove dataloader to free memory
    del train_loader, valid_loader
    gc.collect()

    # Avoid ram out of memory
    test_loader =  test_loaders(conf, test_df_cgmh, test_df_rsna, data_source="all", class_type=class_type, test_transforms=valid_transforms)
    # validation is same as testing
    print(f"Best accuracy:{tracker.best_metric}")
    
    load_weight = f"{check_path}/{tracker.best_metric}.pth"
    model.load_state_dict(torch.load(load_weight))

    # record paramter
    accuracy_list.append(tracker.best_metric)
    file_list.append(now)
    epoch_list.append(tracker.best_metric_epoch)

    evaluator = Evaluator(device=device, tracker=tracker)
    test_acc = evaluator.evaluate_segmentation(model, test_loader)
    
    test_accuracy_list.append(test_acc)
    del test_loader
    gc.collect()

    print(f"\n Best accuracy:{tracker.best_metric}, Best test accuracy:{test_acc}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    class_type = args.class_type
    # 讀檔路徑，之後可自己微調
    if args.file.endswith("ini"):
        cfgpath = f"/tf/yilian618/ABD_Trauma_detection/config/{class_type}/{args.file}"
    else:
        cfgpath = (
            f"/tf/yilian618/ABD_Trauma_detection/config/{class_type}/{args.file}.ini"
        )
        
    conf = ConfigLoader(cfgpath)
    setting = conf.data_setting
    classification_type = "Multilabel" if class_type == "multiple" else "Binary" if setting.n_classes <= 2 else "Multiclass"

    # Data progressing
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

    
    # Training by cross validation
    accuracy_list = []
    test_accuracy_list = []
    file_list = []
    epoch_list = []

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    check_path = (f"/tf/yilian618/ABD_Trauma_detection/weights/{classification_type}/{class_type}/{now}")
    os.makedirs(check_path, exist_ok=True)
    print(f"\n Weight location:{check_path}", flush=True)


    first_start_time = time.time()

    for k in range(len(setting.init_lr)):
        run_once(k)


    final_end_time = time.time()
    hours, rem = divmod(final_end_time - first_start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    all_time = "All time:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds
    )
    print(all_time)

    # write some output information in ori ini
    conf.write_output_section(
        all_time = all_time,
        file_list = str(file_list),
        accuracy_list = str(accuracy_list),
        test_accuracy_list = str(test_accuracy_list),
        epoch_list = str(epoch_list),
    )


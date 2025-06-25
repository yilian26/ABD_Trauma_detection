import os
import torch
import pandas as pd
import numpy as np
import argparse
import gc
import math
import functools
from models import prepare_model
from config.config_loader import ConfigLoader
from utils.data_utils import str2num_or_false
from data.transforms import get_valid_transforms
from utils.cam_utils import CAM_plot
from monai.data import CacheDataset, DataLoader

print = functools.partial(print, flush=True)

def get_parser():
    parser = argparse.ArgumentParser(description="Run Grad-CAM Subprocess")
    parser.add_argument("-c", "--class_type", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--dir_path", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--first_idx", type=int, required=True)
    parser.add_argument("--split_num", type=int, required=True)
    parser.add_argument("--cfgpath", type=str, required=True)
    parser.add_argument("--label", action="store_const", const=True, default=False)
    parser.add_argument("--heatmap_type", type=str, default="all")
    return parser

def data_progress_all_in_split(file, class_type, n_classes):
    dicts = []
    for index, row in file.iterrows():
        source = row['source']
        if source == "rsna":
            dirs = "/tf/TotalSegmentator/rsna_select_z_whole_image"
            mask_dirs = "/tf/SharedFolder/TotalSegmentator/gaussian_mask"
            output = str(row['file_id'].replace(".0", ""))
        else:
            dirs = "/tf/TotalSegmentator/ABD_select_z_whole_image"
            mask_dirs = "/tf/SharedFolder/TotalSegmentator/gaussian_mask_cgmh"
            output = str(row['file_id'])
        image = os.path.join(dirs, output) + ".nii.gz"

        if class_type == "spleen":
            mask = os.path.join(mask_dirs, "spl", output) + ".nii.gz"
            label = np.argmax([int(row["spleen_healthy"]), int(row["spleen_low"]), int(row["spleen_high"])]) if n_classes > 2 else int(np.where(row["spleen_healthy"]==1,0,1))
            dicts.append({"image": image, "label": label, "mask": mask})
        elif class_type == "kidney":
            mask = os.path.join(mask_dirs, "kid_all", output) + ".nii.gz"
            label = int(np.where(row["kidney_healthy"]==1, 0, 1))
            dicts.append({"image": image, "label": label, "mask": mask})
        elif class_type == "liver":
            mask = os.path.join(mask_dirs, "liv", output) + ".nii.gz"
            label = int(np.where(row["liver_healthy"]==1, 0, 1))
            dicts.append({"image": image, "label": label, "mask": mask})
        elif class_type == "all":
            mask = os.path.join(mask_dirs, "any_inj", output) + ".nii.gz"
            label = 0 if (row["liver_healthy"] == 1 and row["spleen_healthy"] == 1 and row["kidney_healthy"] == 1) else 1
            dicts.append({"image": image, "label": label, "mask": mask})
        elif class_type == "multiple":
            mask_liv = os.path.join(mask_dirs, "liv", output) + ".nii.gz"
            mask_spl = os.path.join(mask_dirs, "spl", output) + ".nii.gz"
            mask_kid = os.path.join(mask_dirs, "kid_all", output) + ".nii.gz"
            mask_kid_health = os.path.join(mask_dirs, "kid_health", f"{output}.nii.gz")
            label = np.stack([
                int(row["liver_label"]),
                int(row["spleen_label"]),
                int(row["kidney_label"])
            ])
            sample = {
                "image": image,
                "label": label,
                "mask_liv": mask_liv,
                "mask_spl": mask_spl,
                "mask_kid": mask_kid
            }
            if row['source'] == "cgmh" and os.path.exists(mask_kid_health):
                sample["mask_kid_health"] = mask_kid_health
            dicts.append(sample)
    return dicts

if __name__ == "__main__":
    args = get_parser().parse_args()
    class_type = args.class_type
    Label = args.label

    conf = ConfigLoader(args.cfgpath)
    setting = conf.data_setting
    n_classes = setting.n_classes

    test_df = pd.read_csv(args.file_path)
    test_df = test_df.reset_index(drop=True)

    test_data_dicts = data_progress_all_in_split(test_df, class_type, n_classes)
    test_data_dicts_sel = test_data_dicts[args.first_idx : args.split_num]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prepare_model(conf, device, resnet_depth=50, n_input_channels=1, class_type=class_type, pretrain=True)
    model.load_state_dict(torch.load(args.weight_path))

    test_transforms = get_valid_transforms(class_type=class_type, cfg=conf)
    test_ds = CacheDataset(
        data=test_data_dicts_sel,
        transform=test_transforms,
        cache_rate=1,
        num_workers=1
    )
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

    CAM_plot(conf, model, test_loader, test_df, class_type, device, args.first_idx, args.heatmap_type, Label, args.dir_path)

    model.to("cpu")
    del model, test_loader, test_ds, test_transforms
    torch.cuda.empty_cache()
    gc.collect()

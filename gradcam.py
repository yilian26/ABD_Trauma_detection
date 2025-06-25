# gradcam_main.py
import argparse
import subprocess
import math
import pandas as pd
import os
from config.config_loader import ConfigLoader
from utils.data_utils import str2num_or_false

def get_parser():
    parser = argparse.ArgumentParser(description="Main Grad-CAM Controller")
    parser.add_argument("-c", "--class_type", type=str, required=True)
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-s", "--select", type=str2num_or_false, default=False)
    parser.add_argument("-l", "--label", action="store_const", const=True, default=False)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    class_type = args.class_type
    label = str(args.label)
    split_num = 6

    cfgpath = f"/tf/yilian618/ABD_Trauma_detection/config/{class_type}/{args.file}" if args.file.endswith("ini") \
              else f"/tf/yilian618/ABD_Trauma_detection/config/{class_type}/{args.file}.ini"

    conf = ConfigLoader(cfgpath)
    setting = conf.data_setting
    n_classes = setting.n_classes
    result_info = conf.read_output_section()

    if args.select is not False:
        data_file_name = [result_info["data_file_name"][args.select]]
        data_acc = [result_info["best_accuracy"][args.select]]
    else:
        data_file_name = result_info["data_file_name"]
        data_acc = result_info["best_accuracy"]

    file_name = os.path.basename(cfgpath)[:-4] + "_test"
    if len(data_file_name) == 1:
        dir_path = f"/tf/yilian618/ABD_Trauma_detection/result/grad_cam_image/{class_type}/{file_name}"
        data_path = f"/tf/yilian618/ABD_Trauma_detection/result/{class_type}/{file_name}/{file_name}.csv"
    else:
        dir_path = f"/tf/yilian618/ABD_Trauma_detection/result/grad_cam_image/{class_type}/{file_name}/{k}"
        data_path = f"/tf/yilian618/ABD_Trauma_detection/result/{class_type}/{file_name}/{k}/{file_name}.csv"
    
    for k in range(len(data_file_name)):
        weight_path = f"/tf/yilian618/ABD_Trauma_detection/weights/"
        classification_type = "Multilabel" if class_type == "multiple" else "Binary" if n_classes <= 2 else "Multiclass"
        weight_path += f"{classification_type}/{class_type}/{data_file_name[k]}/{data_acc[k]}.pth"

        test_df = pd.read_csv(data_path)
        test_df_pos = test_df[test_df["inj_solid"] == 1]
        test_df_fp = test_df[(test_df["inj_solid"] == 0) & (test_df["ori_pre_Inj_sold"] > 0.5)]
        test_df = pd.concat([test_df_pos, test_df_fp]).reset_index(drop=True)

        for split_idx in range(math.ceil(len(test_df) / split_num)):
            first_idx = split_idx * split_num
            last_idx = min((split_idx + 1) * split_num, len(test_df))
            print(f"\n[Main] Running Grad-CAM Split {split_idx+1}: {first_idx} to {last_idx - 1}")

            cmd = [
                "python3", "/tf/yilian618/ABD_Trauma_detection/gradcam_subprocess.py",
                "--class_type", class_type,
                "--file_path", data_path,
                "--dir_path", dir_path,
                "--weight_path", weight_path,
                "--first_idx", str(first_idx),
                "--split_num", str(last_idx),
                "--cfgpath", cfgpath,
                "--heatmap_type", "all",     
            ]
            if args.label:
                cmd.append("--label")
            cmd = [x for x in cmd if x != ""]  # 清除空字串
            subprocess.run(cmd)

    print("\n[Main] All Grad-CAM splits finished.")

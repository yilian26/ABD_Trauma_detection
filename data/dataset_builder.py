import os
import numpy as np
import pandas as pd
from utils.data_utils import convert_date


def CGMH_inj_check(row):
    """
    Convert injury labels into binary form: 0 = healthy, 1 = injury.

    Returns:
        pd.Series: [kidney_injury, liver_injury, spleen_injury].
    """
    kid_inj_tmp = np.where((row["kid_inj_rt"] == 0) & (row["kid_inj_lt"] == 0), 0, 1)
    liv_inj_tmp = np.where(row["liv_inj"] == 0, 0, 1)
    spl_inj_tmp = np.where(row["spl_inj"] == 0, 0, 1)
    return pd.Series([kid_inj_tmp, liv_inj_tmp, spl_inj_tmp])


def load_rsna_data(rsna_path, noseg_path, test_path, seed, neg_sample=800):
    all_data = pd.read_csv(rsna_path)
    
    # 篩除無 segmentation 及某一筆特定異常影像
    if noseg_path:
        no_seg = pd.read_csv(noseg_path)
        all_data = all_data[~all_data['file_paths'].isin(no_seg['file_paths'])]
    
    all_data = all_data[~all_data['file_paths'].isin([
        "/SSD/rsna-2023/train_images_new/63501/SDY00001/1.2.123.12345.1.2.3.63501.7194.nii.gz"
    ])]

    if test_path:
        test_data = pd.read_csv(test_path)
        df_all = all_data[~all_data['file_paths'].isin(test_data['file_paths'])]

    # downsample negative cases
    pos_data = df_all[df_all['any_injury'] == 1]
    if neg_sample:
        neg_data = df_all[df_all['any_injury'] == 0].sample(n=neg_sample, random_state=seed)
    else:
        neg_data = df_all[df_all['any_injury'] == 0]

    return pd.concat([pos_data, neg_data]), test_data


def load_cgmh_data(path, rm_list):
    df = pd.read_csv(path)
    df = df[~df.chartNo.isin(rm_list)]
    df[["kid_inj_lt", "kid_inj_rt", "liv_inj", "spl_inj"]] = df[["kid_inj_lt", "kid_inj_rt", "liv_inj", "spl_inj"]].fillna(0)
    df = df.dropna(subset=["label"])
    df = df[df.label != "exclude"]

    # 處理日期
    df["TRDx_ER_arrival_time_tmp"] = pd.to_datetime(df["TRDx_ER_arrival_time"], errors='coerce').dt.strftime("%Y%m%d")
    df["examdate_tmp"] = df["examdate"].apply(convert_date)
    df["examdate_tmp"] = df["examdate_tmp"].fillna(df["TRDx_ER_arrival_time_tmp"])
    df["year"] = pd.to_datetime(df["examdate_tmp"], errors='coerce').dt.year

    return df


def load_dataset(dataset_source: str, seed: int, neg_sample: int,
                 rsna_path: str, noseg_path: str, test_path: str,
                 cgmh_path: str, rm_list: list):
    """
    Load CGMH and RSNA datasets based on the selected data source.

    Args:
        dataset_source (str): "cgmh", "rsna", or "multiple"
        seed (int): random seed for sampling
        neg_sample (int): number of negative samples to keep for RSNA
        rsna_path (str): path to RSNA dataset CSV
        noseg_path (str): path to RSNA no-segmentation exclusion list
        test_path (str): path to RSNA test file
        cgmh_path (str): path to CGMH dataset CSV
        rm_list (list): chartNo to remove from CGMH

    Returns:
        Tuple of (df_cgmh, df_rsna, test_data), unused df will be None
    """
    df_cgmh, df_rsna, test_data = None, None, None

    if dataset_source.lower() in ["rsna", "multi", "2", "1"]:
        df_rsna, test_data = load_rsna_data(
            rsna_path=rsna_path,
            noseg_path=noseg_path,
            test_path=test_path,
            seed=seed,
            neg_sample=neg_sample,
        )

    if dataset_source.lower() in ["cgmh", "multi", "2", "0"]:
        df_cgmh = load_cgmh_data(
            path=cgmh_path,
            rm_list=rm_list,
        )

    return df_cgmh, df_rsna, test_data




def data_split(df, test_data=None, test_fix=None, source="RSNA", ratio=(0.7, 0.1, 0.2), seed=0):
    """
    Splits the dataset into training, validation, and testing sets.

    Parameters:
        df (pd.DataFrame): Input dataframe to split.
        test_data (pd.DataFrame, optional): Test dataset for RSNA source. Required if source is 'RSNA'.
        test_fix (int, optional): Year used to fix the test set for CGMH source.
        source (str): Source of the data, must be either "RSNA" or "CGMH".
        ratio (tuple): A 3-element tuple indicating the split ratio for (train, valid, test).
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, valid_df, test_df)
    """

    df = df.reset_index()

    if source.lower() == "cgmh":
        df[["kid_inj_tmp", "liv_inj_tmp", "spl_inj_tmp"]] = df.apply(CGMH_inj_check, axis=1)
        # Assign group_key for stratified sampling based on injury patterns
        df["group_key"] = df.apply(lambda r: f"l{r['liv_inj_tmp']}_k{r['kid_inj_tmp']}_s{r['spl_inj_tmp']}", axis=1)

        if test_fix:
            test_df = df[df["year"] == test_fix]
            df = df[df["year"] != test_fix]
        else:
            test_df = None

    elif source.lower() == "rsna":
        # Assign group_key for stratified sampling based on injury patterns
        df["group_key"] = df.apply(
            lambda r: f"{r['liver_low']}_{r['liver_high']}_{r['spleen_low']}_{r['spleen_high']}_"
                      f"{r['kidney_low']}_{r['kidney_high']}_{r['bowel_injury']}_{r['extravasation_injury']}", axis=1
        )
        if test_data is not None:
            test_df = test_data.reset_index()

    else:
        raise ValueError("Unrecognized DataFrame format; should be either RSNA or CGMH")

    # split the dataset
    train_df = df.groupby("group_key", group_keys=False).sample(frac=ratio[0], random_state=seed)
    df_sel = df.drop(train_df.index)

    if test_df is None:  # If test is not predefined, split remaining data into valid and test
        valid_df = df_sel.groupby("group_key", group_keys=False).sample(
            frac=ratio[1] / (ratio[1] + ratio[2]), random_state=seed
        )
        test_df = df_sel.drop(valid_df.index)
    else:
        valid_df = df_sel

    return train_df, valid_df, test_df


def prepare_dataset_dicts(df, dataset_type="rsna", class_type="multiple"):
    dicts = []

    for _, row in df.iterrows():

        if dataset_type.lower() == "rsna":
            file_id = os.path.basename(row["file_paths"])[:-7]
            image_dir = "/tf/TotalSegmentator/rsna_select_z_whole_image"
            mask_dir = "/tf/SharedFolder/TotalSegmentator/gaussian_mask"

        elif dataset_type.lower() == "cgmh":
            image_dir = "/tf/TotalSegmentator/ABD_select_z_whole_image"
            mask_dir = "/tf/SharedFolder/TotalSegmentator/gaussian_mask_cgmh"
            file_id = str(row["chartNo"]) + str(row["examdate"]).replace(".0", "")

        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Expected 'rsna' or 'cgmh'.")

        image = os.path.join(image_dir, f"{file_id}.nii.gz")
        if not os.path.exists(image) and dataset_type.lower() == "cgmh":
            file_id = str(row["chartNo"]) + str(np.NaN)
            image = os.path.join(image_dir, f"{file_id}.nii.gz")


        if class_type == "liver":
            mask = os.path.join(mask_dir, "liv", f"{file_id}.nii.gz")
            label = int(np.where(row["liver_healthy"]==1,0,1)) if "liver_healthy" in row else int(row["liv_inj_tmp"])
            dicts.append({"image": image, "label": label, "mask": mask})

        elif class_type == "spleen":
            mask = os.path.join(mask_dir, "spl", f"{file_id}.nii.gz")
            label = int(np.where(row["spleen_healthy"] == 1, 0, 1)) if "spleen_healthy" in row else int(row["spl_inj_tmp"])
            dicts.append({"image": image, "label": label, "mask": mask})

        elif class_type == "kidney":
            mask = os.path.join(mask_dir, "kid_inj", f"{file_id}.nii.gz")
            label = label = int(np.where(row["kidney_healthy"] == 1, 0, 1)) if "kidney_healthy" in row else int(row["kid_inj_tmp"])
            if os.path.exists(image) and os.path.exists(mask):
                dicts.append({"image": image, "label": label, "mask": mask})

        elif class_type == "all":
            mask = os.path.join(mask_dir, "any_inj", f"{file_id}.nii.gz")
            label = int(row.get("label", "pos") == "pos") if dataset_type == "cgmh" else int(
                not (row["liver_healthy"] == 1 and row["spleen_healthy"] == 1 and row["kidney_healthy"] == 1))
            dicts.append({"image": image, "mask": mask, "label": label})

        elif class_type == "multiple":
            # print(image)
            label_liv = int(np.where(row["liver_healthy"] == 1, 0, 1)) if "liver_healthy" in row else int(row["liv_inj_tmp"])
            label_spl = int(np.where(row["spleen_healthy"] == 1, 0, 1)) if "spleen_healthy" in row else int(row["spl_inj_tmp"])
            label_kid = int(np.where(row["kidney_healthy"] == 1, 0, 1)) if "kidney_healthy" in row else int(row["kid_inj_tmp"])
            label = np.stack([label_liv, label_spl, label_kid], axis=0)

            mask_liv = os.path.join(mask_dir, "liv", f"{file_id}.nii.gz")
            mask_spl = os.path.join(mask_dir, "spl", f"{file_id}.nii.gz")
            if dataset_type.lower() == "cgmh":
                mask_kid = os.path.join(mask_dir, "kid_inj", f"{file_id}.nii.gz")
            else:
                mask_kid = os.path.join(mask_dir, "kid_all", f"{file_id}.nii.gz")
            mask_kid_health = os.path.join(mask_dir, "kid_health", f"{file_id}.nii.gz")

            sample = {
                "image": image,
                "label": label,
                "mask_liv": mask_liv,
                "mask_spl": mask_spl,
                "mask_kid": mask_kid
            }
            if dataset_type == "cgmh" and os.path.exists(mask_kid_health):
                sample["mask_kid_health"] = mask_kid_health
#             else:
#                 sample["mask_kid_health"] = None
            # print(mask_kid)
            if os.path.exists(image) and os.path.exists(mask_kid):
                dicts.append(sample)
            else:
                print(image)
                print(mask_kid)

    return dicts

class DataframeCombiner:
    """
    Combine and standardize CGMH and RSNA DataFrames for downstream training or analysis.
    Supports binary or multi-class labels for kidney, liver, and spleen injuries.
    """

    def __init__(self, df_cgmh=None, df_rsna=None, n_classes=2):
        """
        Args:
            df_cgmh (pd.DataFrame): CGMH dataset.
            df_rsna (pd.DataFrame): RSNA dataset.
            n_classes (int): 2 for binary classification, >2 for multi-class.
        """
        self.df_cgmh = df_cgmh if df_cgmh is not None else pd.DataFrame()
        self.df_rsna = df_rsna if df_rsna is not None else pd.DataFrame()
        self.n_classes = n_classes

    def combine(self):
        """
        Combine both CGMH and RSNA datasets into a single DataFrame with unified label structure.

        Returns:
            pd.DataFrame: Combined dataset with standardized labels.
        """
        all_dfs = []

        if not self.df_cgmh.empty:
            df_cgmh_prepared = self._prepare(self.df_cgmh, "cgmh", self.n_classes)
            all_dfs.append(df_cgmh_prepared)

        if not self.df_rsna.empty:
            df_rsna_prepared = self._prepare(self.df_rsna, "rsna", self.n_classes)
            all_dfs.append(df_rsna_prepared)

        if not all_dfs:
            raise ValueError("Both input DataFrames are empty. Cannot combine.")

        return pd.concat(all_dfs, ignore_index=True)

    def _prepare(self, df, dataset_type="rsna", n_classes=2):
        """
        Convert raw labels to structured organ labels for a specific dataset type.

        Args:
            df (pd.DataFrame): Input DataFrame.
            dataset_type (str): Either 'rsna' or 'cgmh'.
            n_classes (int): Number of classification levels (2 or >2).

        Returns:
            pd.DataFrame: Processed DataFrame with standardized columns.
        """
        records = []

        for _, row in df.iterrows():
            # Determine file ID format based on source
            if dataset_type == "rsna":
                file_id = os.path.basename(row["file_paths"])[:-7]

                if n_classes > 2:
                    kidney_label = np.argmax([int(row["kidney_healthy"]), int(row["kidney_low"]), int(row["kidney_high"])])
                    spleen_label = np.argmax([int(row["spleen_healthy"]), int(row["spleen_low"]), int(row["spleen_high"])])
                    liver_label = np.argmax([int(row["liver_healthy"]), int(row["liver_low"]), int(row["liver_high"])])
                else:
                    kidney_label = 0 if row["kidney_healthy"] == 1 else 1
                    spleen_label = 0 if row["spleen_healthy"] == 1 else 1
                    liver_label = 0 if row["liver_healthy"] == 1 else 1

            elif dataset_type == "cgmh":
                file_id = str(row["chartNo"]) + str(row["examdate"]).replace(".0", "")

                if n_classes > 2:
                    kidney_label = np.where((row["kid_inj_rt"] == 0) & (row["kid_inj_lt"] == 0), 0,
                                            np.where((row["kid_inj_rt"] > 3) | (row["kid_inj_lt"] > 3), 2, 1))
                    liver_label = np.where(row["liv_inj"] == 0, 0, np.where(row["liv_inj"] > 3, 2, 1))
                    spleen_label = np.where(row["spl_inj"] == 0, 0, np.where(row["spl_inj"] > 3, 2, 1))
                else:
                    kidney_label = 0 if (row["kid_inj_rt"] == 0) & (row["kid_inj_lt"] == 0) else 1
                    liver_label = 0 if row["liv_inj"] == 0 else 1
                    spleen_label = 0 if row["spl_inj"] == 0 else 1

            else:
                raise ValueError(f"Unknown dataset_type: {dataset_type}. Expected 'rsna' or 'cgmh'.")

            # inj_solid: set to 1 if any organ is injured
            inj_solid = 0 if (kidney_label == 0 and spleen_label == 0 and liver_label == 0) else 1

            records.append({
                "file_id": file_id,
                "kidney_label": int(kidney_label),
                "spleen_label": int(spleen_label),
                "liver_label": int(liver_label),
                "inj_solid": inj_solid,
                "source": dataset_type
            })

        return pd.DataFrame(records)



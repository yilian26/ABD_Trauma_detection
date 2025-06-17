from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
from .dataset_builder import prepare_dataset_dicts
from typing import List, Tuple, Optional
from config.config_loader import ConfigLoader


def concat_data(df_cgmh, df_rsna, class_type: str):
    """
    Combine CGMH and RSNA DataFrames into a unified list of dicts for MONAI pipeline.

    Args:
        df_cgmh (DataFrame or None): CGMH data.
        df_rsna (DataFrame or None): RSNA data.
        class_type (str): Target organ class type.

    Returns:
        List[dict]: Combined dataset dictionaries.
    """
    data = []

    if df_cgmh is not None:
        data += prepare_dataset_dicts(df_cgmh, "cgmh", class_type)

    if df_rsna is not None:
        data += prepare_dataset_dicts(df_rsna, "rsna", class_type)

    if not data:
        raise ValueError("At least one of df_cgmh or df_rsna must be provided.")

    return data


def train_loaders(
    cfg: ConfigLoader,
    train_df_cgmh,
    train_df_rsna,
    class_type: str = "multiple",
    train_transforms=None,
):
    """
    Initialize the training DataLoader for MONAI segmentation pipeline.

    This function supports single-source training (e.g., CGMH or RSNA) or combined sources.
    If `train_df_extra` is provided, it will be merged with `train_df`.
    Data source must be one of: "cgmh", "rsna", or "all".

    Args:
        cfg (ConfigLoader): Configuration object containing data and training settings.
        train_df (DataFrame): Primary training dataframe (required).
        train_df_extra (DataFrame, optional): Additional training data (e.g., RSNA if CGMH is primary).
        data_source (str): Specifies source of the dataset. Options: "cgmh", "rsna", or "all".
        class_type (str): One of "liver", "spleen", "kidney", "all", "multiple".
        train_transforms (Optional[Transform]): MONAI Compose transform to apply.

    Returns:
        DataLoader: A PyTorch DataLoader containing the prepared CacheDataset for training.
    """
    # Load training-related parameters from config
    setting = cfg.data_setting
    seed = setting.seed
    num_workers = setting.dataloader_num_workers
    batch_size = setting.traning_batch_size
    cache_rate = setting.traning_cache_rate

    # Ensure reproducibility
    set_determinism(seed=seed)

    # Build training dataset dictionary
    train_data_dicts = concat_data(train_df_cgmh, train_df_rsna, class_type)

    # Create MONAI CacheDataset for efficient data loading
    train_ds = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    # Wrap dataset with PyTorch DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader

def valid_loaders(
    cfg: ConfigLoader,
    valid_df_cgmh,
    valid_df_rsna: Optional[object] = None,
    class_type: str = "multiple",
    valid_transforms=None,
):
    """
    Initialize validation DataLoader for MONAI segmentation pipeline.
    Supports CGMH, RSNA, or combined sources.
    """
    setting = cfg.data_setting
    seed = setting.seed
    num_workers = setting.dataloader_num_workers
    batch_size = setting.valid_batch_size
    cache_rate = setting.valid_cache_rate

    set_determinism(seed=seed)

    valid_data_dicts = concat_data(valid_df_cgmh, valid_df_rsna, class_type)

    valid_ds = CacheDataset(
        data=valid_data_dicts,
        transform=valid_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return valid_loader


def test_loaders(
    cfg: ConfigLoader,
    test_df_cgmh,
    test_df_rsna,
    class_type: str = "multiple",
    test_transforms=None,
):
    """
    Initialize test DataLoader for MONAI segmentation pipeline.
    Supports CGMH, RSNA, or combined sources.
    """
    setting = cfg.data_setting
    seed = setting.seed
    num_workers = setting.dataloader_num_workers
    batch_size = setting.testing_batch_size
    cache_rate = setting.test_cache_rate

    set_determinism(seed=seed)

    test_data_dicts = concat_data(test_df_cgmh, test_df_rsna, class_type)

    test_ds = CacheDataset(
        data=test_data_dicts,
        transform=test_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_loader
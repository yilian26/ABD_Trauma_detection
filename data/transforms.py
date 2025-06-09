import torch
import numpy as np
from monai.transforms.transform import MapTransform
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    ScaleIntensityRanged, 
    Spacingd,
    Orientationd, 
    CropForegroundd, 
    Resized, 
    Rand3DElasticd, 
    EnsureTyped
)

class Flexible6ChannelMaskBuilderd(MapTransform):
    """
    根據提供的鍵動態合併為 6 通道 mask。
    第一種格式：mask_liv, mask_spl, mask_kid（各2通道）
    第二種格式：mask_liv, mask_spl（各2通道）+ mask_kid_health, mask_kid（各1通道）
    最終輸出：6 channel mask，用於 one-hot 相對應的類別標籤
    """
    def __init__(self, output_key="multi_mask", labels_key="label", allow_missing_keys=True):
        keys = ["mask_liv", "mask_spl", "mask_kid", "mask_kid_health"]
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        self.labels_key = labels_key

    def __call__(self, data):
        def to_np(mask):
            return mask.numpy().squeeze() if isinstance(mask, torch.Tensor) else mask.squeeze()

        labels = data[self.labels_key]
        mask_liv = to_np(data.get("mask_liv", np.zeros_like(data["image"][0])))
        mask_spl = to_np(data.get("mask_spl", np.zeros_like(data["image"][0])))

        if "mask_kid_health" in data and "mask_kid" in data and data["mask_kid_health"] is not None:
            mask_kid_health = to_np(data["mask_kid_health"])
            mask_kid = to_np(data["mask_kid"])
            channels = [
                mask_liv * (labels[0] == 0),
                mask_liv * (labels[0] == 1),
                mask_spl * (labels[1] == 0),
                mask_spl * (labels[1] == 1),
                mask_kid_health,
                mask_kid
            ]
        else:
            mask_kid = to_np(data.get("mask_kid", np.zeros_like(mask_liv)))
            channels = []
            for i, mask in enumerate([mask_liv, mask_spl, mask_kid]):
                channels.append(mask * (labels[i] == 0))
                channels.append(mask * (labels[i] == 1))

        concatenated = np.stack(channels, axis=0).astype(np.uint8)
        data[self.output_key] = concatenated

        for key in ["mask_liv", "mask_spl", "mask_kid", "mask_kid_health"]:
            data.pop(key, None)
            data.pop(f"{key}_meta_dict", None)

        return data


class ConvertMultiChannelToLabelMapd(MapTransform):
    """
    Convert a multi-channel mask (one-hot or probabilistic) to a single-channel label map.
    The label at each voxel is the channel index with the highest value.

    Args:
        keys (list[str]): Keys in data dict to process (usually one item).
        output_key (str): Key to store the resulting label map.
        allow_missing_keys (bool): Ignore keys not present in input data.
    """
    def __init__(self, keys, output_key="mask", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def __call__(self, data):
        mask = data[self.keys[0]]

        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        if mask.ndim < 2:
            raise ValueError(f"Expected at least 2D mask with channels, got shape {mask.shape}")

        # Use argmax to convert one-hot to label map
        label_map = np.argmax(mask, axis=0).astype(np.uint8)
        data[self.output_key] = label_map

        return data


class ConcatMasksd(MapTransform):
    """
    Custom transform: Concatenates multiple mask arrays along the channel dimension.
    - Supports both 3D and 4D masks by expanding 3D to 4D.
    - Ensures all masks have matching spatial dimensions.
    
    Args:
        keys (list[str]): List of mask keys to concatenate.
        output_key (str): Key to store the concatenated multi-channel mask.
        allow_missing_keys (bool): Ignore missing keys.
    """
    def __init__(self, keys, output_key="multi_mask", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def __call__(self, data):
        def to_np(x):
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        masks = [to_np(data[key]) for key in self.keys]
        masks = [m if m.ndim == 4 else np.expand_dims(m, axis=0) for m in masks]

        # Check spatial dimensions match
        ref_shape = masks[0].shape[1:]
        if not all(m.shape[1:] == ref_shape for m in masks):
            raise ValueError(f"All masks must have the same spatial shape. Got {[m.shape for m in masks]}")

        # Concatenate along channel dimension (axis=0)
        concatenated = np.concatenate(masks, axis=0)
        data[self.output_key] = concatenated.astype(np.uint8)

        # Remove original keys
        for key in self.keys:
            data.pop(key, None)

        return data


def get_train_transforms(class_type: str, cfg) -> Compose:
    aug = cfg.augmentation
    rand3d = cfg.rand3d

    if class_type == "multiple":
        return Compose([
            LoadImaged(keys=["image", "mask_liv", "mask_spl", "mask_kid", "mask_kid_health"], allow_missing_keys=True),
            EnsureChannelFirstd(keys=["image", "mask_liv", "mask_spl", "mask_kid", "mask_kid_health"], allow_missing_keys=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-50,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Spacingd(
                keys=["image", "mask_liv", "mask_spl", "mask_kid","mask_kid_health"], pixdim=(1.5, 1.5, 2.0), mode="bilinear", allow_missing_keys=True
            ),
            Orientationd(keys=["image", "mask_liv", "mask_spl", "mask_kid","mask_kid_health"], axcodes="RAS", allow_missing_keys=True),
            CropForegroundd(keys=["image", "mask_liv", "mask_spl", "mask_kid","mask_kid_health"], source_key="image", allow_missing_keys=True),
            Resized(keys=["image", "mask_liv", "mask_spl", "mask_kid", "mask_kid_health"], spatial_size=aug.size, mode="trilinear", allow_missing_keys=True),
            Rand3DElasticd(
                keys=["image", "mask_liv", "mask_spl", "mask_kid", "mask_kid_health"],
                mode="bilinear",
                prob=rand3d.prob,
                sigma_range=rand3d.sigma_range,
                magnitude_range=rand3d.magnitude_range,
                spatial_size=aug.size,
                translate_range=rand3d.translate_range,
                rotate_range=rand3d.rotate_range,
                scale_range=rand3d.scale_range,
                padding_mode="border",
                allow_missing_keys=True,
            ),
            Flexible6ChannelMaskBuilderd(output_key="mask", labels_key="label"),
            EnsureTyped(keys=["image", "mask"]),
        ])
    else:
        return Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            ScaleIntensityRanged(keys=["image"], a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            Resized(keys=["image", "mask"], spatial_size=aug.size, mode="trilinear"),
            Rand3DElasticd(
                keys=["image", "mask"],
                mode="bilinear",
                prob=rand3d.prob,
                sigma_range=rand3d.sigma_range,
                magnitude_range=rand3d.magnitude_range,
                spatial_size=aug.size,
                translate_range=rand3d.translate_range,
                rotate_range=rand3d.rotate_range,
                scale_range=rand3d.scale_range,
                padding_mode="border",
            ),
            EnsureTyped(keys=["image", "mask"])
        ])

def get_valid_transforms(class_type: str, cfg) -> Compose:
    aug = cfg.augmentation
    if class_type == "multiple":
        return Compose([
            LoadImaged(keys=["image", "mask_liv", "mask_spl", "mask_kid", "mask_kid_health"], allow_missing_keys=True),
            EnsureChannelFirstd(keys=["image", "mask_liv", "mask_spl", "mask_kid", "mask_kid_health"], allow_missing_keys=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-50,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Spacingd(
                keys=["image", "mask_liv", "mask_spl", "mask_kid", "mask_kid_health"], pixdim=(1.5, 1.5, 2.0), mode="bilinear", allow_missing_keys=True
            ),
            Orientationd(keys=["image", "mask_liv", "mask_spl", "mask_kid", "mask_kid_health"], axcodes="RAS", allow_missing_keys=True),
            CropForegroundd(keys=["image", "mask_liv", "mask_spl", "mask_kid","mask_kid_health"], source_key="image", allow_missing_keys=True),
            Resized(keys=["image", "mask_liv", "mask_spl", "mask_kid", "mask_kid_health"], spatial_size=aug.size, mode="trilinear", allow_missing_keys=True),
            Flexible6ChannelMaskBuilderd(output_key="mask", labels_key="label"),
            EnsureTyped(keys=["image", "mask"]),
        ])
    else:
        return Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            ScaleIntensityRanged(keys=["image"], a_min=-50, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            Resized(keys=["image", "mask"], spatial_size=aug.size, mode="trilinear")
        ])



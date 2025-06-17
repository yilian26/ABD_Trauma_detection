import torch
import torch.nn as nn
from config.config_loader import ConfigLoader

from .UNet.ResUNet import generate_model as generate_unet_model
from .ResNet.resnet_3d import generate_model as generate_resnet3d_model
from .UNet.ResUNet import modify_first_conv_layer


def prepare_model(
    cfg: ConfigLoader,
    device: torch.device,
    resnet_depth: int = 50,
    n_input_channels: int = 1,
    class_type: str = "multiple",
    pretrain: bool = True,
) -> nn.Module:
    """
    Build model based on config setting.
    """
    
    setting = cfg.data_setting
    architecture = setting.architecture.lower()

    if architecture == "resnet":
        model = generate_resnet3d_model(
            resnet_depth, normal = setting.normal_structure, n_input_channels = n_input_channels, n_classes = setting.n_classes
        ).to(device)

        if pretrain:
            model = generate_resnet3d_model(50, normal = setting.normal_structure, n_classes=1139, n_input_channels=3)
            pretrain = torch.load("/tf/yilian618/ABD_classification/r3d50_KMS_200ep.pth", map_location='cpu')
            model.load_state_dict(pretrain["state_dict"])
            model.fc = nn.Linear(model.fc.in_features, setting.n_classes)
            model = modify_first_conv_layer(model, new_in_channels = n_input_channels, pretrained=True).to(device)

    elif architecture == "unet":
        model = generate_unet_model(
            model_depth = "seg_muti" if class_type == "multiple" else "seg",
            n_classes = setting.n_classes,
            n_input_channels = n_input_channels,
            mask_classes = setting.mask_classes,
        ).to(device)

        if pretrain:
            net_dict = model.state_dict()
            pretrain = torch.load('/tf/yilian618/ABD_classification/r3d50_KMS_200ep.pth')
            pretrain_dict = {
                new_key: v for k, v in pretrain['state_dict'].items()
                if (new_key := k.replace("module.", "")) in net_dict and not new_key.startswith("fc.")
            }
            if "conv1.weight" in pretrain_dict:
                if pretrain_dict["conv1.weight"].shape[1] != n_input_channels:
                    if n_input_channels == 1:
                        pretrain_dict["conv1.weight"] = pretrain_dict["conv1.weight"].mean(dim=1, keepdim=True)
                    else:
                        avg_weight = conv1_weight.mean(dim=1, keepdim=True)
                        pretrain_dict["conv1.weight"] = avg_weight.repeat(1, n_input_channels, 1, 1, 1)[:, :n_input_channels]

            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict, strict=False)
        model = nn.DataParallel(model)

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model
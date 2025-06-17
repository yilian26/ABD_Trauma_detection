import configparser
import ast
from dataclasses import dataclass, field
from typing import List, Tuple, Union
import numpy as np

def getint_with_default(config, section, option, default):
    return config.getint(section, option) if config.has_option(section, option) else default

def getfloat_with_default(config, section, option, default):
    return config.getfloat(section, option) if config.has_option(section, option) else default

@dataclass
class AugmentationConfig:
    size: Tuple[int, int, int]
    num_sample: int

@dataclass
class Rand3DElasticdConfig:
    prob: float
    translate_range: Tuple[int, int, int]
    rotate_range: Tuple[float, float, float]
    scale_range: Tuple[float, float, float]
    sigma_range: Tuple[int, int]
    magnitude_range: Tuple[int, int]

@dataclass
class DataSettingConfig:
    gpu: Union[int, List[int]]
    seed: int
    n_classes: int
    mask_classes: int
    cross_kfold: int
    data_split_ratio: Tuple[float, float, float]
    architecture: str
    imbalance_data_ratio: float
    normal_structure: bool
    epochs: int
    early_stop: int
    traning_batch_size: int
    dataloader_num_workers: int
    valid_batch_size: int
    testing_batch_size: int
    traning_cache_rate: float
    valid_cache_rate: float
    test_cache_rate: float
    init_lr: List[float]
    lr_decay_rate: float
    lr_decay_epoch: int
    loss: str
    bbox: bool
    attention_mask: bool
    img_hu: Union[str, None]
    test_type: str

@dataclass
class TrainingProgressTracker:
    best_metric: float = 0.0
    best_metric_epoch: int = 0
    metric_values: List[float] = field(default_factory=list)
    epoch_loss_values: List[float] = field(default_factory=list)
    epoch_ce_loss_values: List[float] = field(default_factory=list)
    epoch_amse_loss_values: List[float] = field(default_factory=list)
    epoch_metric: List[float] = field(default_factory=list)
    metric_values_loss: List[float] = field(default_factory=list)
    best_metric_loss: float = 1000.0

class ConfigLoader:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self._parse_configs()
        self.config_path = config_path

    def _parse_configs(self):
        self.augmentation = AugmentationConfig(
            num_sample=self.config.getint("Augmentation", "num_sample"),
            size=ast.literal_eval(self.config.get("Augmentation", "size")),
        )

        self.rand3d = Rand3DElasticdConfig(
            prob=self.config.getfloat("Rand3DElasticd", "prob"),
            translate_range=ast.literal_eval(self.config.get("Rand3DElasticd", "translate_range")),
            rotate_range=eval(self.config.get("Rand3DElasticd", "rotate_range"), {"np": np}),
            scale_range=ast.literal_eval(self.config.get("Rand3DElasticd", "scale_range")),
            sigma_range=ast.literal_eval(self.config.get("Rand3DElasticd", "sigma_range")),
            magnitude_range=ast.literal_eval(self.config.get("Rand3DElasticd", "magnitude_range")),
        )

        gpu_val = self.config.get("Data_Setting", "gpu")
        try:
            gpu_parsed = int(gpu_val)
        except ValueError:
            gpu_parsed = ast.literal_eval(gpu_val)

        self.data_setting = DataSettingConfig(
            gpu=gpu_parsed,
            seed=self.config.getint("Data_Setting", "seed"),
            n_classes = getint_with_default(self.config, "Data_Setting", "n_classes", 2),
            mask_classes=getint_with_default(self.config, "Data_Setting", "mask_classes", 1),
            cross_kfold=self.config.getint("Data_Setting", "cross_kfold"),
            data_split_ratio=ast.literal_eval(self.config.get("Data_Setting", "data_split_ratio")),
            architecture=self.config.get("Data_Setting", "architecture"),
            imbalance_data_ratio=self.config.getfloat("Data_Setting", "imbalance_data_ratio"),
            normal_structure=self.config.getboolean("Data_Setting", "normal_structure"),
            epochs=self.config.getint("Data_Setting", "epochs"),
            early_stop=self.config.getint("Data_Setting", "early_stop"),
            traning_batch_size=self.config.getint("Data_Setting", "traning_batch_size"),
            dataloader_num_workers=self.config.getint("Data_Setting", "dataloader_num_workers"),
            valid_batch_size=self.config.getint("Data_Setting", "valid_batch_size"),
            traning_cache_rate = getfloat_with_default(self.config, "Data_Setting", "traning_cache_rate", 1.0),
            valid_cache_rate=getfloat_with_default(self.config, "Data_Setting", "valid_cache_rate", 1.0),
            test_cache_rate=getfloat_with_default(self.config, "Data_Setting", "test_cache_rate", 1.0),
            testing_batch_size=self.config.getint("Data_Setting", "testing_batch_size"),
            init_lr=ast.literal_eval(self.config.get("Data_Setting", "init_lr")),
            lr_decay_rate=self.config.getfloat("Data_Setting", "lr_decay_rate"),
            lr_decay_epoch=self.config.getint("Data_Setting", "lr_decay_epoch"),
            loss=self.config.get("Data_Setting", "loss"),
            bbox=self.config.getboolean("Data_Setting", "bbox"),
            attention_mask=self.config.getboolean("Data_Setting", "attention_mask"),
            img_hu=self.config.get("Data_Setting", "img_hu"),
            test_type=self.config.get("Data_Setting", "test_type"),
        )

        self.training_tracker = TrainingProgressTracker()

    def write_output_section(
        self,
        all_time: str,
        file_list: list,
        accuracy_list: list,
        test_accuracy_list: list,
        epoch_list: list,
    ):
        self.config["Data output"] = {}
        self.config["Data output"]["Running time"] = all_time
        self.config["Data output"]["Data file name"] = str(file_list)
        self.config["Data output"]["Best accuracy"] = str(accuracy_list)
        self.config["Data output"]["Best Test accuracy"] = str(test_accuracy_list)
        self.config["Data output"]["Best epoch"] = str(epoch_list)

        with open(self.config_path, "w") as f:
            self.config.write(f)

    def read_output_section(self):
        if "Data output" not in self.config:
            raise ValueError("No 'Data output' section found in the config file.")

        section = self.config["Data output"]

        def safe_eval(val):
            try:
                return ast.literal_eval(val)
            except Exception:
                return val  # fallback to raw string if parsing fails

        return {
            "running_time": section.get("Running time", ""),
            "data_file_name": safe_eval(section.get("Data file name", "[]")),
            "best_accuracy": safe_eval(section.get("Best accuracy", "[]")),
            "best_test_accuracy": safe_eval(section.get("Best Test accuracy", "[]")),
            "best_epoch": safe_eval(section.get("Best epoch", "[]")),
        }



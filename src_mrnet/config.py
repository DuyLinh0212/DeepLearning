import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    task: str = "abnormal"
    data_dir: str = "data"
    labels_dir: str = "labels"
    run_dir: str = "runs/mrnet_efficientnet_b0"
    seed: int = 42
    use_gpu: bool = True
    epochs: int = 20
    batch_size: int = 1
    num_workers: int = 4
    image_size: int = 224
    learning_rate: float = 1e-4
    backbone_lr_mult: float = 0.2
    weight_decay: float = 1e-4
    dropout: float = 0.4
    projected_dim: int = 256
    pretrained: bool = True
    grad_clip_norm: float = 1.0
    mixed_precision: bool = False
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    min_lr: float = 1e-7
    early_stopping_patience: int = 8
    save_every: int = 1
    threshold: float = 0.5
    log_interval: int = 10


def _read_json_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _update_config(config: TrainConfig, values: Dict[str, Any]) -> TrainConfig:
    valid_keys = set(asdict(config).keys())
    for key, value in values.items():
        if value is None:
            continue
        if key not in valid_keys:
            continue
        setattr(config, key, value)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MRNet with EfficientNetB0.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")

    parser.add_argument("--task", type=str, default=None, choices=["abnormal", "acl", "meniscus"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--labels_dir", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--backbone_lr_mult", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--projected_dim", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--scheduler_factor", type=float, default=None)
    parser.add_argument("--scheduler_patience", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--log_interval", type=int, default=None)

    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def build_config(args: Optional[argparse.Namespace] = None) -> TrainConfig:
    parsed_args = args if args is not None else parse_args()
    config = TrainConfig()

    if parsed_args.config:
        config_values = _read_json_config(parsed_args.config)
        config = _update_config(config=config, values=config_values)

    cli_values = vars(parsed_args).copy()
    cli_values.pop("config", None)
    config = _update_config(config=config, values=cli_values)
    return config


def config_to_dict(config: TrainConfig) -> Dict[str, Any]:
    return asdict(config)

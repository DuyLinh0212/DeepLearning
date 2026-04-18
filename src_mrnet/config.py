import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    task: str = "abnormal"
    data_dir: str = "data"
    labels_dir: str = "labels"
    exp_name: str = "test"
    seed: int = 42
    use_gpu: bool = True
    pretrained: bool = True
    max_epoch: int = 50
    starting_epoch: int = 0
    batch_size: int = 1
    num_workers: int = 8
    image_size: int = 224
    target_slices: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-4
    patience: int = 5
    save_model: int = 1
    log_train: int = 100
    log_val: int = 10
    threshold: float = 0.5


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
    parser = argparse.ArgumentParser(description="Train baseline MRNet with EfficientNetB0.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")

    parser.add_argument("--task", type=str, default=None, choices=["abnormal", "acl", "meniscus"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--labels_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--starting_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--target_slices", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--save_model", type=int, default=None)
    parser.add_argument("--log_train", type=int, default=None)
    parser.add_argument("--log_val", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
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

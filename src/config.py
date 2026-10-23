
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    config_path = ROOT / "configs" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required = ["model", "tracking", "video", "monitoring"]
    for section in required:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    model = config["model"]
    for key in ["weights", "confidence", "image_size", "device"]:
        if key not in model:
            raise ValueError(f"Missing model config key: {key}")

    if not 0 <= model["confidence"] <= 1:
        raise ValueError("Confidence must be between 0 and 1.")

    if model["image_size"] <= 0:
        raise ValueError("Image size must be positive.")

    return config

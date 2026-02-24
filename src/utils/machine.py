"""Machine-specific configuration resolution.

Resolves machine-specific settings (dataset paths, worker counts, W&B tags)
by matching the current hostname against configs in configs/machines/*.yaml.

Can be overridden with the VCNN_MACHINE environment variable.
"""

import os
import socket
from pathlib import Path

import yaml


def resolve_machine_config(cfg: dict, project_root: Path) -> tuple:
    """Resolve machine-specific config by hostname or VCNN_MACHINE env var.

    Machine config values act as defaults — experiment config values always
    take priority.

    Args:
        cfg: The experiment config dict (will be modified in-place).
        project_root: Path to the project root (parent of configs/).

    Returns:
        (cfg, machine_name) tuple.
    """
    machine_dir = project_root / "configs" / "machines"
    if not machine_dir.exists():
        return cfg, "unknown"

    machine_cfg = None
    machine_name = "unknown"

    # 1. Check VCNN_MACHINE env var (explicit override)
    env_machine = os.environ.get("VCNN_MACHINE", "").lower().strip()
    if env_machine:
        machine_file = machine_dir / f"{env_machine}.yaml"
        if machine_file.exists():
            with open(machine_file) as f:
                machine_cfg = yaml.safe_load(f)
            machine_name = machine_cfg.get("machine", {}).get("name", env_machine)

    # 2. Fall back to hostname matching
    if machine_cfg is None:
        host = socket.gethostname().lower()
        for yaml_file in sorted(machine_dir.glob("*.yaml")):
            with open(yaml_file) as f:
                candidate = yaml.safe_load(f)
            match = candidate.get("machine", {}).get("hostname_match", "")
            if match and match in host:
                machine_cfg = candidate
                machine_name = candidate.get("machine", {}).get("name", yaml_file.stem)
                break

    if machine_cfg is None:
        print("[WARN] No machine config matched. Using experiment config as-is.")
        return cfg, "unknown"

    print(f"[INFO] Machine: {machine_name}")

    # Apply machine defaults
    paths = machine_cfg.get("paths", {})
    data_defaults = machine_cfg.get("data_defaults", {})

    # ImageNet root — only fill in if experiment config has a placeholder
    if cfg.get("data", {}).get("dataset", "").lower() in ("imagenet", "imagenet100"):
        current_root = cfg.get("data", {}).get("root", "")
        if current_root in (None, "<IMAGENET_ROOT>", "", "None"):
            imagenet_root = paths.get("imagenet_root")
            if imagenet_root:
                cfg.setdefault("data", {})["root"] = imagenet_root
                print(f"[INFO] ImageNet root: {imagenet_root}")

    # Log dir — machine config provides a default, but CLI --log_dir overrides
    log_dir = paths.get("log_dir")
    if log_dir:
        cfg.setdefault("_machine", {})["log_dir"] = log_dir

    # Data loading defaults (only fill in when not set in experiment config)
    data = cfg.setdefault("data", {})
    if "num_workers" not in data and "num_workers" in data_defaults:
        data["num_workers"] = data_defaults["num_workers"]
    if "gpu_transforms" not in data and "gpu_transforms" in data_defaults:
        data["gpu_transforms"] = data_defaults["gpu_transforms"]

    # DALI thread defaults
    dali = data.get("dali", {})
    if dali and "num_threads" not in dali and "dali_num_threads" in data_defaults:
        dali["num_threads"] = data_defaults["dali_num_threads"]
        data["dali"] = dali
    elif "dali" not in data and "dali_num_threads" in data_defaults:
        # Don't create dali section if experiment doesn't use it
        pass

    # W&B tags — append machine tags to experiment tags
    machine_wandb = machine_cfg.get("wandb", {})
    machine_tags = machine_wandb.get("tags", [])
    if machine_tags:
        wandb_cfg = cfg.setdefault("wandb", {})
        existing_tags = wandb_cfg.get("tags", [])
        wandb_cfg["tags"] = list(set(existing_tags + machine_tags + [machine_name]))

    return cfg, machine_name

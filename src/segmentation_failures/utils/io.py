import json
import os
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf


def save_json(content, target_name):
    target_name = str(target_name)
    if not str(target_name).endswith(".json"):
        target_name += ".json"
    with open(target_name, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def load_json(target_name):
    target_name = str(target_name)
    if not str(target_name).endswith(".json"):
        target_name += ".json"
    with open(target_name, encoding="utf-8") as f:
        content = json.load(f)
    return content


def load_expt_config(expt_dir: Union[str, Path], resolve=False):
    if isinstance(expt_dir, str):
        expt_dir = Path(expt_dir)
    if os.environ.get("EXPERIMENT_ROOT_DIR", None) is None:
        os.environ["EXPERIMENT_ROOT_DIR"] = "dummy_expt_root_dir"
    if os.environ.get("TESTDATA_ROOT_DIR", None) is None:
        os.environ["TESTDATA_ROOT_DIR"] = "dummy_ds_root_dir"

    if resolve:
        hydra_config = OmegaConf.load(expt_dir / ".hydra" / "hydra.yaml").hydra
        # We need the hydra resolver because the run directory is interpolated using hydra... Not sure if there's a better way
        OmegaConf.register_new_resolver(
            "hydra",
            lambda path: OmegaConf.select(hydra_config, path),
            replace=False,  # for safety
        )
    config = OmegaConf.load(expt_dir / ".hydra" / "config.yaml")
    # I had problems with resolution later, so I just do it here.
    # Not sure how it works under the hood.
    if resolve:
        OmegaConf.resolve(config)
        OmegaConf.clear_resolver("hydra")
    return config

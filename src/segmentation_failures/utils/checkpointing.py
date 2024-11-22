from pathlib import Path
from typing import List

from segmentation_failures.utils.io import load_expt_config


def get_experiments_for_seed_fold(
    search_dir: str | Path, seed: int | list[int], fold: int | list[int]
) -> List[Path]:
    if isinstance(seed, int):
        seed = [seed]
    if isinstance(fold, int):
        fold = [fold]
    # Search in search_dir for the experiment with the given seed and fold.
    search_dir = Path(search_dir)
    if not search_dir.exists():
        raise FileNotFoundError(
            f"Could not find directory {search_dir} for automatic segmentation checkpoint selection."
        )
    matches = []
    for rundir in search_dir.iterdir():
        if not rundir.is_dir():
            continue
        # need to load the config to get the seed and fold
        seg_config = load_expt_config(rundir)
        if "fold" in seg_config.datamodule:
            curr_fold = seg_config.datamodule.fold
        else:
            # fold location in config changed at some point;
            # this is for compatibility with older experiments
            curr_fold = seg_config.datamodule.hparams.fold
        if seg_config.seed in seed and curr_fold in fold:
            matches.append(rundir)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find any experiment with seed {seed} and fold {fold} in {search_dir}."
        )
    return matches


def get_checkpoint_from_experiment(expt_dir: str, last_ckpt: bool) -> Path:
    checkpoint_dir = expt_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Could not find directory {expt_dir}/checkpoints.")
    checkpoints_files = [x for x in Path(checkpoint_dir).iterdir() if x.suffix == ".ckpt"]
    not_last_ckpts = []
    last_ckpt_file = None
    for ckpt in checkpoints_files:
        if ckpt.stem == "last":
            last_ckpt_file = ckpt
            if last_ckpt:
                return ckpt
        if ckpt.stem != "last":
            if not ckpt.name.startswith("epoch"):
                # Just a check
                raise ValueError("Expected checkpoints to start with 'epoch'")
            not_last_ckpts.append(ckpt)
    if len(not_last_ckpts) == 0:
        return last_ckpt_file  # better than nothing
    return sorted(not_last_ckpts)[-1]

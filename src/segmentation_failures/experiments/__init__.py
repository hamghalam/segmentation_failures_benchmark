from .experiment import Experiment


def get_experiments(task: str, group_name: str):
    if task == "train_seg":
        return get_segmentation_experiments(task, group_name)
    elif task in ["validate_pixel_csf", "test_pixel_csf"]:
        return get_pixelcsf_experiments(task, group_name)
    elif task in ["train_image_csf", "test_fd"]:
        return get_failuredet_experiments(task, group_name)
    else:
        raise ValueError(f"Task {task} not supported")


def filter_experiments(experiments: list[Experiment], **kwargs) -> list[Experiment]:
    """Filters predefined experiments by combining all provided properties with AND."""

    def filter_expt(expt: Experiment):
        for k, v in kwargs.items():
            expt_value = getattr(expt, k)
            if expt_value is not None:
                v = type(expt_value)(v)
            if expt_value != v:
                return False
        return True

    return list(filter(filter_expt, experiments))


def get_segmentation_experiments(task, group_name=None) -> list[Experiment]:
    """Get a list of all pre-defined segmentation experiments.

    Args:
        group_name (str, optional): If desired, group the experiments under this name. Defaults to None.

    Returns:
        list[Experiment]: list of experiment instances
    """
    _experiments = []
    nnunet_datasets = (
        "acdc",
        "prostate_gonzalez",
        "kits23",
        "brats19_lhgg",
        "mnms",
        "covid_gonzalez",
        "retina",
        "retouch_cirrus",
        "retouch_spectralis",
        "retouch_topcon",
        "mvseg23",
        "octa500",
    )
    all_folds = tuple(range(25))
    all_seeds = (0, 1, 2, 3, 4)
    dynamic_backbones = (
        "dynamic_unet",
        "dynamic_unet_dropout",
        "dynamic_wideunet",
        "dynamic_wideunet_dropout",
        "dynamic_resencunet",
        "dynamic_resencunet_dropout",
        "dynamic_resencunet_deepsup",
        # no deepsup + dropout because dropout layer is appended
    )

    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=("simple_fets22_corrupted", "simple_fets22"),
            fold=all_folds,
            seed=all_seeds,
            backbone=("monai_unet", "monai_unet_dropout"),
            segmentation=("baseline",),
            csf_pixel=(None,),
            csf_image=(None,),
            csf_aggregation=(None,),
        )
    )
    # ---
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=(
                "fets22_corrupted",
                "acdc_dynamic",
                "covid_dynamic",
            ),
            fold=all_folds,
            seed=all_seeds,
            backbone=dynamic_backbones,
            segmentation=("dynunet",),
            csf_pixel=(None,),
            csf_image=(None,),
            csf_aggregation=(None,),
        )
    )
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=nnunet_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=dynamic_backbones,
            segmentation=("dynunet",),
            csf_pixel=(None,),
            csf_image=(None,),
            csf_aggregation=(None,),
        )
    )
    return _experiments


def get_failuredet_experiments(task, group_name=None) -> list[Experiment]:
    """Get a list of all pre-defined confidence experiments.

    Args:
        group_name (str, optional): If desired, group the experiments under this name. Defaults to None.

    Returns:
        list[Experiment]: list of experiment instances
    """
    _experiments = []

    brats_toy_datasets = ("simple_fets22_corrupted", "simple_fets22")
    nnunet_datasets = (
        "acdc",
        "prostate_gonzalez",
        "kits23",
        "brats19_lhgg",
        "mnms",
        "covid_gonzalez",
        "retouch_cirrus",
        "mvseg23",
        "octa500",
        "retina",
    )
    vae_variants = (
        "vae_image_and_mask",
        "vae_image_only",
        "vae_mask_only",
        # "vae_iterative_surrogate",
    )
    all_folds = tuple(range(25))
    all_seeds = (0, 1, 2, 3, 4)
    dynamic_backbones = (
        "dynamic_unet",
        "dynamic_unet_dropout",
        "dynamic_wideunet",
        "dynamic_wideunet_dropout",
        "dynamic_resencunet",
        "dynamic_resencunet_dropout",
    )

    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=brats_toy_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=("monai_unet", "monai_unet_dropout"),
            segmentation=("baseline",),
            csf_pixel=("baseline", "mcdropout", "deep_ensemble"),
            csf_image=(None,),
            csf_aggregation=(
                "all_simple",
                "predictive_entropy+heuristic",
                "predictive_entropy+radiomics",
            ),
        )
    )
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=brats_toy_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=("monai_unet", "monai_unet_dropout"),
            segmentation=("baseline",),
            csf_pixel=(None,),
            csf_image=(
                "mahalanobis",
                "mahalanobis_gonzalez",
            ),
            csf_aggregation=(None,),
        )
    )
    # quality regression (two-stage)
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=brats_toy_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=("monai_unet", "monai_unet_dropout"),
            segmentation=("baseline",),
            csf_pixel=("baseline", "mcdropout", "deep_ensemble"),
            csf_image=("quality_regression",),
            csf_aggregation=(None,),
        )
    )
    # VAE (two-stage)
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=brats_toy_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=("monai_unet", "monai_unet_dropout"),
            segmentation=("baseline",),
            csf_pixel=("baseline", "mcdropout", "deep_ensemble"),
            csf_image=vae_variants,
            csf_aggregation=(None,),
        )
    )
    # ---
    # nnunet datasets
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=nnunet_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=dynamic_backbones,
            segmentation=("dynunet",),
            csf_pixel=("baseline", "mcdropout", "deep_ensemble"),
            csf_image=(None,),
            csf_aggregation=(
                "all_simple",
                "predictive_entropy+heuristic",
                "predictive_entropy+radiomics",
            ),
        )
    )
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=nnunet_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=dynamic_backbones,
            segmentation=("dynunet",),
            csf_pixel=(None,),
            csf_image=(
                "mahalanobis",
                "mahalanobis_gonzalez",
            ),
            csf_aggregation=(None,),
        )
    )
    # Quality regression
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=nnunet_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=dynamic_backbones,
            segmentation=("dynunet",),
            csf_pixel=("baseline", "mcdropout", "deep_ensemble"),
            csf_image=("quality_regression",),
            csf_aggregation=(None,),
        )
    )
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=nnunet_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=dynamic_backbones,
            segmentation=("dynunet",),
            csf_pixel=("baseline", "mcdropout", "deep_ensemble"),
            csf_image=vae_variants,
            csf_aggregation=(None,),
        )
    )
    return _experiments


def get_pixelcsf_experiments(task: str, group_name: str = None) -> list[Experiment]:
    """Get a list of all pre-defined validation experiments.

    Args:
        group_name (str, optional): If desired, group the experiments under this name. Defaults to None.

    Returns:
        list[Experiment]: list of experiment instances
    """
    _experiments = []
    nnunet_datasets = (
        "acdc",
        "prostate_gonzalez",
        "kits23",
        "brats19_lhgg",
        "mnms",
        "covid_gonzalez",
        "retina",
        "retouch_cirrus",
        "retouch_spectralis",
        "retouch_topcon",
        "mvseg23",
        "octa500",
    )
    pixel_csf_methods = (None, "baseline", "mcdropout", "deep_ensemble")
    all_folds = tuple(range(25))
    all_seeds = (0, 1, 2, 3, 4)
    dynamic_backbones = (
        "dynamic_unet",
        "dynamic_unet_dropout",
        "dynamic_wideunet",
        "dynamic_wideunet_dropout",
        "dynamic_resencunet",
        "dynamic_resencunet_dropout",
    )

    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=("simple_fets22_corrupted", "simple_fets22"),
            fold=all_folds,
            seed=all_seeds,
            backbone=("monai_unet", "monai_unet_dropout"),
            segmentation=("baseline",),
            csf_pixel=pixel_csf_methods,
            csf_image=(None,),
            csf_aggregation=(None,),
        )
    )
    _experiments.extend(
        Experiment.from_iterables(
            group=group_name,
            task=task,
            dataset=nnunet_datasets,
            fold=all_folds,
            seed=all_seeds,
            backbone=dynamic_backbones,
            segmentation=("dynunet",),
            csf_pixel=pixel_csf_methods,
            csf_image=(None,),
            csf_aggregation=(None,),
        )
    )
    return _experiments

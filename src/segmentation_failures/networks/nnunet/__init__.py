import os

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

from segmentation_failures.utils.data import get_dataset_dir


def build_network(
    dataset_id,
    plans_name,
    config_name,
    input_channels,
    num_outputs,
    allow_init=True,
    deep_supervision=False,
):
    preproc_data_base = get_dataset_dir(dataset_id, os.environ["nnUNet_preprocessed"])
    plans_manager = PlansManager(preproc_data_base / f"{plans_name}.json")
    configuration_manager: ConfigurationManager = plans_manager.get_configuration(config_name)
    return get_network_from_plans(
        arch_class_name=configuration_manager.network_arch_class_name,
        arch_kwargs=configuration_manager.network_arch_init_kwargs,
        arch_kwargs_req_import=configuration_manager.network_arch_init_kwargs_req_import,
        input_channels=input_channels,
        output_channels=num_outputs,
        allow_init=allow_init,
        deep_supervision=deep_supervision,
    )

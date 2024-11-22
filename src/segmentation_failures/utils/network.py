import torch.nn as nn
from loguru import logger


def disable_dropout(nn_module):
    found_dropout = False
    for layer in nn_module.named_modules():
        if isinstance(layer[1], nn.modules.dropout.Dropout):
            layer[1].eval()
            found_dropout = True
    if not found_dropout:
        logger.warning("No dropout layers found in model. Cannot disable dropout.")


def enable_dropout(nn_module):
    found_dropout = False
    for layer in nn_module.named_modules():
        if isinstance(layer[1], nn.modules.dropout.Dropout):
            layer[1].train()
            found_dropout = True
    if not found_dropout:
        logger.warning("No dropout layers found in model. Cannot enable dropout.")

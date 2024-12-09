# pylint: disable=invalid-name, too-few-public-methods
"""Models for experiments and functions for setting them up"""

from importlib import import_module
from torch import nn, optim

from omegaconf import DictConfig


def criterion_factory(cfg: DictConfig, model_cfg: DictConfig):
    """Criterion factory"""
    if "custom_criterion" not in cfg.model or not cfg.model.custom_criterion:
        criterion = CEloss()
    else:
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            get_criterion = model_module.get_criterion
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'get_criterion'. Is the function misnamed/not defined?"
            ) from e

        criterion = get_criterion(cfg, model_cfg)

    return criterion


class CEloss:
    """Basic Cross-entropy loss"""

    def __init__(self):
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum")

    def __call__(self, logits, target, model, device):
        ce_loss = self.ce_loss(logits, target)

        return ce_loss


def optimizer_factory(cfg: DictConfig, model_cfg: DictConfig, model):
    """Optimizer factory"""
    if "custom_optimizer" not in cfg.model or not cfg.model.custom_optimizer:
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(model_cfg["lr"]),
        )
    else:
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            get_optimizer = model_module.get_optimizer
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'get_optimizer'. Is the function misnamed/not defined?"
            ) from e

        optimizer = get_optimizer(cfg, model_cfg, model)

    return optimizer


def scheduler_factory(cfg: DictConfig, model_cfg: DictConfig, optimizer):
    """Scheduler factory"""
    if "custom_scheduler" not in cfg.model or not cfg.model.custom_scheduler:
        scheduler = DummyScheduler()
    else:
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            get_scheduler = model_module.get_scheduler
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'get_scheduler'. Is the function misnamed/not defined?"
            ) from e

        scheduler = get_scheduler(cfg, model_cfg, optimizer)

    return scheduler


class DummyScheduler:
    """Dummy scheduler that does nothing"""

    def __init__(self):
        pass

    def step(self, metric):
        pass

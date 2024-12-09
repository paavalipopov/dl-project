# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, unused-argument, too-few-public-methods, no-member, too-many-arguments, line-too-long, too-many-instance-attributes, too-many-locals
""" FBNetGen model module from https://github.com/Wayfear/BrainNetworkTransformer"""

from copy import deepcopy

import numpy as np

from torch.nn import functional as F
from torch import nn, optim
from omegaconf import OmegaConf, DictConfig, open_dict

from src.models.src.bnt_modules import LRScheduler
from src.models.src.fbnetgen_modules import (
    FBNetGenLoss,
    FBNetGenTrainer,
    ConvKRegion,
    GruKRegion,
    Embed2GraphByLinear,
    Embed2GraphByProduct,
    GNNPredictor,
)


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return FBNETGEN(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "extractor_type": "gru",
        "embedding_size": 16,
        "window_size": 4,
        "cnn_pool_size": 16,
        "graph_generation": "product",  # product or linear
        "num_gru_layers": 4,
        "dropout": 0.5,
        "group_loss": True,
        "sparsity_loss": True,
        "sparsity_loss_weight": 1e-4,
        # data shape
        "timeseries_sz": cfg.dataset.data_info.main.data_shape.TS[1],
        "node_sz": cfg.dataset.data_info.main.data_shape.FNC[1],
        "node_feature_sz": cfg.dataset.data_info.main.data_shape.FNC[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
        "optimizer": {"lr": 1e-4, "weight_decay": 1e-4},
        "scheduler": {
            "mode": "cos",  # ['step', 'poly', 'cos']
            "base_lr": 1e-4,
            "target_lr": 1e-5,
            "decay_factor": 0.1,  # for step mode
            "milestones": [0.3, 0.6, 0.9],
            "poly_power": 2.0,  # for poly mode
            "lr_decay": 0.98,
            "warm_up_from": 0.0,
            "warm_up_steps": 0,
        },
    }
    return OmegaConf.create(model_cfg)


def data_postproc(cfg: DictConfig, model_cfg: DictConfig, original_data):
    # FBNetGen requires TS data to have shape [samples, feature_size, time_length]
    # 4 is GRU window_size, time_length must be divisible by it
    data = deepcopy(original_data)
    for key in data:
        ts_data = data[key]["TS"]
        tail = ts_data.shape[1] % 4
        if tail != 0:
            print(f"Cropping '{key}' TS data time length by {tail}")
            ts_data = ts_data[:, :-tail, :]
        ts_data = np.swapaxes(ts_data, 1, 2)
        data[key]["TS"] = ts_data

        with open_dict(model_cfg):
            model_cfg.timeseries_sz = ts_data.shape[2]

        with open_dict(cfg):
            cfg.dataset.data_info[key].data_shape.TS = ts_data.shape

        print(f"New cfg.dataset.data_info.{key}.data_shape.TS:")
        print(OmegaConf.to_yaml(cfg.dataset.data_info[key].data_shape.TS))
        print("New model config:")
        print(OmegaConf.to_yaml(model_cfg))

    return data


def get_criterion(cfg: DictConfig, model_cfg: DictConfig):
    criterion = FBNetGenLoss(model_cfg)

    return criterion


def get_optimizer(cfg: DictConfig, model_cfg: DictConfig, model):
    optimizer = optim.Adam(
        model.parameters(),
        lr=model_cfg.optimizer.lr,
        weight_decay=model_cfg.optimizer.weight_decay,
    )

    return optimizer


def get_scheduler(cfg: DictConfig, model_cfg: DictConfig, optimizer):
    return LRScheduler(cfg, model_cfg, optimizer)


def get_trainer(
    cfg, model_cfg, dataloaders, model, criterion, optimizer, scheduler, logger
):
    return FBNetGenTrainer(
        cfg, model_cfg, dataloaders, model, criterion, optimizer, scheduler, logger
    )


class FBNETGEN(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        assert model_cfg.extractor_type in ["cnn", "gru"]
        assert model_cfg.graph_generation in ["linear", "product"]
        assert model_cfg.timeseries_sz % model_cfg.window_size == 0

        self.graph_generation = model_cfg.graph_generation
        if model_cfg.extractor_type == "cnn":
            self.extract = ConvKRegion(
                out_size=model_cfg.embedding_size,
                kernel_size=model_cfg.window_size,
                time_series=model_cfg.timeseries_sz,
            )
        elif model_cfg.extractor_type == "gru":
            self.extract = GruKRegion(
                out_size=model_cfg.embedding_size,
                kernel_size=model_cfg.window_size,
                layers=model_cfg.num_gru_layers,
            )
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                model_cfg.embedding_size, roi_num=model_cfg.node_sz
            )
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_cfg.embedding_size, roi_num=model_cfg.node_sz
            )

        self.predictor = GNNPredictor(
            model_cfg.node_feature_sz,
            roi_num=model_cfg.node_sz,
            n_classes=model_cfg.output_size,
        )

    def forward(self, time_seires, node_feature):
        x = self.extract(time_seires)
        x = F.softmax(x, dim=-1)
        m = self.emb2graph(x)
        m = m[:, :, :, 0]

        return self.predictor(m, node_feature), m

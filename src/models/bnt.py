# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, unused-argument, too-few-public-methods, no-member, too-many-arguments, line-too-long, too-many-instance-attributes
""" BNT model module from https://github.com/Wayfear/BrainNetworkTransformer"""


import numpy as np

from torch import nn, optim
import torch

from omegaconf import OmegaConf, DictConfig, open_dict

from src.models.src.bnt_modules import LRScheduler, TransPoolingEncoder


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return BrainNetworkTransformer(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "sizes": [360, 100],  # Note: The input node size should not be included here
        "pooling": [False, True],
        "pos_encoding": "none",  # 'identity', 'none'
        "orthogonal": True,
        "freeze_center": True,
        "project_assignment": True,
        "pos_embed_dim": 360,
        "node_sz": cfg.dataset.data_info.main.data_shape[1],
        "node_feature_sz": cfg.dataset.data_info.main.data_shape[2],
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
    # 4 is the number of heads in the BNT TransPoolingEncoder
    if cfg.dataset.data_info.main.data_shape[2] % 4 != 0:
        addendum_size = 4 - cfg.dataset.data_info.main.data_shape[2] % 4
        print(f"Adding {addendum_size} column(s) of zeros to the FNC matrices")

        with open_dict(model_cfg):
            model_cfg.node_feature_sz = model_cfg.node_feature_sz + addendum_size

        for key in original_data:
            fnc = original_data[key]["FNC"]
            addendum = np.zeros((fnc.shape[0], fnc.shape[1], addendum_size))
            expanded_fnc = np.concatenate((fnc, addendum), axis=2)
            original_data[key]["FNC"] = expanded_fnc

            with open_dict(cfg):
                cfg.dataset.data_info[key].data_shape = expanded_fnc.shape

        print("New cfg.dataset.data_info:")
        print(OmegaConf.to_yaml(cfg.dataset.data_info))
        print("New model config:")
        print(OmegaConf.to_yaml(model_cfg))

    return original_data


def get_optimizer(cfg: DictConfig, model_cfg: DictConfig, model):
    optimizer = optim.Adam(
        model.parameters(),
        lr=model_cfg.optimizer.lr,
        weight_decay=model_cfg.optimizer.weight_decay,
    )

    return optimizer


def get_scheduler(cfg: DictConfig, model_cfg: DictConfig, optimizer):
    return LRScheduler(cfg, model_cfg, optimizer)


class BrainNetworkTransformer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = model_cfg.node_feature_sz

        self.pos_encoding = model_cfg.pos_encoding
        if self.pos_encoding == "identity":
            self.node_identity = nn.Parameter(
                torch.zeros(model_cfg.node_sz, model_cfg.pos_embed_dim),
                requires_grad=True,
            )
            forward_dim = model_cfg.node_sz + model_cfg.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = model_cfg.sizes
        sizes[0] = model_cfg.node_sz
        in_sizes = [model_cfg.node_sz] + sizes[:-1]
        do_pooling = model_cfg.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[index],
                    hidden_size=1024,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    orthogonal=model_cfg.orthogonal,
                    freeze_center=model_cfg.freeze_center,
                    project_assignment=model_cfg.project_assignment,
                )
            )

        self.dim_reduction = nn.Sequential(nn.Linear(forward_dim, 8), nn.LeakyReLU())

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, model_cfg.output_size),
        )

    def forward(self, node_feature: torch.tensor):
        (
            bz,
            _,
            _,
        ) = node_feature.shape

        if self.pos_encoding == "identity":
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for _, atten in enumerate(self.attention_list):
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all

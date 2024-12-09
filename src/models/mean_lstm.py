# pylint: disable=invalid-name, missing-function-docstring, no-member
""" MeanLSTM model module """
from random import uniform, randint

import torch
from torch import nn

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return MeanLSTM(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "dropout": 0.8,
        "hidden_size": 210,
        "num_layers": 1,
        "bidirectional": True,
        "lr": 4e-5,
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


def random_HPs(cfg: DictConfig):
    model_cfg = {
        "dropout": uniform(0.1, 0.9),
        "hidden_size": randint(32, 256),
        "num_layers": randint(1, 5),
        "bidirectional": bool(randint(0, 1)),
        "lr": 10 ** uniform(-5, -3),
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


class MeanLSTM(nn.Module):
    """
    LSTM model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        hidden_size: int
        num_layers: int
        bidirectional: bool
    Data info expected in model_cfg:
        input_size: int - input_feature_size
        output_size: int - n_classes
    """

    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        input_size = model_cfg.input_size
        output_size = model_cfg.output_size

        dropout = model_cfg.dropout
        self.hidden_size = model_cfg.hidden_size
        num_layers = model_cfg.num_layers
        self.bidirectional = model_cfg.bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                2 * self.hidden_size if self.bidirectional else self.hidden_size,
                output_size,
            ),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        lstm_output = torch.mean(lstm_output, dim=1)

        fc_output = self.fc(lstm_output)
        return fc_output

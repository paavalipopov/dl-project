# pylint: disable=invalid-name, missing-function-docstring, no-member
""" Transformer model module """
from random import uniform, randint

from torch import nn

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return Transformer(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "dropout": 0.35,
        "head_hidden_size": 120,
        "num_heads": 5,
        "num_layers": 5,
        "lr": 2e-5,
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


def random_HPs(cfg: DictConfig):
    model_cfg = {
        "dropout": uniform(0.1, 0.9),
        "head_hidden_size": randint(4, 128),
        "num_heads": randint(1, 5),
        "num_layers": randint(1, 5),
        "lr": 10 ** uniform(-5, -3),
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


class Transformer(nn.Module):
    """
    Transformer model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        head_hidden_size: int
        num_heads: int
        num_layers: int
    Data info expected in model_cfg:
        input_size: int - input_feature_size
        output_size: int - n_classes
    """

    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        input_size = model_cfg.input_size
        output_size = model_cfg.output_size

        dropout = model_cfg.dropout
        head_hidden_size = model_cfg.head_hidden_size
        num_heads = model_cfg.num_heads
        num_layers = model_cfg.num_layers

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_embed = nn.Linear(input_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        input_embed = self.input_embed(x)

        tf_output = self.transformer(input_embed)
        tf_output = tf_output[:, 0, :]

        fc_output = self.fc(tf_output)
        return fc_output

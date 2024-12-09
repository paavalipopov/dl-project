# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, unused-argument, too-few-public-methods, no-member, too-many-arguments, line-too-long, too-many-instance-attributes, too-many-locals
""" MILC model module from https://github.com/UsmanMahmood27/MILC"""

from copy import deepcopy

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn, optim
from omegaconf import OmegaConf, DictConfig

from src.settings import WEIGHTS_ROOT


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    model = MILC(model_cfg)

    if model_cfg.pretrained:
        path = WEIGHTS_ROOT.joinpath("whole_milc.pth")
        try:
            model_dict = torch.load(path)
            model.load_state_dict(model_dict)
        except RuntimeError as e:
            if "Error(s) in loading state_dict" in str(e):
                problematic_weights = []
                if "encoder" in str(e):
                    problematic_weights.append("encoder")
                if "decoder" in str(e):
                    problematic_weights.append("decoder")
                if "lstm" in str(e):
                    problematic_weights.append("lstm")
                print(
                    f"Error(s) while loading pre-trained weights: {problematic_weights}"
                )
                print(f"Discarding {problematic_weights} weights and trying again")

                bad_keys = []
                for key in model_dict.keys():
                    for black_mark in problematic_weights:
                        if black_mark in key:
                            bad_keys.append(key)
                for key in bad_keys:
                    del model_dict[key]

                model.load_state_dict(model_dict, strict=False)

            else:
                raise e

    return model


def default_HPs(cfg: DictConfig):
    # sliding window params
    window_size = 20
    window_shift = 10
    # data params
    n_classes = cfg.dataset.data_info.main.n_classes

    model_cfg = {
        "data_params": {},
        "lstm": {"input_feature_size": 256, "hidden_size": 200, "n_layers": 1},
        "optimizer": {
            "lr": 3e-4,
            "eps": 1e-5,
        },
        "reg_param": 1e-3,
        "pretrained": True,
    }

    for key in cfg.dataset.data_info:
        _, time_length, feature_size = cfg.dataset.data_info[key]["data_shape"]
        assert (
            time_length >= window_size
        ), f"Input time length {time_length} is smaller than the time window size {window_size}"

        model_cfg["data_params"][key] = {
            "input_size": feature_size,
            "output_size": n_classes,
            "window_size": window_size,
            "window_shift": window_shift,
            "n_windows": (time_length - window_size) // window_shift + 1,
        }
    return OmegaConf.create(model_cfg)


def data_postproc(cfg: DictConfig, model_cfg: DictConfig, original_data):
    # Apply sliding window technique to the data
    data = original_data

    new_data = {}
    for key in data:
        feature_size = model_cfg.data_params[key]["input_size"]
        window_size = model_cfg.data_params[key]["window_size"]
        window_shift = model_cfg.data_params[key]["window_shift"]
        n_windows = model_cfg.data_params[key]["n_windows"]

        ts_data = data[key]["TS"]

        # set data shape
        sliding_window_data = np.zeros(
            (ts_data.shape[0], n_windows, feature_size, window_size)
        )

        # apply sliding window
        for window_idx in range(n_windows):
            start = window_idx * window_shift
            end = start + window_size

            sliding_window_data[:, window_idx] = ts_data[:, start:end, :].transpose(
                0, 2, 1
            )

        # set new data
        new_data[key] = {}
        new_data[key]["TS"] = sliding_window_data
        new_data[key]["labels"] = data[key]["labels"]

    return new_data


def get_criterion(cfg: DictConfig, model_cfg: DictConfig):
    return MILCregCEloss(model_cfg)


class MILCregCEloss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, model_cfg):
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

        self.reg_param = model_cfg.reg_param

    def __call__(self, logits, target, model, device):
        ce_loss = self.ce_loss(logits, target)

        reg_loss = torch.zeros(1).to(device)

        for name, param in model.lstm.named_parameters():
            if "bias" not in name:
                reg_loss += self.reg_param * torch.sum(torch.abs(param))

        for name, param in model.attn.named_parameters():
            if "bias" not in name:
                reg_loss += self.reg_param * torch.sum(torch.abs(param))

        loss = ce_loss + reg_loss
        return loss


def get_optimizer(cfg: DictConfig, model_cfg: DictConfig, model):
    optimizer = optim.Adam(
        model.parameters(), lr=model_cfg.optimizer.lr, eps=model_cfg.optimizer.eps
    )

    return optimizer


def get_scheduler(cfg: DictConfig, model_cfg: DictConfig, optimizer):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    return scheduler


class MILC(nn.Module):
    """Whole MILC model"""

    def __init__(
        self,
        model_cfg,
    ):
        super().__init__()

        self.encoder = NatureOneCNN(model_cfg)
        self.lstm = LSTM(model_cfg)
        self.attn = nn.Sequential(
            nn.Linear(2 * model_cfg.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(model_cfg.lstm.hidden_size, model_cfg.lstm.hidden_size),
            nn.ReLU(),
            nn.Linear(model_cfg.lstm.hidden_size, model_cfg.data_params.main.output_size),
        )

        self.init_weight()

    def init_weight(self):
        for name, param in self.decoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.65)
        for name, param in self.attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.65)

    def get_attention(self, x):
        weights_list = []
        for X in x:
            result = [torch.cat((X[i], X[-1]), 0) for i in range(X.shape[0])]
            result = torch.stack(result)
            result_tensor = self.attn(result)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)

        weights = weights.squeeze()

        normalized_weights = F.softmax(weights, dim=1)

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), x)

        attn_applied = attn_applied.squeeze()
        return attn_applied

    def forward(self, x):
        bs, nw, fs, ws = x.shape  # [batch_size, n_windows, feature_size, window_size]

        encoder_output = self.encoder(x.view(-1, fs, ws))
        encoder_output = encoder_output.view(bs, nw, -1)

        lstm_output = self.lstm(encoder_output)

        attention_output = self.get_attention(lstm_output)

        logits = self.decoder(attention_output)
        return logits


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class NatureOneCNN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        # original feature size is treated as a number of channels of a 1D image
        feature_size = model_cfg.data_params.main.input_size
        # 1D image has size of the window time length
        window_size = model_cfg.data_params.main.window_size
        # encoder output is passed to LSTM
        output_size = model_cfg.lstm.input_feature_size

        # calculate CNN layers output
        cnn_output_size = window_size - 3
        cnn_output_size = cnn_output_size - 3
        cnn_output_size = cnn_output_size - 2
        final_conv_size = 200 * cnn_output_size

        self.cnn = nn.Sequential(
            self.init_module(nn.Conv1d(feature_size, 64, 4, stride=1)),
            nn.ReLU(),
            self.init_module(nn.Conv1d(64, 128, 4, stride=1)),
            nn.ReLU(),
            self.init_module(nn.Conv1d(128, 200, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            self.init_module(nn.Linear(final_conv_size, output_size)),
        )

    def init_module(self, module):
        # weight_init
        nn.init.orthogonal_(module.weight.data, gain=nn.init.calculate_gain("relu"))
        # bias_init
        nn.init.constant_(module.bias.data, 0)

        return module

    def forward(self, inputs):
        return self.cnn(inputs)


class LSTM(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(self, model_cfg):
        super().__init__()
        input_size = model_cfg.lstm.input_feature_size
        self.hidden_size = model_cfg.lstm.hidden_size
        n_layers = model_cfg.lstm.n_layers

        self.lstm = nn.LSTM(
            input_size,
            self.hidden_size // 2,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.init_weight()

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.65)  # 0.65 is default gain

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        return lstm_output

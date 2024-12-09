# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, unused-argument, too-few-public-methods, no-member, too-many-arguments, line-too-long, too-many-instance-attributes
""" FBNetGen model sub-modules from https://github.com/Wayfear/BrainNetworkTransformer"""

import time

import numpy as np

from torch.nn import Conv1d, MaxPool1d, Linear, GRU
from torch import nn
import torch
from omegaconf import DictConfig

from src.trainer import BasicTrainer

from apto.utils.report import get_classification_report


class FBNetGenLoss:
    def __init__(self, model_cfg: DictConfig):
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum")

        self.group_loss = model_cfg.group_loss
        self.sparsity_loss = model_cfg.sparsity_loss
        self.sparsity_loss_weight = model_cfg.sparsity_loss_weight

    def __call__(self, logits, learned_matrix, target, model, device):
        loss = 2 * self.ce_loss(logits, target)

        if self.group_loss:
            loss += 2 * self.intra_loss(target, learned_matrix) + self.inner_loss(
                target, learned_matrix
            )

        if self.sparsity_loss:
            sparsity_loss = self.sparsity_loss_weight * torch.norm(learned_matrix, p=1)
            loss += sparsity_loss

        return loss

    def inner_loss(self, label, matrixs):
        loss = 0

        if torch.sum(label == 0) > 1:
            loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

        if torch.sum(label == 1) > 1:
            loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

        return loss

    def intra_loss(self, label, matrixs):
        a, b = None, None

        if torch.sum(label == 0) > 0:
            a = torch.mean(matrixs[label == 0], dim=0)

        if torch.sum(label == 1) > 0:
            b = torch.mean(matrixs[label == 1], dim=0)
        if a is not None and b is not None:
            return 1 - torch.mean(torch.pow(a - b, 2))

        return 0


class FBNetGenTrainer(BasicTrainer):
    def run_epoch(self, ds_name):
        """Run single epoch on `ds_name` dataloder"""
        is_train_dataset = ds_name == "train"

        all_scores, all_targets = [], []
        total_loss, total_size = 0.0, 0

        self.model.train(is_train_dataset)
        start_time = time.time()

        with torch.set_grad_enabled(is_train_dataset):
            for ts_data, fnc, target in self.dataloaders[ds_name]:
                ts_data, fnc, target = (
                    ts_data.to(self.device),
                    fnc.to(self.device),
                    target.to(self.device),
                )
                total_size += ts_data.shape[0]

                logits, learned_matrix = self.model(ts_data, fnc)
                loss = self.criterion(
                    logits, learned_matrix, target, self.model, self.device
                )
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()

                if is_train_dataset:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        average_time = (time.time() - start_time) / total_size
        average_loss = total_loss / total_size

        y_test = np.hstack(all_targets)
        y_score = np.vstack(all_scores)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)

        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )

        metrics = {
            ds_name + "_accuracy": report["precision"].loc["accuracy"],
            ds_name + "_score": report["auc"].loc["weighted"],
            ds_name + "_average_loss": average_loss,
            ds_name + "_average_time": average_time,
        }

        return metrics


class GruKRegion(nn.Module):
    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(
            kernel_size, kernel_size, layers, bidirectional=True, batch_first=True
        )

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size * 2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size),
        )

    def forward(self, raw):
        b, k, _ = raw.shape

        x = raw.reshape(b * k, -1, self.kernel_size)

        x, _ = self.gru(x)

        x = x[:, -1, :]

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class ConvKRegion(nn.Module):
    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(
            in_channels=k, out_channels=32, kernel_size=kernel_size, stride=2
        )

        output_dim_1 = (time_series - kernel_size) // 2 + 1

        self.conv2 = Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16, kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size),
        )

    def forward(self, x):
        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b * k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class Embed2GraphByProduct(nn.Module):
    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):
        m = torch.einsum("ijk,ipk->ijp", x, x)
        m = torch.unsqueeze(m, -1)
        return m


class GNNPredictor(nn.Module):
    def __init__(self, node_input_dim, roi_num=360, n_classes=2):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim),
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear(8 * roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, m, node_feature):
        bz = m.shape[0]

        x = torch.einsum("ijk,ijp->ijp", m, node_feature)

        x = self.gcn(x)

        x = x.reshape((bz * self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum("ijk,ijp->ijp", m, x)

        x = self.gcn1(x)

        x = x.reshape((bz * self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum("ijk,ijp->ijp", m, x)

        x = self.gcn2(x)

        x = self.bn3(x)

        x = x.view(bz, -1)

        return self.fcn(x)


class Embed2GraphByLinear(nn.Module):
    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {
                c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
            }
            labels_onehot = np.array(
                list(map(classes_dict.get, labels)), dtype=np.int32
            )
            return labels_onehot

        off_diag = np.ones([roi_num, roi_num])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def forward(self, x):
        batch_sz, region_num, _ = x.shape
        receivers = torch.matmul(self.rel_rec, x)

        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        x = torch.relu(x)

        m = torch.reshape(x, (batch_sz, region_num, region_num, -1))
        return m

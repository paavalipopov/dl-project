# pylint: disable=too-many-function-args, invalid-name, unused-argument
""" ABIDE ROI dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath(
        "abide_roi/data.npz"
    ),
):
    """
    Return ROI ABIDE data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide_roi/data.npz")
    - path to the dataset with labels

    Output:
    features, labels
    """

    with np.load(dataset_path) as npzfile:
        data = npzfile["data"]
        labels = npzfile["labels"]
    # print(data.shape)
    # >>> (863, 200, 316)

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels

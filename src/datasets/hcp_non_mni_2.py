# pylint: disable=too-many-function-args, invalid-name, unused-argument
""" HCP non MNI dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = "/data/users2/ppopov1/datasets/hcp_non_mni/data_2.npz",
):
    """
    Return non MNI HCP data

    Input:
    dataset_path: str = "/data/users2/ppopov1/datasets/hcp_non_mni/data_2.npz"
    - path to the dataset

    Output:
    features, labels
    """

    raw_data = np.load(dataset_path)

    data = raw_data["data"]
    labels = raw_data["labels"]

    return data, labels

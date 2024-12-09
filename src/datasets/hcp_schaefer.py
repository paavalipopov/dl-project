# pylint: disable=too-many-function-args, invalid-name
""" HCP Schaefer 200 ROIs (no FIX-ICA) dataset"""
import numpy as np

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath("hcp_schaefer/data_schaefer_200.npz"),
):
    """
    Return Schaefer 200 ROIs (no FIX-ICA) HCP data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("hcp_schaefer/data_schaefer_200.npz")
    - path to the dataset

    Output:
    features, labels
    """

    # get data
    with np.load(dataset_path) as npzfile:
        data = npzfile["data"]
        labels = npzfile["labels"]

    return data, labels

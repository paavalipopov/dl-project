# pylint: disable=too-many-function-args, invalid-name, unused-argument
""" HCP ROI dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath("hcp_roi"),
):
    """
    Return ROI HCP data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("hcp_roi")
    - path to the dataset

    Output:
    features, labels
    """

    dataset_path1 = dataset_path.joinpath(
        "HCP_fMRI_200ShaeferAtlas_onlytimeserieszscored.npz"
    )
    dataset_path2 = dataset_path.joinpath(
        "HCP_fMRI_remaining190_200ShaeferAtlas_onlytimeserieszscored.npz"
    )

    label_path1 = dataset_path.joinpath("HCPlabelsIhave.csv")
    label_path2 = dataset_path.joinpath("labels_HCP_remaining_190_subjects.csv")

    # get data
    data1 = np.load(dataset_path1)
    data2 = np.load(dataset_path2)
    # print(data1.shape)
    # print(data2.shape)
    # >>> (752, 200, 1200)
    # >>> (190, 200, 1200)

    labels1 = pd.read_csv(label_path1, header=None)
    labels1 = labels1.values.flatten().astype("int")
    # (752,)

    labels2 = pd.read_csv(label_path2, header=None)
    labels2 = labels2.values.flatten().astype("int")
    # (190,)

    data = np.concatenate((data1, data2))
    labels = np.concatenate((labels1, labels2))

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels

# pylint: disable=invalid-name, missing-function-docstring, unused-argument, too-many-arguments, too-few-public-methods
""" Logistic Regression model module """

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from apto.utils.report import get_classification_report

from omegaconf import OmegaConf, DictConfig
from src.dataloader import cross_validation_split


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return LogisticRegression()


def default_HPs(cfg: DictConfig):
    model_cfg = {}
    return OmegaConf.create(model_cfg)


def get_dataloader(cfg: DictConfig, data, k, trial=None):
    """
    Return LR dataloaders
    dataloaders are a dictionary of
    {
        "train": dataloader,
        "valid": dataloader,
        "test": dataloader,
        "any additional test": dataloader,
    }
    Expects the data produced by src.data.data_factory() with common_processor:
    Input data is a dict of
    {
        "main":
            {
            "FNC": flattened tri-FNC data,
            "labels": data,
            },
        "additional test datasets with similar shape"
    }
    {key} dataloaders are dicts {"FNC": tri-FNC data, "labels": labels}
    """
    split_data = {"train": {}, "valid": {}, "test": {}}

    # train/test split
    split_data["train"], split_data["test"] = cross_validation_split(
        data["main"], cfg.mode.n_splits, k
    )

    # train/val split
    splitter = StratifiedShuffleSplit(
        n_splits=cfg.mode.n_trials,
        test_size=split_data["train"]["labels"].shape[0] // cfg.mode.n_splits,
        random_state=42,
    )
    tr_val_splits = list(
        splitter.split(split_data["train"]["labels"], split_data["train"]["labels"])
    )
    train_index, val_index = (
        tr_val_splits[0] if cfg.mode.name == "tune" else tr_val_splits[trial]
    )
    for key in split_data["train"]:
        split_data["train"][key], split_data["valid"][key] = (
            split_data["train"][key][train_index],
            split_data["train"][key][val_index],
        )
        
    # add additional test datasets to split_data
    for key in data:
        if key != "main":
            split_data[key] = data[key]

    # split_data is a valid dataloader for LR model

    return split_data


def get_optimizer(cfg: DictConfig, model_cfg: DictConfig, model):
    return None


def get_trainer(
    cfg, model_cfg, dataloaders, model, criterion, optimizer, scheduler, logger
):
    return LRTrainer(cfg, dataloaders, model, logger)


class LRTrainer:
    """LR training script"""

    def __init__(
        self,
        cfg,
        dataloaders,
        model,
        logger,
    ) -> None:
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.model = model
        self.logger = logger

        self.save_path = self.cfg.run_dir

        # log configs
        self.logger.config.update(
            {"general": OmegaConf.to_container(self.cfg, resolve=True)}
        )

        # save config in the run's directory
        with open(f"{self.save_path}/config.yaml", "w", encoding="utf8") as f:
            OmegaConf.save(self.cfg, f)

    def run(self):
        """Run training script"""
        print("Training model")
        start_time = time.time()
        self.model.fit(
            X=self.dataloaders["train"]["FNC"],
            y=self.dataloaders["train"]["labels"],
        )
        training_time = time.time() - start_time
        self.logger.summary["training_time"] = training_time

        print("Testing trained model")
        test_results = {}
        test_results["training_time"] = training_time
        for key in self.dataloaders:
            if key not in ["train", "valid"]:
                start_time = time.time()
                y_score = self.model.predict_proba(self.dataloaders[key]["FNC"])
                test_time = time.time() - start_time
                average_time = test_time / self.dataloaders[key]["FNC"].shape[0]

                y_pred = np.argmax(y_score, axis=-1).astype(np.int32)
                report = get_classification_report(
                    y_true=self.dataloaders[key]["labels"],
                    y_pred=y_pred,
                    y_score=y_score,
                    beta=0.5,
                )
                metrics = {
                    key + "_accuracy": report["precision"].loc["accuracy"],
                    key + "_score": report["auc"].loc["weighted"],
                    key + "_average_time": average_time,
                }
                test_results.update(metrics)

        self.logger.log(test_results)

        print(f"Test results: {test_results}")
        print("Done!")

        return test_results

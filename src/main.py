import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.model.cnn import Convolutional
from src.model.lightning import LightningModule
from src.model.train import Model
from src.preprocessing.dataloader import Dataloader
from src.utils.mlflow import MLFlowRunManager


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    start = time.time()
    mlflow = MLFlowRunManager().manager
    mlflow.log_dict(cfg, "config")
    dataloader = Dataloader(workers=int(cfg.workers))
    model = Convolutional(cfg)
    if cfg.training == 'lightning':
        model_light = LightningModule(
            model,
            optimizer=torch.optim.SGD(
                model.parameters(), lr=cfg.models.params.train.lr, momentum=cfg.models.params.train.momentum
            ),
            epochs=cfg.models.params.train.epochs,
        )
        model, result = model_light.train_model_lightning(dataloader)
        for k, v in result[0].items():
            mlflow.log_metric(k, v)
            print(f'{k}: {v}')
    elif cfg.training == 'torch':
        model_instance = Model(
            model,
            dataloader,
            optimizer=torch.optim.SGD(
                model.parameters(), lr=cfg.models.params.train.lr, momentum=cfg.models.params.train.momentum
            ),
        )
        model_instance.train_model(cfg.models.params.train.epochs)
        labels, predict = model_instance.test_model()
        results = model_instance.calculate_metrics(labels, predict, cfg.models.params.metrics)
        for key, value in results.items():
            mlflow.log_metric(key, value)
            print(f'{key}: {value}')
    mlflow.log_metric("Run time", time.time() - start)
    mlflow.pytorch.log_model(model, "model")
    mlflow.close()

if __name__ == "__main__":
    main()

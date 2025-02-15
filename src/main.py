import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.model.cnn import Convolutional
from src.model.lightning import LightningModule, FasterCNNLightning
from src.model.models import PretrainedModel
from src.model.train import Model
from src.preprocessing.dataloader import Dataloader
from src.utils.draw_annotations import draw_annotations, sample_images_prediction
from src.utils.mlflow import MLFlowRunManager


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    start = time.time()
    mlflow = MLFlowRunManager()
    mlflow.manager.log_dict(cfg, "config")
    dataloader = Dataloader(
        workers=int(cfg.workers),
        annotations_path=Path(cfg.data.annotations_path),
        batch_size=cfg.models.params.train.batch_size,
    )
    if cfg.models.type == 'cnn':
        model = Convolutional(cfg)
    else:
        if cfg.models.type == 'fr-cnn':
            dataloader.train_loader.collate_fn = dataloader.collate_fn
            dataloader.test_loader.collate_fn = dataloader.collate_fn
        model = PretrainedModel.get_model(cfg)
    optimizer_cls = hydra.utils.get_class(cfg.models.params.optimizers._target_)
    optimizer_params = cfg.models.params.optimizers.params
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    if cfg.training == 'lightning':
        if cfg.models.type == 'fr-cnn':
            model_light = FasterCNNLightning(
                model,
                optimizer=optimizer,
                mlflow=mlflow.logger,
                epochs=cfg.models.params.train.epochs,
            )
        else:
            model_light = LightningModule(
                model,
                optimizer=optimizer,
                mlflow=mlflow.logger,
                epochs=cfg.models.params.train.epochs,
            )
        model, results = model_light.train_model_lightning(dataloader)
        for k, v in results[0].items():
            mlflow.manager.log_metric(k, v)
            print(f'{k}: {v}')
    elif cfg.training == 'torch':
        model_instance = Model(
            model,
            dataloader,
            optimizer=optimizer,
        )
        model_instance.train_model(cfg.models.params.train.epochs)
        labels, predict = model_instance.test_model()
        results = model_instance.calculate_metrics(labels, predict, cfg.models.params.metrics)
        for key, value in results.items():
            mlflow.manager.log_metric(key, value)
            print(f'{key}: {value}')
    if cfg.models.type == 'fr-cnn' and cfg.show_annotations is True:
        sample_predictions, images, sample_labels = sample_images_prediction(model, dataloader.test_loader, 5)
        draw_annotations(images, sample_predictions, sample_labels)
    mlflow.manager.log_metric("Run time", time.time() - start)
    mlflow.manager.pytorch.log_model(model, "model")
    print("Training took: ", time.time() - start)
    mlflow.close()


if __name__ == "__main__":
    main()

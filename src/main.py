import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.model.cnn import Convolutional
from src.model.lightning import LightningModule
from src.model.train import Model
from src.preprocessing.dataloader import Dataloader


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
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
    elif cfg.training == 'torch':
        model_instance = Model(
            model,
            dataloader,
            optimizer=torch.optim.SGD(
                model.parameters(), lr=cfg.models.params.train.lr, momentum=cfg.models.params.train.momentum
            ),
        )
        model = model_instance.train_model(cfg.models.params.train.epochs)
        result = model_instance.test_model()
    print(f'Accuracy of the network on the test images: {result["test"]}')


if __name__ == "__main__":
    main()

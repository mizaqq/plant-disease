import torch

from src.model.cnn import Convolutional
from src.model.lightning import LightningModule
from src.model.train import Model
from src.preprocessing.dataloader import Dataloader
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    dataloader = Dataloader(workers=int(cfg.workers))
    model = Convolutional(cfg)
    if cfg.training == 'lightning':
        model_light = LightningModule(model,**cfg.models.params.train)
        model, result = model_light.train_model_lightning(dataloader)
    elif cfg.training == 'torch':
        model_instance = Model(model, dataloader)
        model = model_instance.train_model(**cfg.models.params.train)
        result = model_instance.test_model()
    print(f'Accuracy of the network on the test images: {result["test"]}')

if __name__ == "__main__":
    main()
import torch

from src.model.cnn import Convolutional
from src.model.train import Model
from src.preprocessing.dataloader import Dataloader
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    dataloader = Dataloader(workers=int(cfg.workers))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Convolutional(cfg)
    model_instance = Model(model, device, dataloader)
    model_instance.train_model(**cfg.models.params.train)
    model_instance.test_model()


if __name__ == "__main__":
    main()

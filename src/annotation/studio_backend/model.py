import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import requests
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image
from pytorch_lightning.loggers import MLFlowLogger

from src.annotation.studio_backend.utils import (
    convert_boxes_to_ls_format,
    parse_annotation_to_fasterrcnn_format_from_list,
    serialize_data,
)
from src.model.lightning import FasterCNNLightning
from src.preprocessing.dataloader import Dataloader, LabelStudioDataset


class NewModel(LabelStudioMLBase):
    def setup(self) -> None:
        """Configure any parameters of your model here"""
        version = self.get("model_version")
        if Path(__file__).parent.joinpath('checkpoints/' + version).exists():
            self.model = torch.load(Path(__file__).parent.joinpath('checkpoints/' + version), weights_only=False)
        else:
            self.model = self.get_model()
        self.model.to('cuda')
        self.model.eval()
        self.transformer = Dataloader.get_transformer(255, 224)
        self.labels = ['curl', 'healthy', 'slug', 'spot']
        if self.get('data') is None:
            self.set('data', serialize_data([]))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs: dict[str, Any]) -> ModelResponse:
        imagepath = Path(os.getcwd() + tasks[0]['data']['image'].split('plant-disease', 1)[1])
        image = Image.open(imagepath).convert("RGB")
        img = self.transformer(image)
        img = img.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        predictions = self.model([img])
        boxes = [convert_boxes_to_ls_format(box, img.shape) for box in predictions[0]["boxes"].detach().cpu().tolist()]
        scores = predictions[0]["scores"].detach().cpu().tolist()
        labels = predictions[0]["labels"].detach().cpu().tolist()
        max_index = scores.index(max(scores))
        box = boxes[max_index]
        label = labels[max_index]
        prediction = {
            "result": [
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": box[0],
                        "y": box[1],
                        "width": box[2],
                        "height": box[3],
                        "rectanglelabels": [self.labels[label]],
                    },
                }
            ]
        }
        return ModelResponse(predictions=[prediction])

    def fit(self, event: str, data: dict, **kwargs: dict[str, Any]) -> None:
        if event == 'START_TRAINING':
            batch_size = 10
            old_model_version = self.get('model_version')
            print(f'Old model version: {old_model_version}')
            index = int(self.get('data_index'))
            print('Starting model training...')
            annotations_data = self.fetch_data()[index:]
            if len(annotations_data) >= batch_size:
                run = self.start_or_get_run()
                mlflow.log_metric('data_training', len(annotations_data))
                mlflow.log_metric('data_total_cases', len(annotations_data) + index)
                self.model = self.start_training(self.model, annotations_data, batch_size, 4, 5, self.transformer, run)
                new_model_version = self.model_version._increment_string(old_model_version)
                version = self.patch_model(self.model)
                self.set('model_version', new_model_version)
                print(f'Model version updated to: {version}')
                torch.save(self.model, Path(__file__).parent.joinpath('checkpoints/' + new_model_version))
                self.model.eval()
                self.set('data_index', str(len(annotations_data)))
            else:
                print('Not enough data for training. Please collect more annotations.')

    @staticmethod
    def start_or_get_run() -> mlflow.ActiveRun:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_URI"))
        if run := mlflow.active_run() is not None:
            return run
        else:
            return mlflow.start_run()

    @staticmethod
    def start_training(
        model: torch.nn.Module,
        data: list,
        batch_size: int,
        workers: int,
        epochs: int,
        transformer: torch.nn.Transformer,
        run: mlflow.ActiveRun,
    ) -> torch.nn.Module:
        model.train()
        logger = MLFlowLogger(
            experiment_name="Default", tracking_uri=os.environ.get("MLFLOW_URI"), run_id=run.info.run_id
        )
        lightning_model = FasterCNNLightning(model, torch.optim.SGD(model.parameters(), momentum=0.9), logger, epochs)
        train, val = torch.utils.data.random_split(data, [0.8, 0.2], torch.Generator().manual_seed(42))
        train_data = LabelStudioDataset(train, transformer)
        val_data = LabelStudioDataset(val, transformer)
        train_dataloader = Dataloader.get_loader(
            train_data, batch_size, workers, shuffle=True, collate_fn=Dataloader.collate_fn
        )
        val_dataloader = Dataloader.get_loader(
            val_data, batch_size, workers, shuffle=False, collate_fn=Dataloader.collate_fn
        )
        test_loader = Dataloader(
            annotations_path=Path('/home/miza/plant-disease/data/annotation/YOLO/leaves'),
            batch_size=batch_size,
            workers=workers,
            train_split=0.9,
            val_split=0.05,
        ).test_loader
        test_loader.collate_fn = Dataloader.collate_fn
        trained_model, _ = lightning_model.train_model_lightning(train_dataloader, val_dataloader, test_loader)
        return trained_model

    def get_model(self) -> torch.nn.Module:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_URI"))
        model = mlflow.pytorch.load_model("models:/label-studio2/latest")
        torch.save(model, Path(__file__).parent.joinpath('checkpoints/' + self.get("model_version")))
        return model

    @staticmethod
    def patch_model(model: torch.nn.Module) -> str:
        mlflow.pytorch.log_model(model, artifact_path="model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_details = mlflow.register_model(model_uri=model_uri, name="label-studio2")
        return str(model_details.version)

    @staticmethod
    def fetch_data(project_id: str = '1') -> list:
        token = os.getenv("ls_token")
        url = f"http://localhost:8080/api/projects/{project_id}/export?exportType=JSON"
        headers = {"Authorization": f"Token {token}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return parse_annotation_to_fasterrcnn_format_from_list(data)
        else:
            return []

from pathlib import Path
from typing import Dict, List, Optional, Any

from src.utils.mlflow import MLFlowRunManager
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image

from src.annotation.studio_backend.utils import (
    convert_boxes_to_ls_format,
    deserialize_data,
    parse_annotation_to_fasterrcnn_format,
    serialize_data,
)
from src.model.lightning import FasterCNNLightning
from src.preprocessing.dataloader import Dataloader, LabelStudioDataset


class NewModel(LabelStudioMLBase):
    def setup(self) -> None:
        """Configure any parameters of your model here"""
        self.set("model_version", "0.0.1")
        self.checkpoint = self.get("last_checkpoint")
        if self.checkpoint is not None:
            self.model = torch.load(self.checkpoint)
        else:
            self.model = self.get_model()
        self.model.to('cuda')
        self.model.eval()
        self.transformer = Dataloader.get_transformer(255, 224)
        self.labels = ['curl', 'healthy', 'slug', 'spot']
        if self.get('data') is None:
            self.set('data', serialize_data([]))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs: dict[str, Any]) -> ModelResponse:
        imagepath = Path('/home/miza').joinpath(Path(tasks[0]['data']['image'][21:]))
        image = Image.open(imagepath).convert("RGB")
        img = self.transformer(image)
        img = img.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        predictions = self.model([img])
        boxes = [convert_boxes_to_ls_format(box, img.shape) for box in predictions[0]["boxes"].detach().cpu().tolist()]
        scores = predictions[0]["scores"].detach().cpu().tolist()
        labels = predictions[0]["labels"].detach().cpu().tolist()
        prediction = {
            "result": [
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": box[0],
                        "y": box[1],
                        "width": box[2] - box[0],
                        "height": box[3] - box[1],
                        "rectanglelabels": [self.labels[labels[i]]],
                    },
                }
                for i, box in enumerate(boxes)
                if scores[i] > 0.5
            ]
        }
        return ModelResponse(predictions=[prediction])

    def fit(self, event: str, data: dict, **kwargs: dict[str, Any]) -> None:
        batch_size = 10
        old_data = deserialize_data(self.get('data'))
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        if event in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            print('Updating data with new annotations...')
            new_data = parse_annotation_to_fasterrcnn_format(data.get('annotation'))
            new_data['image'] = data.get('task')['data']['image'][21:]  # type: ignore
            old_data.append(new_data)
            self.set('data', serialize_data(old_data))
        elif event == 'START_TRAINING':
            print('Starting model training...')

            if len(old_data) >= batch_size:
                self.model = self.start_training(self.model, old_data, batch_size, 4, 1)
                new_model_version = self.model_version._increment_string(old_model_version)
                self.set('model_version', new_model_version)
                torch.save(
                    self.model.state_dict(),
                    Path(__file__).parent.joinpath('checkpoints').joinpath(new_model_version),
                )
                print(f'Model version updated to: {new_model_version}')
                self.model.eval()
                self.set('data', '')

            else:
                print('Not enough data for training. Please collect more annotations.')

    @staticmethod
    def start_training(
        model: torch.nn.Module, data: list, batch_size: int, workers: int, epochs: int
    ) -> torch.nn.Module:
        model.train()
        lightning_model = FasterCNNLightning(
            model, torch.optim.SGD(model.parameters(), momentum=0.9), None, epochs, ['f1']
        )
        dataset = LabelStudioDataset(data, Dataloader.get_transformer(255, 224))
        dataloader = Dataloader.get_loader(dataset, batch_size, workers, shuffle=True, collate_fn=Dataloader.collate_fn)
        trained_model, _ = lightning_model.train_model_lightning(dataloader)
        return trained_model

    @staticmethod
    def get_model() -> torch.nn.Module:
        mlflowhandler = MLFlowRunManager(run_id='c1172abbfc3b45469e0bb755b7cc4168')
        model = mlflowhandler.manager.pytorch.load_model(
            "mlflow-artifacts:/0/c1172abbfc3b45469e0bb755b7cc4168/artifacts/model"
        )
        torch.save(
            model.state_dict(),
            Path(__file__).parent.joinpath('checkpoints').joinpath('model.pth'),
        )
        mlflowhandler.close()
        return model

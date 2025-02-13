import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from src.preprocessing.dataloader import Dataloader


def draw_yolo_annotations(
    model: torch.nn.Module,
    dataloader: Dataloader,
    class_names: List[str] = ['curl', 'healthy', 'slug', 'spot'],
    color_map: dict = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    if color_map is None:
        color_map = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(class_names))}
    model.eval()
    for image, _ in dataloader:
        preds = model(image)
        boxes = preds[0]['boxes'].detach().cpu().numpy()
        labels = preds[0]['labels'].detach().cpu().numpy()
        for box, label, imagex in zip(boxes, labels, image):
            imagex = imagex.permute(1, 2, 0).cpu().numpy()
            image_vis = (imagex * 255).astype(np.uint8)
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness)
            cv2.putText(
                image_vis,
                class_names[label - 1],
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness,
            )

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

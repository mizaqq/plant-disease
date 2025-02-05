from typing import List

import cv2
import numpy as np

from src.preprocessing.dataloader import Dataloader


def draw_yolo_annotations(
    results: list,
    dataloader: Dataloader,
    class_names: List[str] = ['curl', 'healthy', 'slug', 'spot'],
    color_map: dict = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    if color_map is None:
        color_map = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(class_names))}

    for box, label, image in zip(results, dataloader.test_loader):
        h, w = image.shape[:2]
        annotated_image = image.copy()
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        color = color_map[label]

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

        label_text = f"{class_names[label]}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(
            annotated_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )

    return annotated_image

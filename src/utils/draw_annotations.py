import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from src.preprocessing.dataloader import Dataloader


import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import random


def sample_images_prediction(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, num_samples: int = 3
) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    images, label = next(iter(dataloader))
    if isinstance(images, list):
        images = [image.to(device) for image in images]
    else:
        images.to(device)

    indices = random.sample(range(len(images)), min(num_samples, len(images)))  # Sample `num_samples` unique indices
    sampled_images = [images[i] for i in indices]
    sampled_labels = [label[i] for i in indices]
    return model(sampled_images), sampled_images, sampled_labels


def draw_annotations(
    images: list,
    preds: list,
    sample_labels: list,
    class_names: list = ['curl', 'healthy', 'slug', 'spot'],
    color_map: dict = None,
    thickness: int = 2,
) -> None:
    if color_map is None:
        color_map = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(class_names))}

    pil_images = []
    for i, image in enumerate(images):
        image = image.permute(1, 2, 0).cpu().numpy()
        image_vis = (image * 255).astype(np.uint8)

        # Convert to PIL image
        pil_image = Image.fromarray(image_vis)
        draw = ImageDraw.Draw(pil_image)

        # Get predictions for this image
        boxes = preds[i]['boxes'].detach().cpu().numpy()
        labels = preds[i]['labels'].detach().cpu().numpy()
        scores = preds[i]['scores'].detach().cpu().numpy()  # Get confidence scores

        # if len(scores) > 0:
        #     best_idx = np.argmax(scores)  # Index of the highest-confidence prediction
        #     boxes = boxes[best_idx : best_idx + 1]  # Keep only the best box
        #     labels = labels[best_idx : best_idx + 1]  # Keep the corresponding label
        # else:
        #     boxes = np.array([])  # No detections
        #     labels = np.array([])

        # Draw bounding boxes
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = map(int, box)
            color = color_map.get(label - 1, (255, 0, 0))  # Default to red if label missing

            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=thickness)

            if 0 <= label < len(class_names):
                draw.text((x_min, y_min - 10), class_names[label], fill=color)
        samplebox = sample_labels[i]['boxes'].detach().cpu().numpy()
        x_s_min, y_s_min, x_s_max, y_s_max = map(int, samplebox[0])
        draw.rectangle([x_s_min, y_s_min, x_s_max, y_s_max], outline=(255, 0, 0), width=thickness)
        classname = class_names[sample_labels[i]['labels'][0]]
        draw.text(
            (x_s_min, y_s_min - 10),
            classname,
            fill=(255, 0, 0),
        )
        pil_images.append(pil_image)

    # Display all sampled images in a grid
    fig, axes = plt.subplots(1, len(pil_images), figsize=(5 * len(pil_images), 5))
    if len(pil_images) == 1:
        axes = [axes]  # Ensure axes is iterable when only one image

    for ax, img in zip(axes, pil_images):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

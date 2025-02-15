import random

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms


def generate_saliency_map(data: torch.Tensor, model: torch, batch_index: int) -> None:
    input_data = data[0][batch_index].unsqueeze(0).clone()
    input_data.requires_grad = True
    model.eval()
    output = model(input_data)
    target_class = output.argmax(dim=1).item()
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()
    saliency, _ = torch.max(input_data.grad.data.abs(), dim=1)
    saliency = saliency[0].cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    input_data.requires_grad = False
    plt.imshow(input_data[0].permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='hot')
    plt.show()


def generate_grad_cam_map(
    data: torch.Tensor, model: torch, target_layer: torch, batch_index: int, alpha: float = 0.7
) -> None:
    gradients = []
    activations = []

    def backward_hook(module: torch.Tensor, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
        gradients.append(grad_output[0])

    def forward_hook(module: torch.Tensor, input: torch.Tensor, output: torch.Tensor) -> None:
        activations.append(output)

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(data[0][batch_index].unsqueeze(0))
    target_class = output.argmax(dim=1).item()
    loss = output[0, target_class]
    loss.backward()
    grads = gradients[0].cpu().detach().numpy()
    acts = activations[0].cpu().detach().numpy()
    weights = grads.mean(axis=(2, 3), keepdims=True)
    cam = (weights * acts).sum(axis=1).squeeze()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    image = np.array(transforms.ToPILImage()(data[0][batch_index]).resize((224, 224)))
    overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

    image_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()


def get_sample_images(txt_paths_list: list, image_paths_list: list, sample: int = 8) -> list:
    image_text = []
    for path in random.sample(txt_paths_list, sample):
        with open(path, 'r') as f:
            txt = f.read().split('\n')[0]
        image = next(f for f in image_paths_list if path.name.split('.')[0] in f)
        img = plt.imread(image)
        image_text.append((img, txt))
    return image_text


def show_annotations(image_text: list, row_images: int = 4) -> None:
    for j in range(0, len(image_text), row_images):
        fig, ax = plt.subplots(1, row_images, figsize=(15, 15))
        for i in range(j - row_images, j):
            img, txt = image_text[i]
            image_height, image_width, _ = img.shape
            elements = txt.split()
            class_id = int(elements[0])
            x_center, y_center, bbox_width, bbox_height = map(float, elements[1:])

            x_center_pixel = x_center * image_width
            y_center_pixel = y_center * image_height
            bbox_width_pixel = bbox_width * image_width
            bbox_height_pixel = bbox_height * image_height

            x1 = int(x_center_pixel - bbox_width_pixel / 2)
            y1 = int(y_center_pixel - bbox_height_pixel / 2)
            x2 = int(x_center_pixel + bbox_width_pixel / 2)
            y2 = int(y_center_pixel + bbox_height_pixel / 2)
            ax[i - j].imshow(img)
            rect = patches.Rectangle(
                (x1, y1), bbox_width_pixel, bbox_height_pixel, linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax[i - j].add_patch(rect)
            label = f"Class {class_id}"
            ax[i - j].text(x1, y1 - 10, label, color='lime', fontsize=12, backgroundcolor='black')
            plt.axis('off')
        plt.show()

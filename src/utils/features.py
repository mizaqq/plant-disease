import matplotlib.pyplot as plt
import torch


def generate_saliency_map(data, model, batch_index):
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

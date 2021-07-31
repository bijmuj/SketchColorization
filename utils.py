from matplotlib import pyplot as plt
import numpy as np
import wandb


def display(label, sketch, output):
    images = [label, sketch, output]
    plt.figure(figsize = (15, 15))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def log(losses, color, sketch, output, n):
    images = []
    for i in range(n):
        img_row = np.hstack([color[i], sketch[i], output[i]]) * 0.5 + 0.5
        images.append(img_row)
    images = wandb.Image(np.vstack(images), caption="Left: Color, Mid: Sketch, Right: Output")
    step = losses
    step["examples"] = images
    wandb.log(step)
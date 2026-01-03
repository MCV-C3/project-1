import torch
import cv2
import numpy as np

def grad_cam(model, img, target_layer):
    img.requires_grad = True
    out = model(img)
    pred_class = out.argmax()
    score = out[:, pred_class]
    score.backward()

    grads = target_layer.weight.grad
    pooled = grads.mean(dim=[0,2,3])

    fmap = target_layer.weight[0]
    cam = torch.zeros(fmap.shape[1:])

    for i,w in enumerate(pooled):
        cam += w*fmap[i]

    cam = np.maximum(cam.detach().numpy(),0)
    cam /= cam.max()

    return cam

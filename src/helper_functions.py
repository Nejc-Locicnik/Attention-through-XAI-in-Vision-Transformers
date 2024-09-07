import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16
import cv2
import matplotlib.pyplot as plt
from vit_rollout import VITAttentionRollout
import logging

def reshape_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize to 224x224 pixels
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    image = image.transpose(2, 0, 1)  # Convert to (C, H, W) format
    return torch.tensor(image).unsqueeze(0)  # Add batch dimension

def show_mask_on_img():
    pass

def extract_qkv(module, input, output):
    global quaries, keys, values
    print("Output:", output[0].shape)
    quaries, keys, values = output.chunk(3, dim=-1)

if __name__ == "__main__":
    path = 'src/img/plane2.png'

    #model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True, verbose=True)
    model = vit_b_16(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(path)
    #img.resize((224, 224))
    input = transform(img).unsqueeze(0)

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            module.register_forward_hook(extract_qkv)

    #attention_rollout = VITAttentionRollout(model, attention_layer_name='attn_drop', 
    #                                        head_fusion="mean", discard_ratio=0.9)
    # mask = attention_rollout(input)
    #name = "test.png"
    #print(mask)

    with torch.no_grad():
        _ = model(input)

    print(keys.shape)

    #cv2.imshow("test", np.asarray(img))
    #cv2.waitKey(-1)


    
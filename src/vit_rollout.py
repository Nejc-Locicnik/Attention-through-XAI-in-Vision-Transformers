import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights, swin_v2_b, Swin_V2_B_Weights
import cv2
import matplotlib.pyplot as plt

qkv_dict = dict()
cls_out = None

def extract_qkv_vit(name):
    def hook(module, input, output):
        qkv = input[0] @ module.in_proj_weight.T + module.in_proj_bias
        queries, keys, values = qkv.chunk(3, dim=-1)
        qkv_dict[name] = {'Q': queries, 'K': keys, 'V': values}
    return hook

def extract_attention_swin(name):
    pass

def get_attentions():
    attentions = []
    for i in range(12):
        q = qkv_dict[f"encoder.layers.encoder_layer_{i}.self_attention"]["Q"].reshape((1, 197, 12, 64)).permute((0, 2, 1, 3))
        k = qkv_dict[f"encoder.layers.encoder_layer_{i}.self_attention"]["K"].reshape((1, 197, 12, 64)).permute((0, 2, 1, 3))

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        head_dim = q.size(-1)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32)) # norm by sqrt(head_dim)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, num_patches, num_patches]
        attentions.append(attention_weights)
    return attentions

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_result(img, mask):
    # Display the images using matplotlib
    plt.figure(tight_layout=True)

    # Show the input image
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, ::-1])
    plt.title("Input Image")
    plt.axis('off')  # Hide axes

    # Show the masked image
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Attention")  # Assuming 'name' is the title for this image
    plt.axis('off')  # Hide axes
    plt.show()

def rollout(model, attentions, head_fusion, discard_ratio, token = 0):
    result = torch.eye(attentions[0].size(-1)).unsqueeze(0).to(attentions[0].device)
    results = []
    for attention in attentions:
        if head_fusion == 'mean':
            attention_heads_fused = attention.mean(dim=1)  # Fuse heads by averaging
        elif head_fusion == 'max':
            attention_heads_fused = attention.max(dim=1)[0]
        elif head_fusion == 'min':
            attention_heads_fused = attention.min(dim=1)[0]
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), dim=-1, largest=False) # get indices of the 90% smallest

        num_tokens = attention_heads_fused.size(-1)
        indices_unflattened = torch.unravel_index(indices, (num_tokens, num_tokens)) # reshape
        attention_heads_fused[range(1), indices_unflattened[0], indices_unflattened[1]] = 0

        I = torch.eye(attention_heads_fused.size(-1))
        a = (attention_heads_fused + 1.0*I) # + I  is simulatinig the residual connections
        a = a / a.sum(dim=-1) # normalize attention (rows)

        result = torch.matmul(a, result)
        results.append(result)

    mask = result[0, token, 1:] # show CLS token (0), other intersting ones for both.png: cat=137, dog=50
    #mask = result.mean(1)[0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask, results

if __name__ == "__main__":
    
    vit_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    swin_model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
    model = vit_model
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            module.register_forward_hook(extract_qkv_vit(name))

    path = 'src/img/plane2.png'
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])

    img = Image.open(path)
    img.resize((224, 224))
    input = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(input)

    attentions = get_attentions()
    mask, results = rollout(model, attentions, 'mean', 0.90, 0)
    img = img.resize((224, 224))
    np_img = np.array(img)[:, :, ::-1] # flip rgb -> bgr
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    show_result(np_img, mask[:, :, ::-1])


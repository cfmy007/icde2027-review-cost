import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_b0,resnet18,EfficientNet_B0_Weights,ResNet18_Weights

@torch.inference_mode()
def encode_pretrained(arch,images):
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) if arch=='RN18' else efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    if arch=='RN18':model.fc=torch.nn.Identity()
    else:model.classifier=torch.nn.Identity()
    model=model.to(device='cuda',memory_format=torch.channels_last).eval()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    out=[]
    for start in range(0,len(images),192):
        x=torch.as_tensor(images[start:start+192],device='cuda').float()/255
        x=F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False)
        x=((x-mean)/std).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
        out.append(F.normalize(z.float(),dim=1).cpu().numpy())
    return np.concatenate(out)

import numpy as np
import torch
from model import StyledGenerator
import math 
from PIL import Image

def save_image(image, save_path):
    image = np.transpose(image, (1,2,0)).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_path)
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style
device='cuda'
path="res.npy"
vector=np.load(path)
generator = StyledGenerator(512).to(device)
generator.load_state_dict(torch.load('stylegan-256px-new.model')['g_running'])
generator.eval()
mean_style=get_mean_style(generator,device)
step = int(math.log(256, 2)) - 2
with torch.no_grad():
    image=generator( torch.from_numpy(vector).cuda(), step=step, 
            alpha=1,mean_style=mean_style, style_weight=0.7).data.cpu()
    image=image.squeeze(0).numpy()
    print(image)
    save_image(image,'res.png')


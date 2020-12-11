import argparse
from tqdm import tqdm
import numpy as np
import torch
#from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from model import  Discriminator
from torch.nn import functional as F
from utilities.images import load_images, images_to_video, save_image
from utilities.files import validate_path
from torchvision import utils,transforms
import cv2
from torchvision import utils
import math
device='cuda'
discriminator = Discriminator(from_rgb_activate=True).to(device)
discriminator.load_state_dict(torch.load('090000.model')['discriminator'])
discriminator.eval()

class normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_size = 128
        self.mean = torch.tensor([0.5,0.5,0.5], device="cpu").view(-1, 1, 1)
        self.std = torch.tensor([0.5,0.5,0.5], device="cpu").view(-1, 1, 1)

    def forward(self, image):
        image = image / torch.tensor(255).float()
        #image = F.adaptive_avg_pool2d(image, self.image_size)

        image = (image - self.mean) / self.std #-1~1
        return image
length=8
normalize=normalize()
step = int(math.log(128, 2)) - 2
l=np.zeros((30))
for j in range(30):
    y=torch.zeros((length))
    x=torch.zeros((length,3,128,128))
    for i in range(length): #3 for each pictrue
        #image_path='./sample_32/ref_'+str(i)+'.png'
        image_path='./real/'+str(i+10*j)+'.jpg'
        print(image_path)
        reference_image = load_images([image_path])
        reference_image = torch.from_numpy(reference_image)
        reference_image = normalize(reference_image) #normalize
        reference_image = reference_image.detach()
        x[i]=reference_image
    label=discriminator(x.to(device), step=step, alpha=1)
    l[j]=label.mean()
    del(x)
    del(y)
print(l.mean())
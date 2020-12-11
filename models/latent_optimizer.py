import torch.nn.functional as F
from torchvision.models import vgg16
import torch
import math
from torchvision import utils

class PostSynthesisProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mean = 0.5
        self.std = 0.5

    def forward(self, synthesized_image):
        synthesized_image = (synthesized_image) * torch.tensor(255).float()*self.std+self.mean
        synthesized_image = torch.clamp(synthesized_image, min=0, max=255) 

        return synthesized_image

class VGGProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_size = 128
        self.mean = torch.tensor([0.5,0.5,0.5], device="cuda").view(-1, 1, 1)
        self.std = torch.tensor([0.5,0.5,0.5], device="cuda").view(-1, 1, 1)

    def forward(self, image):
        image = image / torch.tensor(255).float()
        #image = F.adaptive_avg_pool2d(image, self.image_size)

        image = (image - self.mean) / self.std #-1~1
        return image

#@torch.no_grad()
def normalize(tensor):
    mean=torch.mean(tensor)
    std= torch.std(tensor)
    if std==0:
        std=1e-14
    return (tensor-mean)/std

class LatentOptimizer(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()
        self.flag=0
        self.synthesizer = synthesizer.cuda().eval()
        self.post_synthesis_processing = PostSynthesisProcessing()
        self.vgg_processing = VGGProcessing()
        self.vgg16 = vgg16(pretrained=True).features[:layer].cuda().eval()
    def normalize(self,tensor):
        mean=torch.mean(tensor)+0.0001*torch.randn(1).to('cuda')
        std= torch.std(tensor)+0.0001*torch.randn(1).to('cuda')
        if std==0:
            std=1e-14
        return (tensor-mean)/std

    def forward(self, dlatents,mean_style,i):
        step = int(math.log(128, 2)) - 2
        # print(dlatents)
        self.flag+=1
        generated_image = self.synthesizer(dlatents, step=step, alpha=1, mean_style=mean_style, style_weight=0.7)
        if self.flag==1:
            utils.save_image(generated_image, './sample9/ref_'+str(i)+'.png', nrow=1, normalize=True, range=(-1, 1))
        if self.flag==500:
            utils.save_image(generated_image, './sample9/done_'+str(i)+'.png', nrow=1, normalize=True, range=(-1, 1))
            self.flag=0
        #print(generated_image)
        
        #generated_image = self.post_synthesis_processing(generated_image)
        #generated_image = self.vgg_processing(generated_image)
        features = self.vgg16(generated_image)
        return features

import argparse
from tqdm import tqdm
import numpy as np
import torch
#from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from model import StyledGenerator
from models.latent_optimizer import LatentOptimizer
from models.image_to_latent import ImageToLatent
from models.losses import LatentLoss
from utilities.hooks import GeneratedImageHook
from utilities.images import load_images, images_to_video, save_image
from utilities.files import validate_path
from torchvision import utils,transforms
import cv2
from torchvision import utils

parser = argparse.ArgumentParser(description="Find the latent space representation of an input image.")
#parser.add_argument("image_path", help="Filepath of the image to be encoded.")
parser.add_argument("dlatent_path", help="Filepath to save the dlatent (WP) at.")

parser.add_argument("--save_optimized_image", default=False, help="Whether or not to save the image created with the optimized latents.", type=bool)
parser.add_argument("--optimized_image_path", default="optimized.png", help="The path to save the image created with the optimized latents.", type=str)
parser.add_argument("--video", default=False, help="Whether or not to save a video of the encoding process.", type=bool)
parser.add_argument("--video_path", default="video.avi", help="Where to save the video at.", type=str)
parser.add_argument("--save_frequency", default=10, help="How often to save the images to video. Smaller = Faster.", type=int)
parser.add_argument("--iterations", default=500, help="Number of optimizations steps.", type=int)
parser.add_argument('--path', type=str, help='path to checkpoint file')
parser.add_argument("--learning_rate", default=0.01, help="Learning rate for SGD.", type=int)
parser.add_argument("--vgg_layer", default=12, help="The VGG network layer number to extract features from.", type=int)
parser.add_argument("--use_latent_finder", default=False, help="Whether or not to use a latent finder to find the starting latents to optimize from.", type=bool)
parser.add_argument("--image_to_latent_path", default="image_to_latent.pt", help="The path to the .pt (Pytorch) latent finder model.", type=str)

args, other = parser.parse_known_args()
device='cuda'

@torch.no_grad()
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


def optimize_latents():
    print("Optimizing Latents.")
    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path)['generator'])
    generator.eval()
    latent_optimizer = LatentOptimizer(generator, args.vgg_layer)
    mean_style = get_mean_style(generator,device)
    total=np.zeros((83*5,512))
    # Optimize only the dlatents.
    for param in latent_optimizer.parameters():
        param.requires_grad_(False)
    
    if args.video or args.save_optimized_image:
        # Hook, saves an image during optimization to be used to create video.
        generated_image_hook = GeneratedImageHook(latent_optimizer.post_synthesis_processing, args.save_frequency)
    
    
    for i in range(5*83): #3 for each pictrue
        path=int(i/5)
        image_path='./data/'+str(path)+'.jpg'
        print(image_path)
        reference_image = load_images([image_path])
        reference_image = torch.from_numpy(reference_image).to(device)
        reference_image = latent_optimizer.vgg_processing(reference_image) #normalize
        reference_features = latent_optimizer.vgg16(reference_image).detach() #vgg
        reference_image = reference_image.detach()
    
        if args.use_latent_finder:
            image_to_latent = ImageToLatent().cuda()
            image_to_latent.load_state_dict(torch.load(args.image_to_latent_path))
            image_to_latent.eval()

            latents_to_be_optimized = image_to_latent(reference_image)
            latents_to_be_optimized = latents_to_be_optimized.detach().cuda().requires_grad_(True)
        else:
            latents_to_be_optimized = torch.randn((1,512)).cuda().requires_grad_(True)
        
        criterion = LatentLoss()
        optimizer = torch.optim.SGD([latents_to_be_optimized], lr=args.learning_rate)

        progress_bar = tqdm(range(args.iterations))
    
        for step in progress_bar:
            optimizer.zero_grad()

            generated_image_features = latent_optimizer(latents_to_be_optimized,mean_style,i)
            #print(latents_to_be_optimized)
            loss = criterion(generated_image_features, reference_features)
            loss.backward()
            loss = loss.item()

            optimizer.step()
            if step==args.iterations:
                break
            with torch.no_grad():
                latents_to_be_optimized.add_(-latents_to_be_optimized.mean()+3e-2*torch.randn(1).to('cuda'))
                latents_to_be_optimized.div_(latents_to_be_optimized.std()+3e-2*torch.randn(1).to('cuda'))

            progress_bar.set_description("Step: {}, Loss: {}".format(step, loss))

        optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
        total[i]=optimized_dlatents[0]
    
    np.save(args.dlatent_path, total)

def main():
    assert(validate_path(args.dlatent_path, "w"))
    assert(1 <= args.vgg_layer <= 16)
    if args.video: assert(validate_path(args.video_path, "w"))
    if args.save_optimized_image: assert(validate_path(args.optimized_image_path, "w"))
    if args.use_latent_finder: assert(validate_path(args.image_to_latent_path, "r"))
    
    optimize_latents()

if __name__ == "__main__":
    main()


    
    



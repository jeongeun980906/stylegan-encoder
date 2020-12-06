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

from torchvision import datasets, transforms, utils

from dataset import MultiResolutionDataset
parser = argparse.ArgumentParser(description="Find the latent space representation of an input image.")
parser.add_argument("--image_path", help="Filepath of the image to be encoded.")
parser.add_argument("--dlatent_path", help="Filepath to save the dlatent (WP) at.")

parser.add_argument("--save_optimized_image", default=False, help="Whether or not to save the image created with the optimized latents.", type=bool)
parser.add_argument("--optimized_image_path", default="optimized.png", help="The path to save the image created with the optimized latents.", type=str)
parser.add_argument("--save_frequency", default=10, help="How often to save the images to video. Smaller = Faster.", type=int)
parser.add_argument("--iterations", default=1000, help="Number of optimizations steps.", type=int)
parser.add_argument('--path', type=str, help='path to trained file')
parser.add_argument("--learning_rate", default=1, help="Learning rate for SGD.", type=int)
parser.add_argument("--vgg_layer", default=12, help="The VGG network layer number to extract features from.", type=int)
parser.add_argument("--use_latent_finder", default=False, help="Whether or not to use a latent finder to find the starting latents to optimize from.", type=bool)
parser.add_argument("--image_to_latent_path", default="image_to_latent.pt", help="The path to the .pt (Pytorch) latent finder model.", type=str)

args= parser.parse_known_args()

def sample_data(dataset, batch_size):
    dataset.resolution = 128
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)

    return loader

def optimize_latents():
    print("Optimizing Latents.")
    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path)['g_running'])
    generator.eval()
    latent_optimizer = LatentOptimizer(generator, args.vgg_layer)
    
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.image_path, transform)
    batch_size=32
    # Optimize only the dlatents.
    for param in latent_optimizer.parameters():
        param.requires_grad_(False)
    loader = sample_data(dataset, batch_size)
    data_loader = iter(loader)
    
    image_to_latent = ImageToLatent().cuda()
    criterion = LatentLoss()
    optimizer = torch.optim.SGD([image_to_latent, lr=args.learning_rate)
    
    progress_bar = tqdm(range(args.iterations))
    
    for step in progress_bar:
        loader = sample_data(dataset, batch_size)
        data_loader = iter(loader)
        reference_image=next(data_loader)
        reference_image = torch.from_numpy(reference_image).cuda()
        reference_image = latent_optimizer.vgg_processing(reference_image)
        reference_features = latent_optimizer.vgg16(reference_image).detach()
        reference_image = reference_image.detach()
        latents_to_be_optimized = image_to_latent(reference_image) #output latent zector by our network

        optimizer.zero_grad()

        generated_image_features = latent_optimizer(latents_to_be_optimized) #by gan generator -> vgg network feature
        
        loss = criterion(generated_image_features, reference_features)
        loss.backward()
        #loss = loss.item()
        optimizer.step()
        progress_bar.set_description("Step: {}, Loss: {}".format(step, loss))
    
    # optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
    # np.save(args.dlatent_path, optimized_dlatents)
    # if args.save_optimized_image:
    #     save_image(generated_image_hook.last_image, args.optimized_image_path)

def main():
    optimize_latents()

if __name__ == "__main__":
    main()


    
    



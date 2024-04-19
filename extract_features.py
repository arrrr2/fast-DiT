import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import argparse
import os
from tqdm import tqdm
from diffusers.models import AutoencoderKL

def main(args):
    device = 'cuda'
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.d, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )
    cur_step = 0
    if args.s == 'single':
        all_feats = torch.zeros((len(dataset), 4, 32, 32), dtype=torch.float32, device='cpu')
        all_labels = torch.zeros((len(dataset),), dtype=torch.int64, device='cpu')
        save_path = os.path.dirname(args.f)
        if not os.path.exists(save_path): os.makedirs(save_path, exist_ok=True)
    else:
        if not os.path.exists(args.f): os.makedirs(args.f, exist_ok=True)
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad(): feats = vae.encode(x).latent_dist.sample().mul_(0.18215)
        if args.s == 'single':
            all_feats[cur_step] = feats.to('cpu')
            all_labels[cur_step] = y.to('cpu')
        else:
            feat_dict = {'feature': feats.to('cpu'), 'label': y.to('cpu')}
            torch.save(feat_dict, f'{save_path}/{cur_step}.pth')
        cur_step += 1
    
    if args.s == 'single':
        feat_dict = {'features': all_feats, 'labels': all_labels}
        torch.save(feat_dict, args.f, _use_new_zipfile_serialization=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=str, default='ema', choices=["ema", "mse"])
    parser.add_argument("-d", type=str, default='/path/to/images')
    parser.add_argument("-f", type=str, default='/path/to/features.pth')
    parser.add_argument("-s", type=str, choice=['single', 'each'], default='single', 
                        help='save single file or separated for each feature')
    args = parser.parse_args()
    main(args)
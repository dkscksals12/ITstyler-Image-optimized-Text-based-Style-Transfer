import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from pathlib import Path
# import models.transformer as transformer
import models.I  as StyTR 
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def denormalzation(tensor, device):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    tensor = torch.clamp(tensor * std + mean, 0., 1.)
    return tensor

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='../../datasets/MSCOCO', type=str,   
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='../../datasets/WikiArt', type=str,  #wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint
parser.add_argument('--sample_path', type=str, default='./samples', help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=5.0)
parser.add_argument('--s_a', type=int, default=10.0)
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--log-image-every', type=int, metavar='<int>', default=100, 
                    help='Period for loging generated images, non-positive for disabling, default=100')
parser.add_argument('--gan_weight', type=float, default=1.0)
parser.add_argument('--decoder', type=str, default='./models/decoder.pth')
args = parser.parse_args()

def main():

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    vgg = StyTR.VGG
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.DECODER
    decoder.load_state_dict(torch.load(args.decoder))
    D = StyTR.MultiDiscriminator().to(device)
    clip = StyTR.CLIP(device=device)
    MLP = StyTR.MLP(512, 512, 1024, 6)
    with torch.no_grad():
        network = StyTR.Net(vgg ,decoder, MLP, clip, args)
    network.train()

    network.to(device)
    # network = nn.DataParallel(network, device_ids=[0, 1])
    content_tf = train_transform()
    style_tf = train_transform()

    valid, fake = 1, 0


    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))
    

    # optimizer = torch.optim.Adam([ 
    #                             {'params': network.module.transformer.parameters()},
    #                             {'params': network.module.decode.parameters()},
    #                             {'params': network.module.embedding.parameters()},        
    #                             ], lr=args.lr)

    optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.MappingNet.parameters()}], lr=args.lr)

    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    if not os.path.exists(args.save_dir+"/test"):
        os.makedirs(args.save_dir+"/test")


    for i in tqdm(range(args.max_iter)):

        if i < 1e4:
            warmup_learning_rate(optimizer, iteration_count=i)
        else:
            adjust_learning_rate(optimizer, iteration_count=i)

        print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)  
        out, loss_s_a, loss_c, loss_s = network(content_images, style_images)

        # train discriminator
        loss_gan_d = (D.compute_loss(style_images, valid) + D.compute_loss(out.detach(), fake)) * 0.5
        optimizer_D.zero_grad()
        loss_gan_d.backward()
        optimizer_D.step()

         # train generator
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss_s_a = args.s_a * loss_s_a
        loss_gan_g = args.gan_weight * D.compute_loss(out, valid)
        loss = loss_c + loss_s  + loss_s_a  + loss_gan_g

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0 or i == 0:
            output_name = '{:s}/t1/{:s}{:s}'.format(
                            args.save_dir, str(i),".jpg"
                        )
            out = torch.cat((content_images, out), 0)
            out = torch.cat((style_images, out), 0)
            save_image(out, output_name)
        
        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        writer.add_scalar('loss_s_a', loss_s_a.item(), i + 1)
        writer.add_scalar('loss_gan_d', loss_gan_d.item(), i + 1)
        writer.add_scalar('loss_gan_g', loss_gan_g.item(), i + 1)

        if args.log_image_every > 0 and ((i + 1) % args.log_image_every == 0 or i == 0 or (i + 1) == args.max_iter):
            writer.add_image('Image/Content', content_images[0], i + 1)
            writer.add_image('Image/Style', style_images[0], i + 1)
            writer.add_image('Image/Generated', out[0], i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = network.MappingNet.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                    '{:s}/MLP2_iter_{:d}.pth'.format(args.save_dir,
                                                            i + 1))

            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                    '{:s}/decoder2_iter_{:d}.pth'.format(args.save_dir,
                                                            i + 1))
            # state_dict = network.clip.state_dict()
            # for key in state_dict.keys():
            #     state_dict[key] = state_dict[key].to(torch.device('cpu'))
            # torch.save(state_dict,
            #         '{:s}/clip_iter_{:d}.pth'.format(args.save_dir,
            #                                                 i + 1))

                                                        
    writer.close()

if __name__ == '__main__':
    main()



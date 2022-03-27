"""
TODO: add DINO training as a possible thing through a flag

TODO: add loss function here, take it out of the model
"""

import argparse
import os
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import models.transformer as transformer
import models.StyTR  as StyTR
from ImageDataset import ImageDataset
from ImageDataset import train_transform
from sampler import InfiniteSamplerWrapper

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', default='./datasets/train2014', type=str,   
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', default='./datasets/Images', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='./weights/vgg_normalised.pth')

    # training options
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=7.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,
                            help="Size of the embeddings (dimension of the transformer)")

    # add additional flags for other experiments
    parser.add_argument('--dino_encoder', default="none", type=str, help="Use a pretrained DINO encoder", choices=("none", "vits16", "vits8", "vitb16", "vitb8"))
    parser.add_argument('--freeze_encoder', action='store_true')
    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # Create logging folders
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if not os.path.exists(args.save_dir+"/test"):
        os.makedirs(args.save_dir+"/test")

    writer = SummaryWriter(log_dir=args.log_dir)

    ### Create the model ###
    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    embedding = StyTR.PatchEmbed()

    Trans = transformer.Transformer()
    with torch.no_grad():
        network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
    network.train()

    network.to(device)
    #network = nn.DataParallel(network, device_ids=[0,1])  # Don't use DataParallel

    ### Create Dataset and DataLoader ###
    content_tf = train_transform()
    style_tf = train_transform()
    content_dataset = ImageDataset(args.content_dir, content_tf)
    style_dataset = ImageDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))
    
    ### Create Optimizer (Jimmy Baaaaaaa) ###
    optimizer = torch.optim.Adam([ 
                                {'params': network.transformer.parameters()},
                                {'params': network.decode.parameters()},
                                {'params': network.embedding.parameters()},        
                                ], lr=args.lr)

    ### Training Loop ###
    for i in tqdm(range(args.max_iter)):
        # learning rate strategy
        if i < 1e4:
            warmup_learning_rate(optimizer, iteration_count=i)
        else:
            adjust_learning_rate(optimizer, iteration_count=i)

        # get images from dataloaders
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device) 

        # pass through model and get loss
        out, loss_c, loss_s,l_identity1, l_identity2 = network(content_images, style_images)
            
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
    
        print("Loss:", loss.sum().cpu().detach().numpy(), "-content:", loss_c.sum().cpu().detach().numpy(), 
              "-style:", loss_s.sum().cpu().detach().numpy(), "-l1:", l_identity1.sum().cpu().detach().numpy(), "-l2:", l_identity2.sum().cpu().detach().numpy()
             )

        # save logging images to test folder every 100 iterations
        if i % 100 == 0:
            print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
            output_name = '{:s}/test/{:s}{:s}'.format(
                            args.save_dir, str(i),".jpg"
                        )
            out = torch.cat((content_images,out),0)
            out = torch.cat((style_images,out),0)
            save_image(out, output_name)

        # write logging stats to Tensorboard
        writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
        writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
        writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
        writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
        writer.add_scalar('total_loss', loss.sum().item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            # save transformer model
            torch.save(network.transformer.state_dict(), '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir, i + 1))

            # save CNN Decoder
            torch.save(network.decode.state_dict(), '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir, i + 1))

            # Save network embeddings
            torch.save(network.embedding.state_dict(), '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir, i + 1))
                   
    writer.close()



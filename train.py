"""
TODO: add DINO training as a possible thing through a flag
"""

import argparse
import os
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from function import calc_mean_std, normal, normal_style
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

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return F.mse_loss(input_mean, target_mean) + F.mse_loss(input_std, target_std)

def calc_content_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    return F.mse_loss(input, target)

def get_loss(model, vgg, content_input, style_input):
    """
    Calculate the loss function
        - Ics: Image output created from content and style
        - loss_c: Content perceptual loss
        - loss_s: Style perceptual loss
        - loss_lambda1: Identity loss between Icc and Ic, Iss and Is
        - loss_lambda2: Identity loss between VGG features of Icc and Ic, Iss and Is
    """
    Ics = model(content_input, style_input)
    Icc = model(content_input, content_input)
    Iss = model(style_input, style_input)
    with torch.no_grad():
        content_feats = vgg(content_input)
        style_feats = vgg(style_input)
        Ics_feats = vgg(Ics)
        Icc_feats = vgg(Icc)
        Iss_feats = vgg(Iss)

    # Content Perceptual Loss
    loss_c = calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
    
    # Style Perceptual Loss
    loss_s = calc_style_loss(Ics_feats[0], style_feats[0])
    for i in range(1, 5):
        loss_s += calc_style_loss(Ics_feats[i], style_feats[i])

    # Identity Loss 1
    loss_lambda1 = calc_content_loss(Icc, content_input) + calc_content_loss(Iss, style_input)
    
    # Identity Loss 2
    loss_lambda2 = calc_content_loss(Icc_feats[0], content_feats[0]) + calc_content_loss(Iss_feats[0], style_feats[0])
    for i in range(1, 5):
        loss_lambda2 += calc_content_loss(Icc_feats[i], content_feats[i]) + calc_content_loss(Iss_feats[i], style_feats[i])

    return Ics.detach(), loss_c, loss_s, loss_lambda1, loss_lambda2


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

    parser.add_argument('--amp', action='store_true', help="Use Automatic Mixed Precision (to help with batch sizes)")
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
    vgg = StyTR.VGGFeats(args.vgg)
    vgg.eval()
    vgg.to(device)

    network = StyTR.StyTrans(args)
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
    if args.freeze_encoder:
        for param in network.transformer.encoder_c.parameters():
            param.requires_grad = False
        for param in network.transformer.encoder_s.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam([ 
                                {'params': filter(lambda p: p.requires_grad, network.transformer.parameters())},
                                {'params': network.decode.parameters()},
                                {'params': network.embedding.parameters()},        
                                ], lr=args.lr)

    if args.amp:
        print("Using Automatic Mixed Precision")
        scaler = torch.cuda.amp.GradScaler()

    ### Training Loop ###
    for i in tqdm(range(args.max_iter)):
        # learning rate strategy
        if i < 1e4:
            warmup_learning_rate(optimizer, iteration_count=i)
        else:
            adjust_learning_rate(optimizer, iteration_count=i)
        optimizer.zero_grad()

        # get images from dataloaders
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device) 

        # pass through model and get loss
        if args.amp:
            with torch.cuda.amp.autocast():
                out, loss_c, loss_s, l_identity1, l_identity2 = get_loss(network, vgg, content_images, style_images) 
        else:
            out, loss_c, loss_s, l_identity1, l_identity2 = get_loss(network, vgg, content_images, style_images) 
            
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)

        if args.amp:
            scaler.scale(loss.sum()).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.sum().backward()
            optimizer.step()
    
        print("Loss:", loss.sum().cpu().detach().numpy(), "-content:", loss_c.sum().cpu().detach().numpy(), 
              "-style:", loss_s.sum().cpu().detach().numpy(), "-l1:", l_identity1.sum().cpu().detach().numpy(), "-l2:", l_identity2.sum().cpu().detach().numpy()
             )

        # write logging stats to Tensorboard
        writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
        writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
        writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
        writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
        writer.add_scalar('total_loss', loss.sum().item(), i + 1)

        # checkpoint and save logging images
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
            output_name = '{:s}/test/{:s}{:s}'.format(
                            args.save_dir, str(i),".jpg"
                        )
            out = torch.cat((content_images, out), 0)
            out = torch.cat((style_images, out), 0)
            save_image(out, output_name)

            torch.save(network.transformer.state_dict(), '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir, i + 1))
            torch.save(network.decode.state_dict(), '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir, i + 1))
            torch.save(network.embedding.state_dict(), '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir, i + 1))
            print("Saved Checkpoints")

        # cleanup
        del out, loss, loss_c, loss_s, l_identity1, l_identity2
                   
    writer.close()



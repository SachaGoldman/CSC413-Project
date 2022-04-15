"""
TODO: add DINO training as a possible thing through a flag
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torchvision import models as torchvision_models
from torchvision.utils import save_image
from tqdm import tqdm

import models.StyTR as StyTR
import util.dino_utils as utils
import vision_transformer as vits
from function import calc_mean_std, normal, normal_style
from ImageDataset import ImageDataset, train_transform
from main_dino import DataAugmentationDINO
from models.transformer import Transformer, TransformerEncoder
from sampler import InfiniteSamplerWrapper
from util.misc import nested_tensor_from_tensor_list


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
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


def get_loss(model, vgg, content_input, style_input, origin_content, origin_style, skip_c_encoder=False, skip_s_encoder=False):
    """
    Calculate the loss function
        - Ics: Image output created from content and style
        - loss_c: Content perceptual loss
        - loss_s: Style perceptual loss
        - loss_lambda1: Identity loss between Icc and Ic, Iss and Is
        - loss_lambda2: Identity loss between VGG features of Icc and Ic, Iss and Is
        - ready_images: (Ics, Icc, Iss) that are already calculated. default=none
    """
    Ics = model(content_input, style_input, skip_c_encoder, skip_s_encoder)
    Icc = model(content_input, content_input, skip_c_encoder, skip_s_encoder)
    Iss = model(style_input, style_input, skip_c_encoder, skip_s_encoder)
    with torch.no_grad():
        content_feats = vgg(origin_content)
        style_feats = vgg(origin_style)
        Ics_feats = vgg(Ics)
        Icc_feats = vgg(Icc)
        Iss_feats = vgg(Iss)

    # Content Perceptual Loss
    loss_c = calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + \
        calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))

    # Style Perceptual Loss
    loss_s = calc_style_loss(Ics_feats[0], style_feats[0])
    for i in range(1, 5):
        loss_s += calc_style_loss(Ics_feats[i], style_feats[i])

    # Identity Loss 1
    loss_lambda1 = calc_content_loss(Icc, origin_content) + calc_content_loss(Iss, origin_style)

    # Identity Loss 2
    loss_lambda2 = calc_content_loss(
        Icc_feats[0], content_feats[0]) + calc_content_loss(Iss_feats[0], style_feats[0])
    for i in range(1, 5):
        loss_lambda2 += calc_content_loss(Icc_feats[i], content_feats[i]) + \
            calc_content_loss(Iss_feats[i], style_feats[i])

    return Ics.detach(), loss_c, loss_s, loss_lambda1, loss_lambda2


def encoder_pre_process(sample_images, embedding):
    if isinstance(sample_images, (list, torch.Tensor)):
        sample_images = nested_tensor_from_tensor_list(sample_images)

    result = embedding(sample_images.tensors).flatten(2).permute(2, 0, 1)
    return result


class DINOLoss_new(nn.Module):
    def __init__(self, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center = 0
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def resize_image(images, new_hw, kernel):
    """
    expect images to have shape (HW, N, C)
    apply 1-D convolution to resize image to 
    (new_hw, N, C)
    """
    # permute shape to (N, C, HW)
    images = images.permute(1, 2, 0)
    C = images.size(1)
    hw = images.size(2)
    p = (new_hw + kernel - 1 - hw) // 2
    conv = nn.Conv1d(in_channels=C, out_channels=C, kernel_size=kernel, padding=p)
    resized_im = conv(images)
    return resized_im.permute(2, 0, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser = get_args_parser()
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
    parser.add_argument('--dino_encoder', default="none", type=str,
                        help="Use a pretrained DINO encoder", choices=("none", "vits16", "vits8", "vitb16", "vitb8"))
    parser.add_argument('--freeze_encoder_c', action='store_true')
    parser.add_argument('--freeze_encoder_s', action='store_true')
    parser.add_argument('--dino_s_encoder_training', type=utils.bool_flag, default=False,
                        help="Use the dino training procedure for the style encoder")
    parser.add_argument('--dino_c_encoder_training', type=utils.bool_flag, default=False,
                        help="Use the dino training procedure for the content encoder")
    parser.add_argument('--dino_encoder_loss', default="none", type=str,
                        help="If using the dino training procedure, decide which loss to use", choices=("none", "embedded", "target"))

    # additional flags for training implemented
    parser.add_argument('--amp', action='store_true',
                        help="Use Automatic Mixed Precision (to help with batch sizes)")
    parser.add_argument('--save_img_interval', type=int, default=1000)
    parser.add_argument('--gradient_accum_steps', type=int, default=1,
                        help="Number of steps to accumulate gradients over")
    parser.add_argument("--max_norm", type=float, default=10.0, help="Clipping gradient norm")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=4, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    args = parser.parse_args()

    # We aren't openAI
    if args.dino_s_encoder_training and args.dino_c_encoder_training and args.dino_encoder_loss == "target":
        sys.exit("Arguments unsupported due to computational cost.")

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
    # network = nn.DataParallel(network, device_ids=[0,1])  # Don't use DataParallel

    ### Create teacher content, style encoders ###
    # note: use network.transformer.encoder_c and network.transformer.encoder_s
    # as the student model
    teacher_transformer = Transformer(dino_encoder="none")
    teacher_encoder_c = teacher_transformer.encoder_c
    teacher_encoder_s = teacher_transformer.encoder_s

    student_encoder_c = network.transformer.encoder_c
    student_encoder_s = network.transformer.encoder_s

    # ==== flags to indicate whether encoders should be skipped when using network
    skip_s_encoder = False
    skip_c_encoder = False

    ### Create Dataset and DataLoader ###
    content_tf = train_transform()
    style_tf = train_transform()

    # ======== image preprocessing modules and datasets for dino method ======== #

    if args.dino_s_encoder_training:
        dino_style_tf = DataAugmentationDINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
        )
        skip_s_encoder = True
        dino_style_dataset = ImageDataset(args.style_dir, dino_style_tf)

        dino_style_iter = iter(data.DataLoader(
            dino_style_dataset, batch_size=args.batch_size,
            sampler=InfiniteSamplerWrapper(dino_style_dataset),
            num_workers=args.n_threads))

    if args.dino_c_encoder_training:
        dino_content_tf = DataAugmentationDINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
        )
        skip_c_encoder = True
        dino_content_dataset = ImageDataset(args.content_dir, dino_content_tf)

        dino_content_iter = iter(data.DataLoader(
            dino_content_dataset, batch_size=args.batch_size,
            sampler=InfiniteSamplerWrapper(dino_content_dataset),
            num_workers=args.n_threads))

    # ======= dino loss ======= #
    dino_c_loss_func = DINOLoss_new(
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.max_iter,
    ).to(device)

    dino_s_loss_func = DINOLoss_new(
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.max_iter,
    ).to(device)

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
    if args.freeze_encoder_c:
        for param in network.transformer.encoder_c.parameters():
            param.requires_grad = False
    if args.freeze_encoder_s:
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

        # # get images from dataloaders
        origin_content_images = next(content_iter).to(device)
        origin_style_images = next(style_iter).to(device)

        # TODO Sacha use a different loss if relevant
        # experiment 3
        if args.dino_c_encoder_training:
            dino_content_images = next(dino_content_iter)
            student_c_out = []
            teacher_c_out = []
            # apply teacher and student to each of the dino-preprocessed image
            # teacher is only applied to the first two global views
            for ind, c_images_batch in enumerate(dino_content_images):
                c_images_batch.to(device)
                c_images_batch = encoder_pre_process(c_images_batch, network.embedding)
                student_encoding_c = student_encoder_c(c_images_batch)
                if ind >= 2:
                    student_encoding_c = resize_image(
                        student_encoding_c, student_c_out[ind - 1].size(0), 21)
                student_c_out.append(student_encoding_c)
                if ind < 2:
                    teacher_c_out.append(teacher_encoder_c(c_images_batch))
            # compute dino loss
            student_c_out = torch.stack(student_c_out)
            teacher_c_out = torch.stack(teacher_c_out)
            dino_c_loss = dino_c_loss_func(student_c_out, teacher_c_out, i)

            # this is the content encoding that will be passed
            # into model (skip encoder since it's already encoded)
            model_content_in = 0.5 * (student_c_out[0] + student_c_out[1])
        else:
            model_content_in = origin_content_images

        # TODO Sacha use a different loss if relevant
        if args.dino_s_encoder_training:
            dino_style_images = next(dino_style_iter)
            student_s_out = []
            teacher_s_out = []
            for ind, s_images_batch in enumerate(dino_style_images):
                s_images_batch.to(device)
                s_images_batch = encoder_pre_process(s_images_batch, network.embedding)
                student_encoding_s = student_encoder_s(s_images_batch)
                if ind >= 2:
                    student_encoding_s = resize_image(
                        student_encoding_s, student_s_out[ind - 1].size(0), 21)
                student_s_out.append(student_encoding_s)
                if ind < 2:
                    teacher_s_out.append(teacher_encoder_s(s_images_batch))
            student_s_out = torch.stack(student_s_out)
            teacher_s_out = torch.stack(teacher_s_out)
            dino_s_loss = dino_s_loss_func(student_s_out, teacher_s_out, i)

            # this is the style encoding that will be passed
            # into model (skip encoder since it's already encoded)
            model_style_in = 0.5 * (student_s_out[0] + student_s_out[1])
        else:
            model_style_in = origin_style_images

        # pass through model and get loss
        if args.amp:
            with torch.cuda.amp.autocast():
                out, loss_c, loss_s, l_identity1, l_identity2 = get_loss(
                    network, vgg, model_content_in, model_style_in,
                    origin_content_images, origin_style_images,
                    skip_c_encoder, skip_s_encoder)
        else:
            out, loss_c, loss_s, l_identity1, l_identity2 = get_loss(
                network, vgg, model_content_in, model_style_in,
                origin_content_images, origin_style_images,
                skip_c_encoder, skip_s_encoder)

        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = (loss_c + loss_s + (l_identity1 * 70) +
                (l_identity2 * 1)).sum() / args.gradient_accum_steps

        if args.dino_c_encoder_training:
            loss += dino_c_loss

        if args.dino_s_encoder_training:
            loss += dino_s_loss

        if args.amp:
            scaler.scale(loss).backward()
            if i % args.gradient_accum_steps == 0:
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_norm)
            if i % args.gradient_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print("Loss:", loss.cpu().detach().numpy() * args.gradient_accum_steps, "-content:", loss_c.sum().cpu().detach().numpy(),
              "-style:", loss_s.sum().cpu().detach().numpy(), "-l1:", l_identity1.sum(
        ).cpu().detach().numpy(), "-l2:", l_identity2.sum().cpu().detach().numpy()
        )
        if args.dino_c_encoder_training:
            print("student & teacher content loss: ", dino_c_loss.sum().cpu().detach().numpy())

        if args.dino_s_encoder_training:
            print("student & teacher style loss: ", dino_s_loss.sum().cpu().detach().numpy())

        # write logging stats to Tensorboard
        writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
        writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
        writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
        writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
        writer.add_scalar('total_loss', loss.item() * args.gradient_accum_steps, i + 1)

        # checkpoint and save logging images
        if i == 0 or (i + 1) % args.save_img_interval == 0 or (i + 1) == args.max_iter:
            output_name = '{:s}/test/{:s}{:s}'.format(
                args.save_dir, str(i+1), ".jpg"
            )
            out = torch.cat((origin_content_images, out), 0)
            out = torch.cat((origin_style_images, out), 0)
            save_image(out, output_name, nrow=args.batch_size)

        # cleanup
        del out, loss, loss_c, loss_s, l_identity1, l_identity2

        if args.dino_c_encoder_training:
            del dino_c_loss

        if args.dino_s_encoder_training:
            del dino_s_loss

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
            torch.save(network.transformer.state_dict(),
                       '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir, i + 1))
            torch.save(network.decode.state_dict(),
                       '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir, i + 1))
            torch.save(network.embedding.state_dict(),
                       '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir, i + 1))
            print("Saved Checkpoints")

    writer.close()

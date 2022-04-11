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


from main_dino import DINOLoss, DataAugmentationDINO, get_args_parser
import vision_transformer as vits
from torchvision import models as torchvision_models
import util.dino_utils as utils
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

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

def get_args_parser():
    parser = argparse.ArgumentParser('DINO + StyTR', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    #####  below are the arguments added from the train.py file #####
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
    parser.add_argument('--freeze_encoder_c', action='store_true')
    parser.add_argument('--freeze_encoder_s', action='store_true')

    # additional flags for training implemented
    parser.add_argument('--amp', action='store_true', help="Use Automatic Mixed Precision (to help with batch sizes)")
    parser.add_argument('--save_img_interval', type=int, default=1000)
    parser.add_argument('--gradient_accum_steps', type=int, default=1, help="Number of steps to accumulate gradients over")
    parser.add_argument("--max_norm", type=float, default=10.0, help="Clipping gradient norm")
    

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
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
    parser.add_argument('--freeze_encoder_c', action='store_true')
    parser.add_argument('--freeze_encoder_s', action='store_true')

    # additional flags for training implemented
    parser.add_argument('--amp', action='store_true', help="Use Automatic Mixed Precision (to help with batch sizes)")
    parser.add_argument('--save_img_interval', type=int, default=1000)
    parser.add_argument('--gradient_accum_steps', type=int, default=1, help="Number of steps to accumulate gradients over")
    parser.add_argument("--max_norm", type=float, default=10.0, help="Clipping gradient norm")
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
    

    ### Create student and teacher content, style encoders ###
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student_cont_encoder, student_sty_encoder = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher_cont_encoder, teacher_sty_encoder = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student_cont_encoder.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student_cont_encoder, student_sty_encoder = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher_cont_encoder, teacher_sty_encoder = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student_cont_encoder.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student_cont_encoder, student_sty_encoder = torchvision_models.__dict__[args.arch]()
        teacher_cont_encoder, teacher_sty_encoder = torchvision_models.__dict__[args.arch]()
        embed_dim = student_cont_encoder.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student_cont_encoder, student_sty_encoder = utils.MultiCropWrapper(student_cont_encoder, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher_cont_encoder, teacher_sty_encoder = utils.MultiCropWrapper(
        teacher_cont_encoder,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student_cont_encoder, teacher_cont_encoder, student_sty_encoder, teacher_sty_encoder = student_cont_encoder.cuda(), teacher_cont_encoder.cuda(), student_sty_encoder.cuda(), teacher_sty_encoder.cuda()
    
    # teacher and student start with the same weights
    teacher_cont_encoder.load_state_dict(student_cont_encoder.module.state_dict())
    teacher_sty_encoder.load_state_dict(student_sty_encoder.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    
    for p in teacher_cont_encoder.parameters():
        p.requires_grad = False

    for p in teacher_sty_encoder.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")


    # ============ preparing loss for dino... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()


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
                                {'params': utils.get_params_groups(student_cont_encoder)}, 
                                {'params': utils.get_params_groups(student_sty_encoder)},             
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

        # get images from dataloaders
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device) 

        # pass through model and get loss
        # TODO: rewrite this part to reduce redundancy

        auto_cast_enabled = False
        if args.amp:
            auto_cast_enabled = True

        with torch.cuda.amp.autocast(enabled=auto_cast_enabled):
            # feed content & style images into student and teacher encoders.
            t_cont_out, t_sty_out = teacher_cont_encoder(content_images[:2]), teacher_sty_encoder(style_images[:2])

            s_cont_out, s_sty_out = student_cont_encoder(content_images), student_sty_encoder(style_images)

            dino_c_loss = dino_loss(s_cont_out, t_cont_out, i)
            dino_s_loss = dino_loss(s_sty_out, t_sty_out, i)

            out, loss_c, loss_s, l_identity1, l_identity2 = get_loss(network, vgg, content_images, style_images) 
       
            
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = (loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)).sum() / args.gradient_accum_steps + dino_c_loss + dino_s_loss

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
              "-style:", loss_s.sum().cpu().detach().numpy(), "-l1:", l_identity1.sum().cpu().detach().numpy(), "-l2:", l_identity2.sum().cpu().detach().numpy()
             )

        # write logging stats to Tensorboard
        writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
        writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
        writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
        writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
        writer.add_scalar('total_loss', loss.item() * args.gradient_accum_steps, i + 1)

        # checkpoint and save logging images
        if i == 0 or (i + 1) % args.save_img_interval == 0 or (i + 1) == args.max_iter:
            output_name = '{:s}/test/{:s}{:s}'.format(
                            args.save_dir, str(i+1),".jpg"
                        )
            out = torch.cat((content_images, out), 0)
            out = torch.cat((style_images, out), 0)
            save_image(out, output_name, nrow=args.batch_size)

        # cleanup
        del out, loss, loss_c, loss_s, l_identity1, l_identity2, dino_c_loss, dino_s_loss

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
            torch.save(network.transformer.state_dict(), '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir, i + 1))
            torch.save(network.decode.state_dict(), '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir, i + 1))
            torch.save(network.embedding.state_dict(), '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir, i + 1))
            print("Saved Checkpoints")
                   
    writer.close()



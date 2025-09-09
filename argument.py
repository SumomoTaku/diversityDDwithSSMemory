import argparse
import os

from misc.reproduce import set_arguments
from models import DiT_models

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


# def default_epoch_byipc(ipc, factor, nclass=10, bound=-1):
def default_epoch_by_ipc(arg):
    """Calculating training epochs for ImageNet"""
    if arg.epochs:
        prt_freq = arg.epochs // 100
    elif arg.dataset == 'imagenet':
        factor = max(arg.factor, 1)
        ipc = arg.ipc * (factor ** 2)
        ipc_ticks = [0, 1, 10, 50, 200, 500]
        epoch_ticks = [3000, 2000, 1500, 1000, 500, 300]
        epoch = ([300] + [v for i, v in enumerate(epoch_ticks) if ipc > ipc_ticks[i]])[-1]
        # ???
        if arg.nclass == 100:
            epoch = int((2 / 3) * epoch)
            epoch = epoch - (epoch % 100)
        arg.epochs = epoch
        prt_freq = arg.epochs // 100
    else:
        arg.epochs = 1000
        prt_freq = arg.epochs

    arg.print_freq = arg.print_freq if arg.print_freq else prt_freq


def dataset_default_para(args_attr):
    dataset = args_attr.dataset
    attrs = ["nch", "size", "mixup", "mixup_net", "mix_p", "dsa", "dsa_strategy"]
    targets = {
        "cifar": [3, 32, "cut", "cut", 0.5, True, "color_crop_flip_scale_rotate"],
        "svhn": [3, 32, "cut", "cut", 0.5, True, "color_crop_scale_rotate"],
        "mnist": [1, 28, "cut", "cut", 0.5, True, "color_crop_scale_rotate"],
        "fashion": [1, 28, "cut", "cut", 0.5, True, "color_crop_flip_scale_rotate"],
        "imagenet": [3, 224, "cut", "cut", 1.0, False, "color_crop_flip_scale_rotate"],
        # For speech data, I didn't use data augmentation
        # For speech data, I didn't use data augmentation
        "speech": [1, 64, "vanilla", "vanilla", 1.0, False, ""]
    }
    if dataset in targets.keys():
        for attr, v in zip(attrs, targets[dataset]):
            setattr(args_attr, attr, v)
def get_args():
    parser = argparse.ArgumentParser()
    common_parser = argparse.ArgumentParser(description="common args", add_help=False)
    # common arguments
    common_parser.add_argument("--dit_model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    common_parser.add_argument("--dit_ckpt", type=str, default=None, help="Optional path to a DiT checkpoint (default: auto-download).")
    common_parser.add_argument("--n_model_class", type=int, default=1000, help="Class number of DiT model")

    common_parser.add_argument("--nclass", type=int, default=10, help='the class number for distillation training')
    common_parser.add_argument('--select_list', type=str, help='file containing classes desired')
    common_parser.add_argument("--ipc", type=int, default=100, help='Desired IPC for generation')
    common_parser.add_argument('--epochs', type=int, help='number of training epochs')
    common_parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for training')
    common_parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

    common_parser.add_argument("--save_dir", type=str, default='../results', help='the directory to put the generated images')
    common_parser.add_argument('--print_freq', type=int, help='print frequency')

    common_parser.add_argument("--image_size", type=int, choices=[224, 256, 512], default=256)
    common_parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    common_parser.add_argument('--seed', default=0, type=int, help='random seed for training')
    common_parser.add_argument('--dseed', default=0, type=int, help='random seed for data loading')


    subparsers = parser.add_subparsers(dest="command", required=True)
    # train_dit.py
    dit_parser = subparsers.add_parser("train_dit", parents=[common_parser], help="Train DiT")
    dit_parser.add_argument("--train_dir", type=str, required=True)
    dit_parser.add_argument("--num_classes", type=int, default=1000, help='the class number for the total dataset')
    dit_parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    dit_parser.add_argument("--num_workers", type=int, default=4)
    dit_parser.add_argument("--ckpt_freq", type=int, default=500)
    dit_parser.add_argument("--finetune_ipc", type=int, default=-1,
                        help='the number of samples participating in the fine-tuning (-1 for all)')
    dit_parser.add_argument("--distill", action="store_true", default=False, help='whether conduct distillation')
    dit_parser.add_argument('--lambda_real', default=0.002, type=float, help='weight for representativeness constraint')
    dit_parser.add_argument('--lambda_gen', default=0.008, type=float, help='weight for diversity constraint')
    dit_parser.add_argument("--memory_size", type=int, default=64, help='the memory size')

    # sample.py
    sp_parser = subparsers.add_parser("sample", parents=[common_parser], help="Generate dataset")
    sp_parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    sp_parser.add_argument("--num_denoising", type=int, default=50, help="Denoising steps for diffusion")
    sp_parser.add_argument("--total_shift", type=int, default=0, help='Index offset for the file name')
    sp_parser.add_argument("--all_cls", type=str, default=0, help='File of all classes to get original labels')

    # train_downstream.py
    # Dataset
    ds_parser = subparsers.add_parser("train_downstream", parents=[common_parser], help="Train downstream classification model")
    ds_parser.add_argument('-d', '--dataset', type=str, help="name of dataset",
                        choices=["mnist", "fashion", "svhn", "cifar10", "cifar100", "imagenet"])
    ds_parser.add_argument('--data_dir', type=str, help='directory at containing dataset.(for mnist et al)')
    ds_parser.add_argument('--train_sub', type=str, default="", help='subdirectory name of training dataset')
    ds_parser.add_argument('--val_sub', type=str, default="", help='subdirectory name of validation dataset')
    ds_parser.add_argument('--train_dir', type=str, help='directly specify the train dataset. (for imagenet or customized dataset)')
    ds_parser.add_argument('--val_dir', type=str, help='directly specify the validation dataset. (for imagenet or customized dataset)')

    ds_parser.add_argument('--nclass_sub', default=-1, type=int, help='number of classes for each process')
    ds_parser.add_argument('-l', '--load_memory', type=str2bool, default=True, help='load training images on the memory')
    ds_parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers')

    # Network
    ds_parser.add_argument('-n', '--net_type', default='convnet', type=str, help='network type',
                        choices=["resnet", "resnet_ap", "convnet"])
    ds_parser.add_argument('--depth', default=10, type=int, help='depth of the network')
    ds_parser.add_argument('--width', default=1.0, type=float, help='width of the network')
    ds_parser.add_argument('--norm_type', default='instance', type=str,
                        choices=['batch', 'instance', 'sn', 'none'])

    # Training
    ds_parser.add_argument('--pretrained', action='store_true')

    # Experiment
    ds_parser.add_argument('--val_freq', type=int, default=1)
    ds_parser.add_argument('--save_ckpt', action='store_true', help='save the checkpoints of the model')
    ds_parser.add_argument('--repeat', default=1, type=int, help='number of test repetition')

    # Optimization
    ds_parser.add_argument('--mixup', default='cut', type=str,help='mixup choice for evaluation',
                        choices=('vanilla', 'cut'))
    ds_parser.add_argument('--mixup_net', default='cut', type=str,help='mixup choice for training networks in condensation stage',
                        choices=('vanilla', 'cut'))
    ds_parser.add_argument('--beta', default=1.0, type=float, help='mixup beta distribution')
    ds_parser.add_argument('--mix_p', default=1.0, type=float, help='mixup probability')

    ds_parser.add_argument('--rrc',type=str2bool, default=True, help='use random resize crop for ImageNet')
    ds_parser.add_argument('--rrc_size', type=int, default=-1)
    ds_parser.add_argument('--dsa', type=str2bool, default=False, help='Use DSA augmentation for evaluation or not')
    ds_parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')

    # IDC. Code NOT found
    ds_parser.add_argument('-s', '--slct_type',type=str, default='idc', help='data condensation type ',
                        choices=["idc", "dsa", "kip", "random", "herding"])
    # ds_parser.add_argument('-f', '--factor', type=int, default=1, help='multi-formation factor. (1 for IDC-I)')
    # ds_parser.add_argument('-a', '--aug_type', type=str, default='color_crop_cutout',
    #                     help='augmentation strategy for condensation matching objective')
    # ## Matching objective
    # ds_parser.add_argument('--match', type=str, default='grad', help='feature or gradient matching',
    #                     choices=['feat', 'grad'])
    # ds_parser.add_argument('--metric', type=str, default='l1',help='matching objective',
    #                     choices=['mse', 'l1', 'l1_mean', 'l2', 'cos'])
    # ds_parser.add_argument('--bias', type=str2bool, default=False, help='match bias or not')
    # ds_parser.add_argument('--fc', type=str2bool, default=False, help='match fc layer or not')
    # ds_parser.add_argument('--f_idx', type=str, default='4', help='feature matching layer. comma separation')
    ## Optimization
    ## For small datasets, niter=2000 is enough for the full convergence.
    ## For faster optimzation, you can early stop the code based on the printed log.
    # ds_parser.add_argument('--niter', type=int, default=500, help='number of outer iteration')
    # ds_parser.add_argument('--inner_loop', type=int, default=100, help='number of inner iteration')
    # ds_parser.add_argument('--early', type=int, default=0, help='number of pretraining epochs for condensation networks')
    # ds_parser.add_argument('--fix_iter', type=int, default=-1, help='number of outer iteration maintaining the condensation networks')
    # ds_parser.add_argument('--net_epoch', type=int, default=1, help='number of epochs for training network at each inner loop')
    # ds_parser.add_argument('--n_data', type=int, default=500, help='number of samples for training network at each inner loop')
    # ds_parser.add_argument('--pt_from', type=int, default=-1, help='pretrained networks index')
    # ds_parser.add_argument('--pt_num', type=int, default=1, help='pretrained networks range')
    # ds_parser.add_argument('--batch_syn_max', type=int, default=128,
    #                     help='maximum number of synthetic data used for each matching (ramdom sampling for large synthetic data)')
    # ds_parser.add_argument('--lr_img', type=float, default=5e-3, help='condensed data learning rate')
    # ds_parser.add_argument('--mom_img', type=float, default=0.5, help='condensed data momentum')
    ds_parser.add_argument('--reproduce', action='store_true', help='for reproduce our setting')
    # ds_parser.add_argument('--same_compute', type=str2bool, default=False, help='match evaluation training steps for IDC')
    args = parser.parse_args()

    # some conditional statements of train_downstream
    if args.command == "train_downstream":
        # name and path
        if args.data_dir:
            args.train_dir = os.path.join(args.data_dir, args.sub_train)
            args.val_dir = os.path.join(args.data_dir, args.sub_val)
        else:
            assert args.train_dir and args.val_dir, f"specify --data_dir or --train_dir & --val_dir"
        # only cifar differs in cifar10 and cifar100 et al.
        args.spec_dataset = args.dataset
        if args.dataset.startswith("cifar"):
            args.dataset = "cifar"

        if args.reproduce:
            args = set_arguments(args)

        # data
        dataset_default_para(args)
        default_epoch_by_ipc(args)

        # idc setting
        if args.slct_type == "idc":
            args.init = "mix" if args.factor > 1 else "random"
            if args.match == 'feat':
                f_list = [int(s) for s in args.f_idx.split(',')]
                if len(f_list) == 1:
                    f_list.append(-1)
                args.idx_from, args.idx_to = f_list
                args.metric = 'mse'

            imsize_bases= {"cifar": 32, "svhn": 32, "mnist": 28, "fashion": 32, "imagenet": 32, "speech": 64 }
            ipc_base = 10
            param_ratio = (args.ipc / ipc_base) * ((args.image_size / imsize_bases[args.dataset]) ** 2)
            args.lr_img *= param_ratio

        # Evaluation settings
        ## Setting augmentation
        if args.dsa:
            args.augment = False
            print(f"DSA strategy: {args.dsa_strategy}", )
        else:
            args.augment = True

    return args
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_models import resnet, resnet_ap, convnet
import tools as my
from data import load_data

my.init_logger()
logger = logging.getLogger(__name__)

def define_model(args):
    """Define neural network models"""
    if args.net_type == 'resnet':
        model = resnet.ResNet(
            args.dataset,
            args.depth,
            args.nclass,
            norm_type=args.norm_type,
            size=args.image_size,
            nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = resnet_ap.ResNetAP(
            args.dataset,
            args.depth,
            args.nclass,
            width=args.width,
            norm_type=args.norm_type,
            size=args.image_size,
            nch=args.nch)
    elif args.net_type == 'convnet':
        if args.dataset == "speech":
            args.depth = 4
        width = int(128 * args.width)
        model = convnet.ConvNet(
            args.nclass,
            net_norm=args.norm_type,
            net_depth=args.depth,
            net_width=width,
            channel=args.nch,
            im_size=(args.image_size, args.image_size))
    else:
        raise Exception(f"unknown network architecture: {args.net_type}")

    logger.info(f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}")
    return model

def set_train_para(args):
    """special settings for large imagenet"""
    if args.dataset == 'imagenet' and args.nclass > 100:
        # We need to tune lr and weight decay
        args.lr = 0.1
        args.weight_decay = 1e-4
        args.batch_size = max(128, args.batch_size)

def main(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"save directory: {args.save_dir}")
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    logger.info(f"dataset directory: {args.data_dir}")
    train_loader, val_loader, nclass = load_data(args)

    l_best, l_end = [], []
    set_train_para(args)
    for i in range(args.repeat):
        logger.info(f"repeat: {i+1}/{args.repeat}")
        records = ["acc_tr", "acc_val", "loss_tr", "loss_val"]
        plotter = my.NewPlotter(args.save_dir, records, n_epoch=args.epochs, idx=i)
        plotter.set_values(records, ylim=[[1, 100], [1, 100], [0, 3], [0, 3]])
        # plotter.set_values(records, ylim=[[1, 100]] * 2 + [[0, 3]] * 2)
        model = define_model(args)

        best_acc, last_acc = train(args, model, train_loader, val_loader, plotter)
        l_best.append(round(best_acc, 2))
        l_end.append(round(last_acc, 2))
        logger.info(f'best acc1: {l_best}, {np.mean(l_best):.1f} \u00B1 {np.std(l_best):.1f}')
        logger.info(f'end acc1: {l_end}, {np.mean(l_end):.1f} \u00B1 {np.std(l_end):.1f}')


def train(args, model, train_loader, val_loader, plotter=None):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
    milestones = [2 * args.epochs // 3, 5 * args.epochs // 6]
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.2)

    # Load pretrained
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
    if args.pretrained:
        pretrained = f"{args.save_dir}/checkpoint.pth.tar"
        cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)
        # TODO: optimizer scheduler steps

    model = model.cuda()
    logger.info(f"Start training with augmentation={args.augment}, mixup={args.mixup}")

    # Start training and validation
    for epoch in range(cur_epoch, args.epochs):
        is_best = False
        acc1_tr, acc5_tr, loss_tr = train_epoch(args, train_loader, model, criterion, optimizer, epoch, mixup=args.mixup)

        if epoch % args.val_freq == 0:
            acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch)
            plotter is not None and plotter.update(epoch=epoch, acc_tr=acc1_tr, loss_tr=loss_tr, acc_val=acc1, loss_val=loss_val)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                logger.info(f"(best) [{epoch}/{args.epochs}], Top1:{best_acc1:.1f}, Top5:{best_acc5:.1f}")

        if args.save_ckpt and (is_best or (epoch == args.epochs)):
            state = {
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(args.save_dir, state, is_best)
        scheduler.step()

        acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch)

    return best_acc1, acc1


def train_epoch(args, train_loader, model, criterion, optimizer,
                epoch=0, mixup='vanilla', n_target=None):
    meter = my.AverageMeter(["batch_time", "data_time", "losses", "top1", "top5"])
    end = time.time()
    n_data = 0

    model.train()
    for i, (x_tr, lbl_tr) in enumerate(train_loader):
        if train_loader.device == 'cpu':
            x_tr = x_tr.cuda()
            lbl_tr = lbl_tr.cuda()
        meter.update("data_time", time.time() - end)

        r = np.random.rand(1)
        if r < args.mix_p and mixup == 'cut':
            lam = np.random.beta(args.beta, args.beta)
            rand_index = my.random_indices(lbl_tr, nclass=args.nclass)

            target_b = lbl_tr[rand_index]
            bbx1, bby1, bbx2, bby2 = my.rand_bbox(x_tr.size(), lam)
            x_tr[:, :, bbx1:bbx2, bby1:bby2] = x_tr[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_tr.size()[-1] * x_tr.size()[-2]))

            output = model(x_tr)
            loss = criterion(output, lbl_tr) * ratio + criterion(output, target_b) * (1.0 - ratio)
        else:
            # compute output
            output = model(x_tr)
            loss = criterion(output, lbl_tr)

        # measure accuracy and record loss
        acc1, acc5 = my.accuracy(output.data, lbl_tr, topk=(1, 5))

        meter.update("losses", loss.item(), n=x_tr.size(0))
        meter.update("top1", acc1.item(), n=x_tr.size(0))
        meter.update("top5", acc5.item(), n=x_tr.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        meter.update("batch_time", time.time() - end)
        end = time.time()

        n_data += len(lbl_tr)
        if n_target and n_data >= n_target:
            break

    # noinspection PyUnresolvedReferences
    v0, v1, v2 = meter.top1["ave"], meter.top5["ave"], meter.losses["ave"]
    if epoch % args.print_freq == 0:
        logger.info(f"(Train) [{epoch}/{args.epochs}], Top1:{v0:.1f}, Top5:{v1:.1f}, Loss:{v2:.4f}")
    return v0, v1, v2


def validate(args, val_loader, model, criterion, epoch):
    meter = my.AverageMeter(["batch_time", "losses", "top1", "top5"])

    model.eval()

    end = time.time()
    for i, (x_val, lbl_val) in enumerate(val_loader):
        x_val = x_val.cuda()
        lbl_val = lbl_val.cuda()
        output = model(x_val)

        loss = criterion(output, lbl_val)

        # measure accuracy and record loss
        acc1, acc5 = my.accuracy(output.data, lbl_val, topk=(1, 5))

        meter.update("losses", loss.item(), n=x_val.size(0))
        meter.update("top1", acc1.item(), n=x_val.size(0))
        meter.update("top5", acc5.item(), n=x_val.size(0))

        # measure elapsed time
        meter.update("batch_time", time.time() - end)
        end = time.time()

    # noinspection PyUnresolvedReferences
    v0, v1, v2 = meter.top1["ave"], meter.top5["ave"], meter.losses["ave"]
    if epoch % args.print_freq == 0:
        logger.info(f"(Validation) [{epoch}/{args.epochs}], Top1:{v0:.1f}, Top5:{v1:.1f}, Loss:{v2:.4f}")
    return v0, v1, v2

def load_checkpoint(path, model, optimizer, use_parallel=False):
    if Path(path).is_file():
        logger.info(f"=> loading checkpoint {path}")
        checkpoint = torch.load(path)
        if use_parallel:
            checkpoint['state_dict'] = {key[7:]: value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> checkpoint loaded. (epoch: {cur_epoch}, best acc1: {best_acc1}%)")
    else:
        logger.info(f"=> no checkpoint found at {path}")
        cur_epoch = 0
        best_acc1 = 100

    return cur_epoch, best_acc1

def save_checkpoint(save_dir, state, is_best):
    dir_path = Path(save_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    if is_best:
        ckpt_path = dir_path.joinpath("model_best.pth.tar")
    else:
        ckpt_path = dir_path.joinpath("checkpoint.pth.tar")
    torch.save(state, ckpt_path)
    logger.info(f"checkpoint saved! ({ckpt_path})")


if __name__ == "__main__":
    print(__file__)
    print("Please use main.py to start")
    # main(args)

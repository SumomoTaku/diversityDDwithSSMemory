import random
import logging
from os.path import split

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.backward_compatibility import worker_init_fn
from torchvision.utils import save_image
import torch.nn.functional as func
import os
from pathlib import Path
import numpy as np
from misc import utils

# Values borrowed from https://github.com/VICO-UoE/DatasetCondensation/blob/master/utils.py
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {
    "cifar": [0.4914, 0.4822, 0.4465],
    "svhn": [0.4377, 0.4438, 0.4728],
    "imagenet": [0.485, 0.456, 0.406],
    "mnist": [0.1307],
    "fashion": [0.2861]
}
STDS = {
    "cifar": [0.2023, 0.1994, 0.2010],
    "svhn": [0.1980, 0.2010, 0.1970],
    "imagenet": [0.229, 0.224, 0.225],
    "mnist": [0.3081],
    "fashion": [0.3530]
}

logger = logging.getLogger(__name__)

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        self.images = images.detach().cpu().float()
        self.targets = labels.detach().cpu()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform is not None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]


class ImageFolder(datasets.DatasetFolder):
    """Dataset class for loading subsets with specified IPC.
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 load_memory=False,
                 load_transform=None,
                 nclass=100,
                 phase=0,
                 slct_type="random",
                 ipc=-1,
                 seed=-1,
                 select_list=None,
                 return_origin=False):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        assert Path(root).is_dir(), f"sub directory not found: \n{root}"
        super(ImageFolder, self).__init__(root,
                                          loader,
                                          self.extensions,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        # Override
        self.select_list = select_list
        self.return_origin = return_origin
        if nclass < 1000:
            self.classes, self.class_to_idx = self.find_subclasses(nclass=nclass,
                                                                   phase=phase,
                                                                   seed=seed)
        else:
            self.classes, self.class_to_idx = self.find_classes(self.root)
        self.original_labels = self.find_original_classes()
        self.nclass = nclass
        self.samples = datasets.folder.make_dataset(self.root, self.class_to_idx, self.extensions,
                                                    is_valid_file)

        if ipc > 0:
            self.samples = self._subset(slct_type=slct_type, ipc=ipc)

        self.targets = [s[1] for s in self.samples]
        self.original_targets = [self.original_labels[s[1]] for s in self.samples]
        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples

    def find_subclasses(self, nclass=100, phase=0, seed=0):
        """Finds the class folders in a dataset.
        """
        classes = []
        phase = max(0, phase)
        cls_from = nclass * phase
        cls_to = nclass * (phase + 1)
        if self.select_list:
            with open(self.select_list, 'r') as f:
                class_name = f.readlines()
            for c in class_name:
                c = c.split('\n')[0]
                classes.append(c)
            classes = classes[cls_from:cls_to]
        else:
            np.random.seed(seed)
            class_indices = np.random.permutation(len(self.classes))[cls_from:cls_to]
            for i in class_indices:
                classes.append(self.classes[i])

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == nclass, f"current:{len(classes)}, target: {nclass}"

        return classes, class_to_idx

    def find_original_classes(self):
        all_classes = sorted(os.listdir(self.root))
        original_labels = []
        for class_name in self.classes:
            original_labels.append(all_classes.index(class_name))
        return original_labels

    def _subset(self, slct_type='random', ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)

        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc <= min_class

        if slct_type == 'random':
            indices = np.arange(n)
        else:
            raise AssertionError(f'selection type does not exist!')

        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break

        return samples_subset

    def _load_images(self, transform=None):
        """Load images on memory
        """
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform is not None:
                sample = transform(sample)
            imgs.append(sample)
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end='\r')

        print(" " * 50, end='\r')
        return imgs

    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
        else:
            sample = self.imgs[index]

        target = self.targets[index]
        original_target = self.original_targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            original_target = self.target_transform(original_target)

        # Return original labels for DiT generation
        if self.return_origin:
            return sample, target, original_target
        else:
            return sample, target

def get_augment(dataset, rrc=False, rrc_size=-1):
    if dataset == "cifar":
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    elif dataset == "svhn":
        aug = [transforms.RandomCrop(32, padding=4)]
    elif dataset == "mnist" or dataset == "fashion":
        aug = [transforms.RandomCrop(28, padding=4)]
    elif dataset == "imagenet":
        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[
                                      [-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203],
                                  ])
        aug = [transforms.RandomHorizontalFlip(), jittering, lighting]

        if rrc and rrc_size > 0:
            rrc_fn = transforms.RandomResizedCrop(rrc_size, scale=(0.5, 1.0))
            aug = [rrc_fn] + aug
    else:
        raise Exception(f"no augment info of {dataset}")
    logger.info(f"Dataset with basic {dataset} augmentation")
    return aug

def get_resize(dataset, size):
    if dataset == "imagenet":
        if size > 0:
            resize_train = [transforms.Resize(size), transforms.CenterCrop(size)]
            resize_val = [transforms.Resize(size), transforms.CenterCrop(size)]
        else:
            resize_train = [transforms.RandomResizedCrop(224)]
            resize_val = [transforms.Resize(256), transforms.CenterCrop(224)]
        return resize_train, resize_val
    else:
        return [], []

def get_transform(args, from_tensor=False, normalize=True):
    dataset = args.dataset
    resize_train, resize_val = get_resize(dataset, args.image_size)
    cast = [] if from_tensor else [transforms.ToTensor()]
    rrc_size = args.image_size if args.rrc_size == -1 else args.rrc_size
    aug = [] if not args.augment else get_augment(dataset, rrc=args.rrc, rrc_size=rrc_size)
    norm = [] if not normalize else [transforms.Normalize(mean=MEANS[dataset], std=STDS[dataset])]

    trans_train = transforms.Compose(resize_train + cast + aug + norm)
    trans_val = transforms.Compose(resize_val + cast + norm)
    return trans_train, trans_val


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


class ClassBatchSampler(object):
    """Intra-class batch sampler 
    """
    def __init__(self, cls_idx, batch_size, drop_last=True):
        self.samplers = []
        for indices in cls_idx:
            n_ex = len(indices)
            sampler = torch.utils.data.SubsetRandomSampler(indices)
            batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                          batch_size=min(n_ex, batch_size),
                                                          drop_last=drop_last)
            self.samplers.append(iter(_RepeatSampler(batch_sampler)))

    def __iter__(self):
        while True:
            for sampler in self.samplers:
                yield next(sampler)

    def __len__(self):
        return len(self.samplers)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """Multi epochs data loader
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()  # Init iterator and sampler once

        self.convert = None
        if self.dataset[0][0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        if self.dataset[0][0].device == torch.device('cpu'):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for i in range(len(self)):
            data, target = next(self.iterator)
            if self.convert is not None:
                data = self.convert(data)
            yield data, target


class ClassDataLoader(MultiEpochsDataLoader):
    """Basic class loader (might be slow for processing data)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(self.dataset)):
            self.cls_idx[self.dataset.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(self.cls_idx, self.batch_size, drop_last=True)

        self.cls_targets = torch.tensor([np.ones(self.batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device='cuda')

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.dataset[i][0] for i in indices])
        target = torch.tensor([self.dataset.targets[i] for i in indices])
        return data.cuda(), target.cuda()

    def sample(self):
        data, target = next(self.iterator)
        if self.convert is not None:
            data = self.convert(data)

        return data.cuda(), target.cuda()


class ClassMemDataLoader():
    """Class loader with data on GPUs
    """
    def __init__(self, dataset, batch_size, drop_last=False, device='cuda'):
        self.device = device
        self.batch_size = batch_size

        self.dataset = dataset
        self.data = [d[0].to(device) for d in dataset]  # uint8 data
        self.targets = torch.tensor(dataset.targets, dtype=torch.long, device=device)

        sampler = torch.utils.data.SubsetRandomSampler([i for i in range(len(dataset))])
        self.batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                           batch_size=batch_size,
                                                           drop_last=drop_last)
        self.iterator = iter(_RepeatSampler(self.batch_sampler))

        self.nclass = dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(dataset)):
            self.cls_idx[self.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(self.cls_idx, self.batch_size, drop_last=True)
        self.cls_targets = torch.tensor([np.ones(batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device=self.device)

        self.convert = None
        if self.data[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.data[i] for i in indices])
        if self.convert is not None:
            data = self.convert(data)

        # print(self.targets[indices])
        return data, self.cls_targets[c]

    def sample(self):
        indices = next(self.iterator)
        data = torch.stack([self.data[i] for i in indices])
        if self.convert is not None:
            data = self.convert(data)
        target = self.targets[indices]

        return data, target

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            data, target = self.sample()
            yield data, target


class ClassPartMemDataLoader(MultiEpochsDataLoader):
    """Class loader for ImageNet-100 with multi-processing.
       This loader loads target subclass samples on GPUs
       while can loading full training data from storage. 
    """
    def __init__(self, subclass_list, real_to_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.mem_cls = subclass_list
        self.real_to_idx = real_to_idx

        self.cls_idx = [[] for _ in range(self.nclass)]
        idx = 0
        self.data_mem = []
        print("Load target class data on memory..")
        for i in range(len(self.dataset)):
            c = self.dataset.targets[i]
            if c in self.mem_cls:
                self.data_mem.append(self.dataset[i][0].cuda())
                self.cls_idx[c].append(idx)
                idx += 1

        if self.data_mem[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)
        print(f"Subclass: {subclass_list}, {len(self.data_mem)}")

        class_batch_size = 64
        self.class_sampler = ClassBatchSampler([self.cls_idx[c] for c in subclass_list],
                                               class_batch_size,
                                               drop_last=True)
        self.cls_targets = torch.tensor([np.ones(class_batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device='cuda')

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            idx = self.real_to_idx[c]
            indices = next(self.class_sampler.samplers[idx])

        data = torch.stack([self.data_mem[i] for i in indices])
        if self.convert is not None:
            data = self.convert(data)

        # print([self.dataset.targets[i] for i in self.slct[indices]])
        return data, self.cls_targets[c]

    def sample(self):
        data, target = next(self.iterator)
        if self.convert is not None:
            data = self.convert(data)

        return data.cuda(), target.cuda()

def load_data(args):
    """Load training and validation data
    """
    # based on the habit on imagenet, here uses "validation" instead of "test"
    trans_train, trans_val = get_transform(args)
    nclass = 10
    if args.dataset == "cifar":
        if args.spec_dataset == "cifar100":
            train_dataset = datasets.CIFAR100(args.data_dir, train=True, transform=trans_train)
            val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=trans_val)
            nclass = 100
        elif args.spec_dataset == "cifar10":
            train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=trans_train)
            val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=trans_val)
        else:
            raise Exception(f"unknown cifar subset {args.spec_dataset}")
    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(args.data_dir, split='train', download=False, transform=trans_train)
        val_dataset = datasets.SVHN(args.data_dir, split='test', download=False, transform=trans_val)
    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir, train=True, transform=trans_train)
        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=trans_val)
    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=trans_train)
        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=trans_val)
    elif args.dataset == "imagenet":
        nclass = args.nclass
        if nclass > 100 and args.load_memory:
            args.load_memory = False
            logger.info(f"load_memory is set as False for large dataset ({nclass} classes)")
        trans_train, trans_val = get_transform(args)
        train_dataset = ImageFolder(args.train_dir,
                                    transform=trans_train,
                                    nclass=args.nclass,
                                    seed=args.dseed,
                                    slct_type=args.slct_type,
                                    ipc=args.ipc,
                                    load_memory=args.load_memory,
                                    select_list=args.select_list)
        val_dataset = ImageFolder(args.val_dir,
                                  transform=trans_val,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory,
                                  select_list=args.select_list)

        assert nclass == args.nclass, f"mismatch of class between specified ({args.nclass}) and actual ({nclass})"
        for trn, val in zip(train_dataset.classes, val_dataset.classes):
            assert trn == val, f"mismatch of name between train ({args.nclass}) and validation ({nclass})"

        logger.info(
            "Subclass is extracted: \n"
            f"n class:{nclass}, n train: {len(train_dataset.targets)}, n validation: {len(val_dataset.targets)}"
        )
        if args.ipc > 0:
            logger.info(f"subsample (type = {args.slct_type}, ipc = {args.ipc})")
    else:
        raise Exception(f"unknown dataset: {args.dataset}")

    def seed_worker(worker_id):
        worker_seed = 42
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = MultiEpochsDataLoader(train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.workers,
                                         pin_memory=True,
                                         worker_init_fn=seed_worker)
    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.workers,
                                       pin_memory=True,
                                       worker_init_fn=seed_worker)

    return train_loader, val_loader, nclass

# part of save_img (not used)
def img_denormlaize(img, dataset='imagenet'):
    """
    Scaling and shift a batch of images (NCHW)
    """
    nch = img.shape[1]
    mean = torch.tensor(MEANS[dataset], device=img.device).reshape(1, nch, 1, 1)
    std = torch.tensor(STDS[dataset], device=img.device).reshape(1, nch, 1, 1)
    return img * std + mean

# not used
def save_img(save_dir, img, unnormalize=True, max_num=200, size=64, nrow=10, dataset='imagenet'):
    img = img[:max_num].detach()
    if unnormalize:
        img = img_denormlaize(img, dataset=dataset)
    img = torch.clamp(img, min=0., max=1.)

    if img.shape[-1] > size:
        img = func.interpolate(img, size)
    # torchvision.utils
    save_image(img.cpu(), save_dir, nrow=nrow)

def main():
    print(__path__)

if __name__ == "__main__":
    main()

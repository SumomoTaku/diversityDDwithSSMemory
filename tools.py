import logging
import time
import torch
import numpy as np
from pathlib import Path
import copy
import matplotlib.pyplot as plt

def init_logger(save_path=None):
    if save_path is None:
        save_path = f"{time.strftime("%y%m%d")}.log"
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        handlers=[console_handler, file_handler],
        level=logging.DEBUG,
    )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, attrs):
        init_value = {"value": 0, "sum": 0, "count": 0, "ave": 0}
        for attr in attrs:
            setattr(self, attr, copy.deepcopy(init_value))

    def update(self, name, value, n=1):
        attr = getattr(self, name)
        attr["value"] = value
        attr["sum"] += value * n
        attr["count"] += n
        attr["ave"] = attr["sum"] / attr["count"]
        # setattr(self, name, attr)

class Plotter:
    def __init__(self, path, n_epoch, idx=0, plot_freq=None):
        self.path = path
        init_dict = {"epoch_tr": [], "epoch_val":[], "acc_tr": [], "acc_val": [], "loss_tr": [], "loss_val": []}
        self.data = copy.deepcopy(init_dict)
        self.n_epoch = n_epoch
        self.plot_freq = plot_freq if plot_freq else n_epoch // 25
        self.idx = idx

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.data[key].append(value)

        if len(self.data["epoch_tr"]) % self.plot_freq == 0:
            self.plot()

    def plot(self, color="black"):
        fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 3))
        fig.tight_layout(h_pad=3, w_pad=3)

        fig.suptitle(f"{self.path}", size=16, y=1.1)

        axes[0].plot(self.data["epoch_tr"], self.data["acc_tr"], color, lw=0.8)
        axes[0].set_xlim([0, self.n_epoch])
        axes[0].set_ylim([0, 100])
        axes[0].set_title("acc train")

        axes[1].plot(self.data["epoch_val"], self.data["acc_val"], color, lw=0.8)
        axes[1].set_xlim([0, self.n_epoch])
        axes[1].set_ylim([0, 100])
        axes[1].set_title("acc val")

        axes[2].plot(self.data["epoch_tr"], self.data["loss_tr"], color, lw=0.8)
        axes[2].set_xlim([0, self.n_epoch])
        axes[2].set_ylim([0, 3])
        axes[2].set_title("loss train")

        axes[3].plot(self.data["epoch_val"], self.data["loss_val"], color, lw=0.8)
        axes[3].set_xlim([0, self.n_epoch])
        axes[3].set_ylim([0, 3])
        axes[3].set_title("loss val")

        for ax in axes:
            ax.set_xlabel("epochs")

        plt.savefig(f"{self.path}/curve_{self.idx}.png", bbox_inches="tight")
        plt.close()

class NewPlotter:
    def __init__(self, save_dir, attr_names, n_epoch=-1, idx=0, plot_freq=None):
        self.save_dir = save_dir
        self.save_path = f"{save_dir}/curve{idx}_{time.strftime("%y%m%d%H%M%S")}.png"
        self.names = attr_names
        init_value = {"value": []}
        for attr in attr_names:
            setattr(self, attr, copy.deepcopy(init_value))
        # init_dict = {"epoch_tr": [], "epoch_val":[], "acc_tr": [], "acc_val": [], "loss_tr": [], "loss_val": []}
        # self.data = init_dict.deepcopy()
        # lims = [[0, 1] for _ in range(len(attr_names))]
        # self.ylims = {k: v for k,v in zip(attr_names, lims)}
        # keep format with other attrs
        self.epoch = {"value":[]}
        self.n_epoch = n_epoch
        self.plot_freq = plot_freq if plot_freq else n_epoch // 25
        self.idx = idx

    # set_value("acc_tr", ylim=[0, 10], title="training accuracy")
    def set_values(self, attrs, **kwargs):
        for key, values in kwargs.items():
            if not isinstance(values, list):
                values = [values] * len(attrs)
            for attr, value in zip (attrs, values):
                getattr(self, attr)[key] = value

    def set_value(self, attr, **kwargs):
        for key, value in kwargs.items():
            getattr(self, attr)[key] = value

    def update(self, **kwargs):
        for key, value in kwargs.items():
            getattr(self, key)["value"].append(value)
        self.plot()

        # if len(self.epoch) % self.plot_freq == 0:
        #     self.plot()

    def plot(self, color="black"):
        n = len(self.names)
        fig, axes = plt.subplots(1, n, figsize=(n * n, 3))
        fig.tight_layout(h_pad=3, w_pad=3)

        fig.suptitle(f"{self.save_dir}", size=16, y=1.1)
        for i, name in enumerate(self.names):
            attr = getattr(self, name)
            axes[i].plot(self.epoch["value"], attr["value"], color, lw=0.8)
            axes[i].set_xlim([0, self.n_epoch])
            ylim = attr["ylim"] if "ylim" in attr.keys() else [0, 100]
            axes[i].set_ylim(ylim)
            title = attr["title"] if "title" in attr.keys() else f"{name}"
            axes[i].set_title(title)
            axes[i].set_xlabel("epochs")

        # axes[0].plot(self.data["epoch_tr"], self.data["acc_tr"], color, lw=0.8)
        # axes[0].set_xlim([0, self.n_epoch])
        # axes[0].set_ylim([0, 100])
        # axes[0].set_title("acc train")
        #
        # axes[1].plot(self.data["epoch_val"], self.data["acc_val"], color, lw=0.8)
        # axes[1].set_xlim([0, self.n_epoch])
        # axes[1].set_ylim([0, 100])
        # axes[1].set_title("acc val")
        #
        # axes[2].plot(self.data["epoch_tr"], self.data["loss_tr"], color, lw=0.8)
        # axes[2].set_xlim([0, self.n_epoch])
        # axes[2].set_ylim([0, 3])
        # axes[2].set_title("loss train")
        #
        # axes[3].plot(self.data["epoch_val"], self.data["loss_val"], color, lw=0.8)
        # axes[3].set_xlim([0, self.n_epoch])
        # axes[3].set_ylim([0, 3])
        # axes[3].set_title("loss val")

        for ax in axes:
            ax.set_xlabel("epochs")

        plt.savefig(self.save_path, bbox_inches="tight")
        plt.close()

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels.to("cpu")]

def random_indices(y, nclass=10, intraclass=False, device="cuda"):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cosine_similarity(ta, tb):
    bs1, bs2 = ta.shape[0], tb.shape[0]
    frac_up = torch.matmul(ta, tb.T)
    frac_down = torch.norm(ta, dim=-1).view(bs1, 1).repeat(1, bs2) * \
                torch.norm(tb, dim=-1).view(1, bs2).repeat(bs1, 1)
    return frac_up / frac_down

def find_ok_path(path):
    path = Path(path).resolve()
    stem = path.stem
    parent = path.parent
    ok_path = path
    num_suf = 0
    while ok_path.exists():
        ok_path = parent.joinpath(f"{stem}_{num_suf}")
        num_suf += 1
    return ok_path
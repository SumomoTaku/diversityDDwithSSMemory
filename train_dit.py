"""
Fine-tuning DiT with minimax criteria.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from collections import OrderedDict, defaultdict
from PIL import Image
from copy import deepcopy
import logging
from pathlib import Path
import tools as my

from data import ImageFolder
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def mark_difffit_trainable(model, is_bitfit=False):
    """
    Mark the parameters that require updating by difffit.
    """
    if is_bitfit:
        trainable_names = ["bias"]
    else:
        trainable_names = ["bias", "norm", "gamma", "y_embed"]

    for par_name, par_tensor in model.named_parameters():
        par_tensor.requires_grad = any([kw in par_name for kw in trainable_names])
    return model

def update_memory(memory, cls, strategy="max"):
    elements = torch.cat(memory[cls]).flatten(start_dim=1)
    distance = my.cosine_similarity(elements, elements)
    similarity_degree = list(distance.sum(1))

    if strategy == "max":
        value = max(similarity_degree)
    elif strategy == "min":
        value = min(similarity_degree)
    else:
        raise Exception(f"unknown memory update strategy: {strategy}")
    idx = similarity_degree.index(value)
    memory[cls].pop(idx)


#################################################################################
#                                  Training Loop                                #
#################################################################################

logger = logging.getLogger(__name__)
def main(args):
    """
    Fine-tune a DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    wrd_size = dist.get_world_size()
    assert args.batch_size % wrd_size == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * wrd_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={wrd_size}.")

    # Setup an experiment folder:
    ckpt_dir = Path(args.save_dir).joinpath("ckpt")
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.propagate = False

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.dit_model](input_size=latent_size,num_classes=args.n_model_class)
    state_dict = find_model(args.dit_ckpt)
    model.load_state_dict(state_dict, strict=False)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    for p in ema.parameters():
        p.requires_grad = False
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    logger.info(f"DiT parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-3 in the Difffit paper):
    model = mark_difffit_trainable(model)
    model = DDP(model.to(device), device_ids=[rank])
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params_to_optimize)
    logger.info(f"number of trainable parameters: {total_params:,}")
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.train_dir, transform=transform, nclass=args.nclass,
                          ipc=args.finetune_ipc, select_list=args.select_list, phase=args.phase,
                          seed=0, return_origin=True)
    sampler = DistributedSampler(dataset, num_replicas=wrd_size, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.train_dir})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    # train()使模型进入训练模型，从而启用dropout等设置
    model.train()
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    running_loss, running_loss_pos, running_loss_neg = 0, 0, 0
    # real samples for Representativeness, generative samples for Diversity
    real_memory = defaultdict(list)
    gen_memory = defaultdict(list)

    train_steps = 0
    log_steps = 0
    logger.info(f"training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        class_point = defaultdict(int)
        sampler.set_epoch(epoch)
        for i, (x, lbl, lbl_ori) in enumerate(loader):
            lbl = lbl.numpy()
            x = x.to(device)
            lbl_ori = lbl_ori.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=lbl_ori)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            output_latent = loss_dict["output"].flatten(start_dim=1)
            loss = loss_dict["loss"].mean()
            running_loss += loss.item()

            if args.distill:
                real_loss = torch.tensor(0.).to(device)
                gen_loss = torch.tensor(0.).to(device)
                # Calculate minimax criteria
                lbl_set = set(lbl)
                n_lbl = len(lbl_set)
                for cls in lbl_set:
                    if len(gen_memory[cls]):
                        real_latent = torch.cat(real_memory[cls]).flatten(start_dim=1)
                        gen_latent = torch.cat(gen_memory[cls]).flatten(start_dim=1)

                        # Representativeness
                        real_similarity = 1 - my.cosine_similarity(output_latent[lbl == cls], real_latent).min()
                        real_loss += real_similarity * args.lambda_real / n_lbl
                        # Diversity
                        gen_similarity = my.cosine_similarity(output_latent[lbl == cls], gen_latent).max()
                        gen_loss += gen_similarity * args.lambda_gen / n_lbl

                        running_loss_pos += real_loss.item()
                        running_loss_neg += gen_loss.item()

                    # add used images to memory
                    real_latent = x.flatten(start_dim=1)
                    cls_idx = np.array(lbl == cls)
                    real_memory[cls].extend(real_latent[cls_idx].detach().split(1))
                    gen_memory[cls].extend(output_latent[cls_idx].detach().split(1))

                    while len(real_memory[cls]) > args.memory_size:
                        update_memory(real_memory, cls, strategy="max")
                    while len(gen_memory[cls]) > args.memory_size:
                        update_memory(gen_memory, cls, strategy="max")
                    loss = loss + real_loss + gen_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema, model.module)

            # Log loss values:
            # TODO: use plotter or meter to record. delete dist maybe
            log_steps += 1
            train_steps += 1
            if train_steps % args.print_freq == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss_pos = torch.tensor(running_loss_pos / log_steps, device=device)
                avg_loss_neg = torch.tensor(running_loss_neg / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_pos, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_neg, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_loss_pos = avg_loss_pos.item() / dist.get_world_size()
                avg_loss_neg = avg_loss_neg.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} {avg_loss_pos:.4f} {avg_loss_neg:.4f}")
                # Reset monitoring variables:
                running_loss = 0
                running_loss_pos = 0
                running_loss_neg = 0
                log_steps = 0


            # Save DiT checkpoint:
            if train_steps % args.ckpt_freq == 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args
                    }
                    ckpt_path = ckpt_dir.joinpath(f"{i:07d}.pt")
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"checkpoint saved to {ckpt_path}")
                dist.barrier()

    logger.info("training done!")
    # cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    print(__file__)
    print("Please use main.py to start")
    # main(args)

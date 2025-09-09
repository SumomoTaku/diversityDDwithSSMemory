"""
Sample new images from a pre-trained DiT.
"""
from pathlib import Path
import time
import torch
import logging
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import tools as my

logger = logging.getLogger(__name__)

def fetch_class_list(all_path, sel_path):
    with open(all_path, "r") as f:
        all_classes = f.readlines()
    all_classes = [text.strip() for text in all_classes]
    with open(sel_path, "r") as f:
        sel_cls = f.readlines()
    sel_cls = [text.strip() for text in sel_cls]
    cls_lbl = [all_classes.index(x) for x in sel_cls]
    return sel_cls, cls_lbl

def main(args):
    # Setup PyTorch:
    seed = int(time.time()) if args.seed == -1 else args.seed
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels to condition the model
    sel_cls, sel_lbl = fetch_class_list(args.all_cls, args.select_list)
    logger.info(f"{len(sel_cls)} classes selected.({args.select_list})")
    if args.dit_ckpt is None:
        assert args.dit_model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.n_model_class == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.dit_model](input_size=latent_size, num_classes=args.n_model_class).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.dit_ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    diffusion = create_diffusion(str(args.num_denoising))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    batch_size = 1
    save_dir = Path(args.save_dir).joinpath("sample")
    for cls, lbl in zip(sel_cls, sel_lbl):
        save_path = save_dir.joinpath(cls)
        save_path.mkdir(parents=True, exist_ok=True)
        for shift in tqdm(range(args.ipc // batch_size)):
        # for shift in range(args.ipc // batch_size):
            # Create sampling noise:
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor([lbl], device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=4.0)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            for image_index, image in enumerate(samples):
                index = image_index + shift * batch_size + args.total_shift
                save_image(image, save_path.joinpath(f"{index}.png"), normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    print(__file__)
    print("Please use main.py to start")
    # main()

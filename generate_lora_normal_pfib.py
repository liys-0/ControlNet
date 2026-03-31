import argparse
import os
import cv2
import numpy as np
import torch
import einops
import random

from pytorch_lightning import seed_everything
from share import *
import config

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from lora import inject_trainable_lora, load_lora_up_down


def generate_defect(args):
    apply_canny = CannyDetector()

    print("Loading model...")
    model = create_model("./models/cldm_v15.yaml").cpu()

    if os.path.exists(args.controlnet_path):
        print(f"Loading ControlNet from {args.controlnet_path}")
        model.load_state_dict(load_state_dict(args.controlnet_path, location="cuda"))
    else:
        print(
            f"ControlNet {args.controlnet_path} not found. Attempting default model..."
        )
        model.load_state_dict(
            load_state_dict("./models/control_sd15_canny.pth", location="cuda")
        )

    if os.path.exists(args.lora_path):
        print(f"Injecting LoRA and loading weights from {args.lora_path}")
        inject_trainable_lora(model.model.diffusion_model, rank=4)
        lora_dict = torch.load(args.lora_path, map_location="cuda")
        model.model.diffusion_model = load_lora_up_down(
            model.model.diffusion_model, lora_dict
        )
    else:
        print(
            f"Warning: LoRA weights {args.lora_path} not found. Generating without LoRA."
        )

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    print(f"Processing image {args.input_image}")
    input_image = cv2.imread(args.input_image)
    if input_image is None:
        raise ValueError(f"Could not read {args.input_image}")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, args.image_resolution)
        H, W, C = img.shape

        if args.raw_control_image:
            control_img = cv2.imread(args.raw_control_image)
            control_img = cv2.cvtColor(control_img, cv2.COLOR_BGR2RGB)
            detected_map = resize_image(HWC3(control_img), args.image_resolution)
        else:
            detected_map = apply_canny(img, args.low_threshold, args.high_threshold)
            detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(args.num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        seed = args.seed
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning(
                    [args.prompt + ", " + args.a_prompt] * args.num_samples
                )
            ],
        }
        un_cond = {
            "c_concat": None if args.guess_mode else [control],
            "c_crossattn": [
                model.get_learned_conditioning([args.n_prompt] * args.num_samples)
            ],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [args.strength * (0.825 ** float(12 - i)) for i in range(13)]
            if args.guess_mode
            else ([args.strength] * 13)
        )
        samples, intermediates = ddim_sampler.sample(
            args.ddim_steps,
            args.num_samples,
            shape,
            cond,
            verbose=False,
            eta=args.eta,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        for i in range(args.num_samples):
            out_name = (
                args.output_image.replace(".png", f"_{i}.png")
                if args.num_samples > 1
                else args.output_image
            )
            cv2.imwrite(out_name, cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR))
            print(f"Saved generated image to {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate defect images using ControlNet + LoRA"
    )
    parser.add_argument(
        "--input_image", type=str, required=True, help="Path to input normal pfib image"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="output_defect.png",
        help="Path to save output image",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default="./models/control_sd15_normal_pfib.ckpt",
        help="Path to trained ControlNet model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./models/lora_defect/lora_epoch_19.ckpt",
        help="Path to trained LoRA model",
    )
    parser.add_argument(
        "--prompt", type=str, default="defect pfib", help="Prompt to generate"
    )
    parser.add_argument(
        "--a_prompt",
        type=str,
        default="best quality, extremely detailed",
        help="Additional prompt",
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        help="Negative prompt",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "--image_resolution", type=int, default=512, help="Output image resolution"
    )
    parser.add_argument(
        "--low_threshold",
        type=int,
        default=100,
        help="Canny low threshold",
    )
    parser.add_argument(
        "--high_threshold",
        type=int,
        default=200,
        help="Canny high threshold",
    )
    parser.add_argument(
        "--raw_control_image",
        type=str,
        default="",
        help="Pass a raw control image (e.g. pre-computed edges/normals) instead of computing Canny",
    )
    parser.add_argument(
        "--strength", type=float, default=1.0, help="ControlNet strength"
    )
    parser.add_argument(
        "--scale", type=float, default=9.0, help="Classifier free guidance scale"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=20, help="DDIM sampling steps"
    )
    parser.add_argument("--guess_mode", action="store_true", help="Enable guess mode")
    parser.add_argument(
        "--seed", type=int, default=-1, help="Random seed (-1 for random)"
    )
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta")

    args = parser.parse_args()
    generate_defect(args)

import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import argparse
import os


def generate_images_from_folder_4ch(
    input_folder,
    mask_folder,
    output_folder,
    checkpoint_path,
    prompt,
    a_prompt="best quality, extremely detailed",
    n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
    num_samples=1,
    image_resolution=256,
    ddim_steps=20,
    guess_mode=False,
    strength=1.0,
    scale=9.0,
    seed=-1,
    eta=0.0,
):
    print(f"Loading model architecture from ./models/cldm_v15_4ch.yaml")
    model = create_model("./models/cldm_v15_4ch.yaml").cpu()

    print(f"Loading trained weights from {checkpoint_path}")
    model.load_state_dict(load_state_dict(checkpoint_path, location="cuda"), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images in {input_folder}. Starting processing...")

    for img_file in image_files:
        input_image_path = os.path.join(input_folder, img_file)
        base_name = os.path.splitext(img_file)[0]

        print(f"\nProcessing {img_file}...")
        input_image = cv2.imread(input_image_path)
        if input_image is None:
            print(f"Could not read {input_image_path}, skipping.")
            continue

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        mask_name = img_file
        if mask_name.startswith("defect_"):
            mask_name = mask_name[len("defect_"):]
        elif mask_name.startswith("normal_"):
            mask_name = mask_name[len("normal_"):]
        
        mask_path = os.path.join(mask_folder, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Could not read mask at {mask_path}, skipping.")
            continue

        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape
            
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            source = img.astype(np.float32) / 255.0
            mask_float = mask_resized.astype(np.float32) / 255.0
            mask_expanded = np.expand_dims(mask_float, axis=-1)

            hint_4ch = np.concatenate([source, mask_expanded], axis=-1)

            control = torch.from_numpy(hint_4ch.copy()).float().cuda()
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, "b h w c -> b c h w").clone()

            import random
            current_seed = random.randint(0, 65535) if seed == -1 else seed
            seed_everything(current_seed)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {
                "c_concat": [control],
                "c_crossattn": [
                    model.get_learned_conditioning(
                        [prompt + ", " + a_prompt] * num_samples
                    )
                ],
            }
            un_cond = {
                "c_concat": None if guess_mode else [control],
                "c_crossattn": [
                    model.get_learned_conditioning([n_prompt] * num_samples)
                ],
            }
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = (
                [strength * (0.825 ** float(12 - i)) for i in range(13)]
                if guess_mode
                else ([strength] * 13)
            )

            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
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

            results = [x_samples[i] for i in range(num_samples)]

            for i, result in enumerate(results):
                suffix = f"_sample{i}" if num_samples > 1 else "_generated"
                out_path = os.path.join(output_folder, f"{base_name}{suffix}.png")
                cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    import config
    import random

    parser = argparse.ArgumentParser(
        description="Test trained 4-channel ControlNet offline on a folder of images"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to input folder containing images (source)",
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        required=True,
        help="Path to mask folder containing masks",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="test_output_4ch",
        help="Path to save generated images",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained 4ch .ckpt file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="normal pfib image",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of images to generate per input",
    )

    args = parser.parse_args()

    generate_images_from_folder_4ch(
        input_folder=args.input_folder,
        mask_folder=args.mask_folder,
        output_folder=args.output_folder,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        num_samples=args.num_samples,
    )

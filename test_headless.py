import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import argparse
import os
import glob


def generate_images_from_folder(
    input_folder,
    output_folder,
    checkpoint_path,
    prompt,
    a_prompt="best quality, extremely detailed",
    n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
    num_samples=1,
    image_resolution=512,
    ddim_steps=20,
    guess_mode=False,
    strength=1.0,
    scale=9.0,
    seed=-1,
    eta=0.0,
    low_threshold=100,
    high_threshold=200,
    raw_control_image=False,
):
    print(f"Loading model architecture from ./models/cldm_v15.yaml")
    model = create_model("./models/cldm_v15.yaml").cpu()

    print(f"Loading trained weights from {checkpoint_path}")
    model.load_state_dict(
        load_state_dict(checkpoint_path, location="cuda"), strict=False
    )
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    apply_canny = CannyDetector()

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

        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            if raw_control_image:
                # Bypass Canny, use the image exactly as it is (for pre-computed edge maps)
                detected_map = img.copy()
                print("Using raw image as control (bypassing Canny detector)...")
            else:
                # Extract Canny edges
                detected_map = apply_canny(img, low_threshold, high_threshold)
                detected_map = HWC3(detected_map)

            edge_output_path = os.path.join(output_folder, f"{base_name}_edges.png")
            cv2.imwrite(edge_output_path, detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, "b h w c -> b c h w").clone()

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
        description="Test trained ControlNet offline on a folder of images"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to input folder containing images",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="test_output",
        help="Path to save generated images",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained .ckpt file"
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

    # We need to pass args to the function to read raw_control_image flag
    global _raw_control_image_flag
    _raw_control_image_flag = args.raw_control_image

    generate_images_from_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        num_samples=args.num_samples,
        raw_control_image=args.raw_control_image,
    )

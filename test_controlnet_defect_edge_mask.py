import os
import cv2
import json
import einops
import numpy as np
import torch
import argparse
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained checkpoint (.ckpt)")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to base model (e.g. control_v11p_sd15_canny.pth)")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing source images or prompt.json")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory containing mask images (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--model_config", type=str, default="./models/cldm_v15.yaml", help="Model config file")
    parser.add_argument("--image_resolution", type=int, default=512, help="Resolution to resize images to")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per image")
    parser.add_argument("--ddim_steps", type=int, default=20, help="DDIM sampling steps")
    parser.add_argument("--guess_mode", action="store_true", help="Enable guess mode")
    parser.add_argument("--strength", type=float, default=1.0, help="Control strength")
    parser.add_argument("--control_end", type=float, default=1.0, help="Percentage of steps to apply ControlNet (0.0 to 1.0)")
    parser.add_argument("--erase_edge_in_mask", action="store_true", help="Erase the edge map inside the masked region (Solution 1)")
    parser.add_argument("--scale", type=float, default=9.0, help="Unconditional guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--a_prompt", type=str, default="best quality, extremely detailed", help="Added prompt")
    parser.add_argument("--n_prompt", type=str, default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality", help="Negative prompt")
    parser.add_argument("--fp16", action="store_true",
                        help="Run sampling under torch.autocast(fp16) for ~2x speedup on V100/A100. "
                             "Weights stay fp32; only ops are cast.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    print(f"Loading model architecture from {args.model_config}")
    model = create_model(args.model_config).cpu()

    if args.resume_path and os.path.exists(args.resume_path):
        print(f"Loading base model from {args.resume_path}")
        model.load_state_dict(load_state_dict(args.resume_path, location="cpu"), strict=False)

    print(f"Loading trained weights from {args.model_path}")
    state_dict = load_state_dict(args.model_path, location="cpu")
    
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    prompt_file = os.path.join(args.test_dir, "prompt.json")
    test_cases = []
    
    if os.path.exists(prompt_file):
        print(f"Found prompt.json in {args.test_dir}")
        with open(prompt_file, 'rt') as f:
            for line in f:
                item = json.loads(line)
                source_filename = item['source']
                target_filename = item.get('target', '')
                prompt = item['prompt']
                test_cases.append({
                    "image_path": os.path.join(args.test_dir, source_filename),
                    "target_filename": target_filename,
                    "prompt": prompt,
                    "filename": os.path.basename(source_filename)
                })
    else:
        print(f"No prompt.json found. Reading all images from {args.test_dir} directly.")
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        for f in os.listdir(args.test_dir):
            if f.lower().endswith(valid_extensions):
                test_cases.append({
                    "image_path": os.path.join(args.test_dir, f),
                    "target_filename": f,
                    "prompt": "defect",
                    "filename": f
                })

    if not test_cases:
        print("No test images found. Exiting.")
        return

    print(f"Testing {len(test_cases)} images...")

    for i, case in enumerate(test_cases):
        image_path = case["image_path"]
        base_prompt = case["prompt"]
        filename = case["filename"]
        target_filename = case.get("target_filename", "")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping.")
            continue

        print(f"[{i+1}/{len(test_cases)}] Processing {filename} with prompt: '{base_prompt}'")
        
        control_img = cv2.imread(image_path)
        if control_img is None:
            continue
            
        control_img = cv2.cvtColor(control_img, cv2.COLOR_BGR2RGB)
        control_img = HWC3(control_img)
        control_img = resize_image(control_img, args.image_resolution)

        H, W, C = control_img.shape

        mask = np.zeros((H, W, 1), dtype=np.uint8)
        mask_path = None
        if args.mask_dir is not None and target_filename:
            mask_name = os.path.basename(target_filename)
            if mask_name.startswith("defect_"):
                mask_name = mask_name[len("defect_"):]
            elif mask_name.startswith("normal_"):
                mask_name = mask_name[len("normal_"):]
            mask_path = os.path.join(args.mask_dir, mask_name)
        else:
            mask_path = os.path.join(os.path.dirname(image_path), "mask.png")
            
        if mask_path is not None and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)[..., np.newaxis]

        if args.erase_edge_in_mask and mask_path is not None and os.path.exists(mask_path):
            control_img[mask[..., 0] > 127] = 0

        control_img = control_img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 64.0

        control_img_4ch = np.concatenate([control_img, mask], axis=-1)

        control = torch.from_numpy(control_img_4ch.copy()).cuda()
        control = torch.stack([control for _ in range(args.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        full_prompt = base_prompt + ', ' + args.a_prompt
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([full_prompt] * args.num_samples)]}
        un_cond = {"c_concat": None if args.guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([args.n_prompt] * args.num_samples)]}
        shape = (4, args.image_resolution // 8, args.image_resolution // 8)

        model.control_scales = [args.strength * (0.825 ** float(12 - i)) for i in range(13)] if args.guess_mode else ([args.strength] * 13)

        def step_callback(step_idx):
            if step_idx >= int(args.ddim_steps * args.control_end):
                model.control_scales = [0.0] * 13

        samples, _ = ddim_sampler.sample(
            args.ddim_steps,
            args.num_samples,
            shape,
            cond,
            verbose=False,
            eta=args.eta,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=un_cond,
            callback=step_callback
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        base_name, ext = os.path.splitext(filename)
        for j in range(args.num_samples):
            res_img = x_samples[j]
            res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
            
            out_name = f"{base_name}_sample{j}{ext}" if args.num_samples > 1 else f"{base_name}_gen{ext}"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, res_img)

if __name__ == "__main__":
    main()
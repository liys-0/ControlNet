"""
Bundle Stable Diffusion 1.5 weights with the slim ControlNet v1.1 canny weights
into a single checkpoint that can be passed directly to --resume_path.

ControlNet v1.1 checkpoints (e.g. control_v11p_sd15_canny.pth) only contain the
ControlNet branch (`control_model.*`). Loading them alone leaves the SD UNet,
VAE, and CLIP text encoder randomly initialized. This tool merges:

    v1-5-pruned.ckpt              ->  model.diffusion_model.* / first_stage_model.* / cond_stage_model.*
    control_v11p_sd15_canny.pth   ->  control_model.*

The 3-channel canny `input_hint_block.0.weight` is also padded to 4 channels
(zero-init for the 4th/mask channel) so the result is directly compatible with
the 4ch hint config used by train_controlnet_defect_edge_mask.py.

Usage:
    python tool_bundle_canny_v11.py \\
        --sd_path ./models/v1-5-pruned.ckpt \\
        --canny_path ./models/control_v11p_sd15_canny.pth \\
        --output ./models/control_sd15_canny_bundled_4ch.pth
"""

import argparse
import os

import torch

from share import *
from cldm.model import create_model, load_state_dict


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ""
    if name[: len(parent_name)] != parent_name:
        return False, ""
    return True, name[len(parent_name) :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd_path", default="./models/v1-5-pruned.ckpt")
    parser.add_argument("--canny_path", default="./models/control_v11p_sd15_canny.pth")
    parser.add_argument("--output", default="./models/control_sd15_canny_bundled_4ch.pth")
    parser.add_argument("--config", default="./models/cldm_v15.yaml")
    args = parser.parse_args()

    assert os.path.exists(args.sd_path), f"SD checkpoint not found: {args.sd_path}"
    assert os.path.exists(args.canny_path), f"Canny checkpoint not found: {args.canny_path}"
    assert not os.path.exists(args.output), f"Output already exists: {args.output}"
    assert os.path.exists(os.path.dirname(args.output)), "Output directory does not exist."

    model = create_model(config_path=args.config)
    scratch = model.state_dict()

    sd_weights = load_state_dict(args.sd_path, location="cpu")
    canny_weights = load_state_dict(args.canny_path, location="cpu")

    target = {}
    newly_initialized = []

    for k in scratch.keys():
        is_control, suffix = get_node_name(k, "control_model.")
        if is_control:
            if k in canny_weights:
                target[k] = canny_weights[k].clone()
            else:
                # Fall back to the SD UNet encoder weight at the matching position
                # (same transfer-init trick as tool_add_control.py).
                mapped = "model.diffusion_model." + suffix
                if mapped in sd_weights:
                    target[k] = sd_weights[mapped].clone()
                else:
                    target[k] = scratch[k].clone()
                    newly_initialized.append(k)
        else:
            if k in sd_weights:
                target[k] = sd_weights[k].clone()
            else:
                target[k] = scratch[k].clone()
                newly_initialized.append(k)

    hint_key = "control_model.input_hint_block.0.weight"
    if hint_key in target and target[hint_key].shape[1] == 3:
        old = target[hint_key]
        new = torch.zeros(
            (old.shape[0], 4, old.shape[2], old.shape[3]), dtype=old.dtype
        )
        new[:, :3, :, :] = old
        target[hint_key] = new
        print(f"Padded {hint_key}: {tuple(old.shape)} -> {tuple(new.shape)} (4th channel zero-init).")

    model.load_state_dict(target, strict=True)

    if newly_initialized:
        print(f"{len(newly_initialized)} key(s) had no source and kept fresh init:")
        for k in newly_initialized:
            print(f"  {k}")

    torch.save(model.state_dict(), args.output)
    print(f"Bundled checkpoint saved at: {args.output}")


if __name__ == "__main__":
    main()

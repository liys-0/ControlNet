import math
import torch
import torch.nn as nn
from ldm.modules.attention import CrossAttention


class LoRALinearLayer(nn.Module):
    def __init__(
        self, in_features, out_features, rank=4, network_alpha=None, bias=False
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        self.regular_linear = nn.Linear(in_features, out_features, bias=bias)

        self.rank = rank
        self.network_alpha = network_alpha or rank
        self.scale = self.network_alpha / self.rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        updown = self.up(self.down(hidden_states.to(dtype)))

        return self.regular_linear(hidden_states) + self.scale * updown.to(orig_dtype)


def inject_trainable_lora(model, rank=4, alpha=None):
    require_grad_params = []

    for name, module in model.named_modules():
        if module.__class__.__name__ in [
            "CrossAttention",
            "MemoryEfficientCrossAttention",
        ]:
            for proj_name in ["to_q", "to_k", "to_v"]:
                orig_proj = getattr(module, proj_name)

                in_features = orig_proj.in_features
                out_features = orig_proj.out_features
                has_bias = orig_proj.bias is not None

                lora_proj = LoRALinearLayer(
                    in_features,
                    out_features,
                    rank=rank,
                    network_alpha=alpha,
                    bias=has_bias,
                )
                lora_proj.regular_linear.weight.data.copy_(orig_proj.weight.data)
                if has_bias:
                    lora_proj.regular_linear.bias.data.copy_(orig_proj.bias.data)

                lora_proj.regular_linear.requires_grad_(False)

                require_grad_params.extend(lora_proj.down.parameters())
                require_grad_params.extend(lora_proj.up.parameters())

                setattr(module, proj_name, lora_proj)

            orig_to_out = module.to_out[0]
            in_features = orig_to_out.in_features
            out_features = orig_to_out.out_features
            has_bias = orig_to_out.bias is not None

            lora_proj = LoRALinearLayer(
                in_features, out_features, rank=rank, network_alpha=alpha, bias=has_bias
            )
            lora_proj.regular_linear.weight.data.copy_(orig_to_out.weight.data)
            if has_bias:
                lora_proj.regular_linear.bias.data.copy_(orig_to_out.bias.data)

            lora_proj.regular_linear.requires_grad_(False)
            require_grad_params.extend(lora_proj.down.parameters())
            require_grad_params.extend(lora_proj.up.parameters())

            module.to_out[0] = lora_proj

    return require_grad_params


def extract_lora_up_down(model):
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinearLayer):
            lora_state_dict[f"{name}.down.weight"] = module.down.weight.data.cpu()
            lora_state_dict[f"{name}.up.weight"] = module.up.weight.data.cpu()
    return lora_state_dict


def load_lora_up_down(model, lora_state_dict, weight=1.0):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinearLayer):
            if f"{name}.down.weight" in lora_state_dict:
                module.down.weight.data.copy_(
                    lora_state_dict[f"{name}.down.weight"].to(module.down.weight.device)
                )
            if f"{name}.up.weight" in lora_state_dict:
                module.up.weight.data.copy_(
                    lora_state_dict[f"{name}.up.weight"].to(module.up.weight.device)
                    * weight
                )
    return model

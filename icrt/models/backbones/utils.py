# Borrowed from https://github.com/Max-Fu/mae-cross/blob/master/transformer_utils.py
# author: Tony Lian and Max Fu
# functionally enable flash attention 2 for torch 2.0 and above

import torch
import torch.nn.functional as F

torch_version = torch.__version__
is_torch2 = torch_version.startswith('2.') 

def handle_flash_attn(args):
    sm = torch.cuda.get_device_capability(0)
    # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    enable_flashattn = sm[0] >= 8 or (sm[0] == 7 and sm[1] >= 5)

    print(f"enable_flashattn: {enable_flashattn}")

    if args.enable_flash_attention2:
        print("Flash attention 2 enabled")
        
        # This requies installing https://github.com/Dao-AILab/flash-attention/tree/v2.2.3
        
        assert enable_flashattn, "Flash attn requires compute capabilities"

        from flash_attn import flash_attn_func

        torch_scaled_dot_product_attention = F.scaled_dot_product_attention

        def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
            # torch convention: B, num heads, seq len, C
            # print(f"Using flash attention, query: {query.shape}, key: {key.shape}, value: {value.shape}")
            assert attn_mask is None, attn_mask
            if query.shape[-1] > 256:
                return torch_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            return torch.permute(flash_attn_func(torch.permute(query, [0, 2, 1, 3]), torch.permute(key, [0, 2, 1, 3]), torch.permute(value, [0, 2, 1, 3]), dropout_p=dropout_p, causal=is_causal), [0, 2, 1, 3])

        F.scaled_dot_product_attention = scaled_dot_product_attention

        # Use memory efficient attention as a fallback
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        print("Flash attention 2 is not enabled. Using built-in attention implementation.")
        torch.backends.cuda.enable_flash_sdp(enable_flashattn)
        torch.backends.cuda.enable_mem_efficient_sdp(not enable_flashattn)
        torch.backends.cuda.enable_math_sdp(False)
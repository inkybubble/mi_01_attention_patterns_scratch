# %%
# Imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from pdb import set_trace

# %%
# Functions
def load_gpt2():
    """
    Load the GPT-2 small model and tokenizer
    Returns (model, tokenizer, device)
    """
    device="mps" if torch.backends.mps.is_available() else "cpu"
    model=GPT2LMHeadModel.from_pretrained("gpt2",
                                          attn_implementation="eager").to(device)
    # without the eager, since torch 2.0+ it uses a faster implementation of the SDPA (Scaled Dot Product attention: faster, but a black box, and we want to see the attention)
    tokenizer=GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer, device

def extract_head_weights(model, layer, head):
    """
    Loads w_q, w_k, w_v (and biases b_q b_k and b_v) and w_o from the model for the head-th head in the layer-th layer
    """

    # Given a layer, the layer has c_attn and c_proj.
    # c_attn weight has this shape:
    # Columns:
    # 0         768       1536      2304
    # |----Q----|----K----|----V----|
    #     ↓         ↓         ↓
    # [768 cols] [768 cols] [768 cols]
    # From transformers/models/gpt2/modeling_gpt2.py
    d_model=model.config.n_embd
    n_head=model.config.n_head
    n_layer=model.config.n_layer
    d_head=d_model//n_head
    assert layer<n_layer, f"The model has {n_layer} layers and you requested layer {layer}"
    assert head<n_head, f"The model has {n_head} heads per layer and you requested head {head}"

    w_q, w_k, w_v = model.transformer.h[layer].attn.c_attn.weight.split(model.transformer.wpe.weight.shape[1], dim=1) # [d_model, d_model]

    # GPT-2 uses biases:
    c_attn_bias = model.transformer.h[layer].attn.c_attn.bias # [d_model*3]
    b_q, b_k, b_v = c_attn_bias.split(d_model) # [d_model]
    b_q_head=b_q[head*d_head:(head+1)*d_head]
    b_k_head=b_k[head*d_head:(head+1)*d_head]
    b_v_head=b_v[head*d_head:(head+1)*d_head]

    
    # for each of w_q, w_k, w_v called w_m, w_m is used for M=w_e @ w_m, where w_m is [d_model, d_head] to go from model_space (residual stream to head), therefore, for the head-th head:
    w_q_head=w_q[:, head*d_head:(head+1)*d_head] # [d_model, d_head]
    w_k_head=w_k[:, head*d_head:(head+1)*d_head]# [d_model, d_head]
    w_v_head=w_v[:, head*d_head:(head+1)*d_head]# [d_model, d_head]
    #  W_o [d_head, d_model] goes from head space to model_space (head to residual stream, to be added there), therefore, for the head-th head:
    w_o=model.transformer.h[layer].attn.c_proj.weight
    w_o_head=w_o[head*d_head:(head+1)*d_head, :] # [d_head, d_model]
    return w_q_head, w_k_head, w_v_head, b_q_head, b_k_head, b_v_head, w_o_head


if __name__=="__main__":
    model, tokenizer, device=load_gpt2()
    print(f"Loaded GPT-2 on {device}")

    w_q, w_k, w_v,b_q, b_k, b_v, w_o = extract_head_weights(model, 0, 3)
    print(f"W_Q: {w_q.shape}, W_K: {w_k.shape}, W_V: {w_v.shape},B_Q: {b_q.shape}, b_K: {b_k.shape}, b_V: {b_v.shape}, W_O:{w_o.shape}")

    # Print c_attn shape for layer 0 to verify access
    layer0_attn = model.transformer.h[0].attn
    print(f"c_attn.weight shape: {layer0_attn.c_attn.weight.shape}")
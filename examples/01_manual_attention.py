# %%
# Imports
import sys
from pathlib import Path
import os

# Add src to path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import transformers
import torch
from model_loader import load_gpt2, extract_head_weights
from attention_manual import attention_manual

from pdb import set_trace


if __name__=="__main__":

    model, tokenizer, device=load_gpt2()

    sentence="Once upon a time there was a bunny"
    x=tokenizer(sentence, return_tensors="pt").to(device)

    input_ids=x["input_ids"].to(device)
    seq_len=input_ids.shape[1]
    position_ids=torch.arange(seq_len).unsqueeze(0).to(device)

    # 1. Get embeddings (what GPT-2 feeds into attention)
    embeddings=model.transformer.wte(input_ids)+model.transformer.wpe(position_ids)
    
    # GPT-2 uses a Layernorm before the attention
    ln_1=model.transformer.h[0].ln_1
    embeddings_normed=ln_1(embeddings)


    # 2. Getting Huggingface attention pattenrns:
    # pass the input through the model:
    with torch.no_grad():
        outputs=model(**x, output_attentions=True)
    # getting layer 0 attention
    hf_attn_layer0=outputs.attentions[0] # [batch, n_heads, seq_len, seq_len]
    # getting attention from batch idx 0, head 0
    hf_attn_head0=hf_attn_layer0[0,0] #[seq, seq] for head 0

    # 3. Getting the manual attention pattern
    w_q, w_k, w_v, b_q, b_k, b_v, w_o=extract_head_weights(model, layer=0, head=0)
    _, manual_attn = attention_manual(embeddings_normed, w_q, w_k, w_v, b_q, b_k, b_v, w_o)
    manual_attn_head0=manual_attn[0] # [seq_len, seq_len]

    # 4. Compare!
    print(f"HF shape: {hf_attn_head0.shape}")
    print(f"Manual shape: {manual_attn_head0.shape}")
    print(f"Match: {torch.allclose(hf_attn_head0, manual_attn_head0, atol=1e-4)}")

    # print(f"HF attn:\n{hf_attn_head0}")
    # print(f"Manual attn:\n{manual_attn_head0}")
    # print(f"Max diff: {(hf_attn_head0 - manual_attn_head0).abs().max()}")

    from visualization import plot_attention

    # Get token strings for labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    plot_attention(manual_attn_head0.detach(), tokens,
                   save_path=os.path.join("outputs", "01_manual_attention_plot.png"))
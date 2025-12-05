# %%
# Imports
import torch
import torch.nn as nn
import math

from pdb import set_trace

# %%
# Functions
def attention_manual(x, w_q, w_k, w_v, b_q, b_k, b_v, w_o):
    """
    Docstring for attention_manual
    
    x: input [batch, seq_len, d_model]
    w_q: query weights [d_model, d_head]
    w_k: key weights [d_model, d_head]
    w_v: value weights [d_model, d_head]
    b_q: query biases [d_model]
    b_k: key biases [d_model]
    b_v: value biases [d_model]
    w_o: projection weights [d_head, d_model]

    Returns:
        output: [batch, seq_len, d_model]
        attention_pattern: [batch, seq_len, seq_len]
    """
    q= x @ w_q +b_q# [batch, seq_len, d_head]
    k= x @ w_k +b_k# [batch, seq_len, d_head]
    v= x @ w_v +b_v# [batch, seq_len, d_head]

    d_head=w_q.shape[1]
    seq_len=x.shape[1]

    # attention scores
    scores= (q @ k.transpose(-2, -1))/math.sqrt(d_head) # [batch, seq_len, seq_len]
    # causal masking
    mask=torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).to(x.device)
    scores=scores.masked_fill(mask==1, float("-inf"))

    attention_pattern=torch.softmax(scores, dim=-1) # [batch, n_head, seq_len, seq_len]

    output=attention_pattern @ v @ w_o

    return output, attention_pattern

if __name__=="__main__":
    from model_loader import load_gpt2,extract_head_weights

    model, tokenizer, device=load_gpt2()
    w_q, w_k, w_v, b_q, b_k, b_v, w_o = extract_head_weights(model, layer=0, head=0)

    # fake input for testin
    x=torch.rand(1,5, 768).to(device)

    output, attn_pattern=attention_manual(x, w_q, w_k, w_v,b_q,b_k, b_v, w_o)
    print(f"Output shape: {output.shape}")           # expect [1, 5, 768]
    print(f"Attention shape: {attn_pattern.shape}")  # expect [1, 5, 5]
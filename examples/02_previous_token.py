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


# %%
# Main gate

if __name__=="__main__":
    """
    This example:
    1. Loads GPT-2 and runs a sentence through it
    2. Gets attention patterns for ALL heads across ALL layers (use HuggingFace output_attentions=True for convenience)
    3. For each head, compute a "previous token score":
    # Average attention to position i-1 (excluding first token which has no previous)
    # prev_token_score = attention_pattern[1:, :-1].diagonal().mean()
    4. Print the top 5 heads with highest previous-token scores
    5. Visualize the best one
    """

    # 1. Loading and tokenizing
    model, tokenizer, device=load_gpt2()

    sentence="Once upon a time there was a bunny"
    x=tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs=model(**x, output_attentions=True)

    # 2. Gets attention patterns for ALL heads across ALL layers (use HuggingFace output_attentions=True for convenience) (we have proven that it's the same)
    config=model.config
    n_ctx=config.n_ctx # 1024
    n_embd=config.n_embd # 768
    n_head=config.n_head # 12
    n_innner=config.n_inner # null
    n_layer=config.n_layer # 12
    n_position=config.n_positions # 1024
    
    attn_dict={}
    for layer_idx in range(n_layer):
        attn_dict[layer_idx]=[]
        hf_attn_layer=outputs.attentions[layer_idx] # [batch, n_heads, seq_len, seq_len]
        for head_idx in range(n_head):
            # fixing batch_0
            hf_attn_head=hf_attn_layer[0,head_idx] #[seq, seq] for head head_idx
            attn_dict[layer_idx].append(hf_attn_head)
    
    # 3. compute the "previous token score" for each head
    '''For a "previous token head', the attention pattern has high value on the diagonal just below the main diagonal (position i attentds to position i-1)'''
    scores=[] # list of (score, layer_idx, head_idx)
    for layer_idx in range(n_layer):
        for head_idx in range(n_head):
            attn=attn_dict[layer_idx][head_idx]
            # attn[1:, :-1]: slice ros 1 onwards, columns 0 to second last
            # diagonal grabs the diagonal() of the sliced matrix
            prev_token_score=attn[1:, :-1].diagonal().mean()
            scores.append((prev_token_score.item(),layer_idx,head_idx))
    
    # 4. Print the top 5 heads with highest previous-token scores
    scores_sorted=sorted(scores, key=lambda x: x[0], reverse=True)

    print("Top 5 Previous token heads:")
    for score, layer, head in scores_sorted[:5]:
        print(f"    Layer {layer}, Head {head}: {score:.4f}")

    # 5. Visualize the best one (i.e. the one that the momst attends to the previous token)
    layer_best_prev=scores_sorted[0][1]
    head_best_prev=scores_sorted[0][2]


    msg_best_prev_head=f"Head that best attends to the previous token\n(head: {head_best_prev} from layer {layer_best_prev})"
    print(msg_best_prev_head)

    tokens = tokenizer.convert_ids_to_tokens(x["input_ids"][0])

    from visualization import plot_attention

    plot_attention(attn_dict[layer_best_prev][head_best_prev].detach(), tokens,
                   title=msg_best_prev_head,
                   save_path=os.path.join("outputs", "02_best_head_at_previous_token_attention.png"))
    '''These PREVIOUS-TOKEN HEADS have strong diagonal patterns just below the main diagonal, i.e. in the attention matrix the entry [i, i-1] is high - 'token i attents to token i-1'
    In the `A Mathematical Framework for Transformer Circuits` they mention that they help copying info from the previous token, which in turn is a building block for more complex behaviours like induction heads
    '''

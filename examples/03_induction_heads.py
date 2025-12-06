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
# Utilities

from collections import defaultdict

def find_repeated_positions(tokens):
    """
    Returns dict: token -> [list of positions]
    Only includes tokens that appear more than once.
    """
    # a dictionary of lists
    positions = defaultdict(list)

    for idx, token in enumerate(tokens):
        positions[token].append(idx)

    # We're keeping the tokens that appear more than ones
    repeated = {tok: pos for tok, pos in positions.items() if len(pos)> 1}
    return repeated

def compute_induction_score(attn, repeated_positions):
    '''
    Args:
        attn: attention
        repeated position: for repeated tokens, their position
    Returns: induction scores for each head
    '''
    score=0.
    for token in repeated_positions:
        # induction means
        # given "The cat sat on the mat. The cat sat on the mat
        # "second "cat" attends to what followed first "cat" (i.e., "sat")"
        score+=attn[repeated_positions[token][1], repeated_positions[token][0]+1].item()
    score/=len(repeated_positions)
    return score

# %%
# Main gate

if __name__=="__main__":
    '''

    This example investigates INDUCTIVE HEADS:
    They perform the logic "I saw [A][B] earlier in this context. Now I see [A] again. I induce (infer) that [B] will follow."
    This counts as a learning pattern from a single example within the context - not from training data, but from that specific prompts. In the Anthropic paper "A Mathematical Framework for Transformer Circuits" this is considered a core mechanism of in-context learning - the model doesn't retrieve memorized facts; it's inducing patterns on-the-fly. They're also called "copying heads" or "in-context learning head", but the paper popularized the name "induction head", because of their behaviour of "reasoning-from-examples".
    1. Uses a repeated-pattern prompt
    2. Gets attention for all heads
    3. Computes an "induction score" — how much the second occurrence of a token attends to what followed the first occurrence
    4. Finds heads with high induction scores
    5. Visualize the best one (i.e. the head with the highest induction score)
    '''

    # 0. Setting up the model, tokenizer and input
    model, tokenizer, device=load_gpt2()

    # 1. Using a repeated pattern. In `A Mathematical Framework for Transformer Circuits` they use "totally random repeated patterns"
    sentence = "The cat sat on the mat. The cat sat on the mat"
    x=tokenizer(sentence, return_tensors="pt").to(device) # access the inputs as x["input_ids"][0]
    tokens=tokenizer.convert_ids_to_tokens(x["input_ids"][0])

    with torch.no_grad():
        outputs=model(**x, output_attentions=True)

    # 2. Getting attention patterns for ALL heads across ALL layers (use HuggingFace output_attentions=True for convenience) (we have proven that it's the same)
    config=model.config
    n_ctx=config.n_ctx # 1024
    n_embd=config.n_embd # 768
    n_head=config.n_head # 12
    n_inner=config.n_inner # null
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

    # 3. Computing an "induction score" — how much the second occurrence of a token attends to what followed the first occurrence
    tokens_repeated_idxs=find_repeated_positions(tokens)
    # -  For each head, sum up attention from second_pos to first_pos + 1
    # - Rank heads by this score
    scores=[]
    for layer_idx in range(n_layer):
        for head_idx in range(n_head):
            score=compute_induction_score(
                attn_dict[layer_idx][head_idx].cpu().numpy(),
                tokens_repeated_idxs
                )
            scores.append((score, layer_idx, head_idx))

    # 4. Finding heads with high induction scores (we're copying code from example 02)
    scores_sorted=sorted(scores, key=lambda x: x[0], reverse=True)

    print("Top 5 Induction heads:")
    for score, layer, head in scores_sorted[:5]:
        print(f"    Layer {layer}, Head {head}: {score:.4f}")


    # 5. Visualizing the best one (i.e. the head with the highest induction score)
    layer_best_induct=scores_sorted[0][1]
    head_best_induct=scores_sorted[0][2]


    msg_best_induct_head=f"Best head at using the second occurrence of a token to attend to what followed the first occurrence of that token\n(head: {head_best_induct} from layer {layer_best_induct})"
    print(msg_best_induct_head)

    tokens = tokenizer.convert_ids_to_tokens(x["input_ids"][0])

    from visualization import plot_attention

    plot_attention(attn_dict[layer_best_induct][head_best_induct].detach(), tokens,
                   title=msg_best_induct_head,
                   save_path=os.path.join("outputs", "03_best_head_at_induction.png"))
    
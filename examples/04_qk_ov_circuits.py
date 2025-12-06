# %%
# Imports
import sys
from pathlib import Path
import os

# Add src to path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from model_loader import load_gpt2, extract_head_weights
from visualization import plot_polar


from pdb import set_trace

# %%
# Main gate

if __name__=="__main__":
    '''
    Background:
    In attention, we have:
    q=x@w_q # w_q is [d_model, d_head]
    k=x@w_k # w_k is [d_model, d_head]
    v=x@w_v # w_v is [d_model, d_head]

    scores=Q@K.T/sqrt(d_head) #[seq, seq]
    attn=softmax(scores)
    output=attn @ V @ W_O # [seq, d_model]

    The anthropic paper `A Mathematical Framework for Transformer Circuits` factors out the input x and analyzes the combined weight matrices as circuits:
    QK: where to look # [d_model, d_model] - d_model is the dimension of the residual stream
        score involves (x@w_q)@(x@w_k).T
    OV: what to copy # [d_model, d_model]
        output involves attn@(x@w_v)@w_o
        i.e., once the attention selects what to copy, what gets copied to the output?
    
    In this example we'll do this:
    1. Loading model, and extracting weights for L5H1 (our best induction head, per example 03)
    2. Printing the shapes of w_q, w_k, w_v, w_o
    3. obtaining qk and ov via matrix multiplications
    4. Compute eigenvalues for both W_QK and W_OV: eigenvalues_ov, eigenvectors_ov = torch.linalg.eig(w_ov)
    5. Compute the copying score for W_OV: # Σλᵢ / Σ|λᵢ|
    6. Print the top eigenvalues by magnitude — see what directions are most amplified
    7. Plot eigenvalues on complex plane like the paper does
    
    EIGENANALYSIS
    eigenvalues amplify the vectors of a vector base. So if their absolute value is higher than 1, they amplify a vector in that directions. If they're lower than 1, they shrink it. When they're negative, they flip it, and when they're 0 they kill the vector amplitude for that direction.

    eigenanalysis - wov (OV: what to copy):
    when a token in position A attends to a token in position B, the OV determines what to write to A's residual stream
    output_at_A=attention_weight*(residual_at_B @ W_OV)
    as a [d_model, d_model] transformation, if its eigenvalues have value lambda:
    - the component of B's residual stream along v gets scaled by lambda
    -> large lambda: that feature is strongly copied
    -> small lambda: that feature gets suppressed.
    ! in induction the job is to copy the token that followed [A] last time. If it saw "the cat" before, and now sees "the" again, the induction head should copy info that helps predict "cat".
    Because token identity lives in the embedding space, we expect W_OV to amplify directions that preserve/copy token-related information.

    eigenanalysis - wqk (QK: where to look)
    The attention score between positions A and B is:
    score=(residual_A @ W_QK @ residual_B.T)/sqrt(d_head)
    w_qk determines which pairs of residual stream vectors have high dot_product (i.e., they attend to each other), and therefore contribute more to the attention score between any A and B
    -> large lambda: directions in the residual stream space that contribute strongly to the attention score

    For our induction head: the job of the induction head is to copy the token that followed [A] last time to help predict what comes after [A] now.
    Copying, as we said, means taking information from position B's residual stream and writing similar information to position A's residual stream
    If the OV circuit has:
    - lambdas >0: information flows in the same direction (copying)
    - lambda< 0: information gets inverted (not copied)
    So we expect an induction head to have a high copying score
    '''

    # 1. Setting up the model
    model, _, device=load_gpt2() # we won't need the tokenizer - this example is about weights
    # L5H1 weights
    layer_idx=5
    head_idx=1
    w_q, w_k, w_v, b_q, b_k, b_v,w_o=extract_head_weights(model, layer=layer_idx, head=head_idx)
    # 2. Printing shapes and make sure they make sense:
    config=model.config
    n_ctx=config.n_ctx # 1024
    n_embd=config.n_embd # 768
    n_head=config.n_head # 12
    n_inner=config.n_inner # null
    n_layer=config.n_layer # 12
    n_position=config.n_positions # 1024
    d_head=n_embd//n_head
    print(f"d_head: {d_head}")
    print(f"d_model: {n_embd}")

    print(f"W_Q: {w_q.shape} # [d_model, d_head]")
    print(f"W_K: {w_k.shape} # [d_model, d_head]")
    print(f"W_V: {w_v.shape} # [d_model, d_head]")
    print(f"W_O: {w_o.shape} # [d_head, d_model]")

    # 3. obtaining qk and ov via matrix multiplications
    w_qk=w_q@(w_k.T) # [d_model, d_model]
    w_ov=w_v@w_o # [d_model, d_model] (from [d_model, d_head] @ [d_head, d_model])
    print(f"W_QK: {w_qk.shape} # [d_model, d_model]")
    print(f"W_OV: {w_ov.shape} # [d_model, d_model]")

    # 4. Compute eigenvalues for both W_QK and W_OV: 
    eigenvalues_qk, eigenvectors_qk = torch.linalg.eig(w_qk.cpu())
    eigenvalues_ov, eigenvectors_ov = torch.linalg.eig(w_ov.cpu())
    # using cpu() because it says The operator 'aten::linalg_eig' is not currently implemented for the MPS device - whaaaaat?

    # 5. Compute the copying score for W_OV
    copying_score=torch.sum(eigenvalues_ov.real)/torch.sum(torch.abs(eigenvalues_ov))

    msg_copying_score=f"Copying score for L{layer_idx}H{head_idx}: {copying_score:.2f}"
    print(msg_copying_score)

    # 6. Print the top eigenvalues by magnitude
    eigenvalues_ov_top_5_amplitudes_idx=torch.topk(torch.abs(eigenvalues_ov), largest=True, k=5)
    msg_top_evalues=f"Top eigenvalues for L{layer_idx}H{head_idx}: {eigenvalues_ov_top_5_amplitudes_idx.values.tolist()}"
    print(msg_top_evalues)

    # 7. 
    # eigenvalues_ov is a complex tensor
    # for the best induction head (per example 03)
    title_polar_ov="OV Circuit Eigenvalues - L5H1 (best induction head)"
    save_path_polar_ov=os.path.join("outputs", "04_ov_eigenvalues_complex_best_induction.png")
    plot_polar(eigenvalues_ov, title=title_polar_ov, save_path=save_path_polar_ov)
    meaningful = eigenvalues_ov.abs() > 0.01  # threshold
    print(f"Non-zero eigenvalues (L5H1): {meaningful.sum()}")  # should be ~64

    # 8. Best copying head (from example)
    # L4H11 weights (best previous token head, from example 02)
    layer_idx=4
    head_idx=11

    w_q, w_k, w_v, b_q, b_k, b_v,w_o=extract_head_weights(model, layer=layer_idx, head=head_idx)
    w_qk=w_q@(w_k.T)
    w_ov=w_v@w_o
    eigenvalues_ov, eigenvectors_ov = torch.linalg.eig(w_ov.cpu())
    copying_score=torch.sum(eigenvalues_ov.real)/torch.sum(torch.abs(eigenvalues_ov))
    msg_copying_score=f"Copying score for L{layer_idx}H{head_idx}: {copying_score:.2f}"
    print(msg_copying_score)
    title_polar_ov="OV Circuit Eigenvalues - L4H11 (best prev token)"
    save_path_polar_ov=os.path.join("outputs", "04_ov_eigenvalues_complex_best_prev_token.png")
    plot_polar(eigenvalues_ov, title=title_polar_ov, save_path=save_path_polar_ov)
    meaningful = eigenvalues_ov.abs() > 0.01  # threshold
    print(f"Non-zero eigenvalues (L4H11): {meaningful.sum()}")  # should be ~64
# %%
# Imports
import matplotlib.pyplot as plt

from pdb import set_trace

# %%
# Plotting utilities

def plot_attention(attention_pattern, tokens, save_path=None,
                   dark: bool=True):
    '''
    Docstring for plot_attention
    
    Args: 
    attention_pattern: [seq_len, seq_len] tensor
    tokens: list of token string (for axis labels)
    save_path: optional path to save a .png file
    '''
    if dark is True:
        plt.style.use('dark_background')
    fig, ax=plt.subplots() # for when creading a figure - plt.plot() works only for line plots
    ap_np=attention_pattern.cpu().numpy()
    
    c_min=ap_np.min()
    c_max=ap_np.max()
    im=ax.imshow(ap_np, vmin=c_min, vmax=c_max)
    plt.xlabel("Source (Key) - 'attends to'")
    plt.ylabel("Destination (Query) - 'token'")
    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=45, ha='right')
    plt.yticks(ticks=range(len(tokens)), labels=tokens)

    plt.colorbar(im)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

"""
Visual helpers for Transformers & Attention notebook.

This module contains visualization functions and utility functions to keep the notebook 
clean and focused on concepts rather than implementation details.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_evolution_timeline():
    """
    Plot the timeline of sequence modeling evolution from LSTMs to modern transformers.
    """
    timeline = [
        (1997, "LSTM", "Sequential processing\nwith memory cells"),
        (2014, "Seq2Seq +\nAttention", "First attention\nmechanism"),
        (2017, "Transformer", "Pure attention,\nno recurrence"),
        (2018, "BERT", "Bidirectional\npre-training"),
        (2020, "GPT-3", "Large-scale\ngeneration"),
        (2023, "ChatGPT", "Instruction-tuned\nconversational AI")
    ]

    years = [item[0] for item in timeline]
    labels = [item[1] for item in timeline]
    descriptions = [item[2] for item in timeline]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Draw timeline
    ax.plot([min(years) - 1, max(years) + 1], [0, 0], 'k-', linewidth=2, alpha=0.3)

    # Plot milestones
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C7CEEA']
    for i, (year, label, desc) in enumerate(timeline):
        # Alternate heights for better readability
        height = 0.3 if i % 2 == 0 else -0.3
        
        # Draw vertical line
        ax.plot([year, year], [0, height], 'k--', linewidth=1, alpha=0.5)
        
        # Draw point
        ax.scatter(year, height, s=300, c=[colors[i]], zorder=10, edgecolors='white', linewidth=2)
        
        # Add year label
        ax.text(year, -0.02 if height > 0 else 0.02, str(year), 
                ha='center', va='top' if height > 0 else 'bottom', 
                fontsize=9, fontweight='bold', color='gray')
        
        # Add model label
        va = 'bottom' if height > 0 else 'top'
        ax.text(year, height + (0.05 if height > 0 else -0.05), label, 
                ha='center', va=va, fontsize=11, fontweight='bold')
        
        # Add description
        desc_y = height + (0.15 if height > 0 else -0.15)
        ax.text(year, desc_y, desc, ha='center', va=va, 
                fontsize=8, alpha=0.7, style='italic')

    # Formatting
    ax.set_xlim(min(years) - 2, max(years) + 2)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title('From RNNs to Modern Transformers', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()

    print("Key Insight: The 2017 'Attention Is All You Need' paper eliminated recurrence entirely!")
    print("This unlocked parallelization and enabled models to scale to billions of parameters.")


def plot_rnn_vs_transformer():
    """
    Visualize the difference between RNN sequential processing and Transformer
    parallel all-to-all attention.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sample sentence
    words = ["The", "cat", "sat", "on", "mat"]
    n_words = len(words)

    # Left plot: RNN (Sequential)
    ax1.set_title("RNN: Sequential Processing", fontsize=13, fontweight='bold', pad=10)
    for i in range(n_words):
        # Draw word box
        ax1.add_patch(plt.Rectangle((i, 0), 0.8, 0.6, 
                                     facecolor='lightblue', edgecolor='black', linewidth=2))
        ax1.text(i + 0.4, 0.3, words[i], ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw arrows between words (sequential flow)
        if i < n_words - 1:
            ax1.arrow(i + 0.8, 0.3, 0.15, 0, head_width=0.15, head_length=0.08, 
                      fc='red', ec='red', linewidth=2)

    # Add "information bottleneck" annotation
    ax1.annotate('Information\nBottleneck', xy=(2.5, 0.3), xytext=(2.5, -0.8),
                fontsize=9, ha='center', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(2.5, -1.2, 'Early info must pass through\nALL intermediate steps', 
             ha='center', fontsize=8, style='italic', color='red')

    ax1.set_xlim(-0.5, n_words)
    ax1.set_ylim(-1.5, 1.2)
    ax1.axis('off')

    # Right plot: Transformer (All-to-All Attention)
    ax2.set_title("Transformer: All-to-All Attention", fontsize=13, fontweight='bold', pad=10)

    for i in range(n_words):
        # Draw word box
        ax2.add_patch(plt.Rectangle((i, 0), 0.8, 0.6, 
                                     facecolor='lightgreen', edgecolor='black', linewidth=2))
        ax2.text(i + 0.4, 0.3, words[i], ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw ALL attention connections (not just from one word)
    # Create a grid showing all-to-all connections
    connection_strength = [
        [0.3, 0.2, 0.2, 0.1, 0.1],  # The
        [0.5, 0.3, 0.4, 0.1, 0.1],  # cat
        [0.4, 0.5, 0.3, 0.3, 0.2],  # sat
        [0.2, 0.2, 0.4, 0.3, 0.5],  # on
        [0.2, 0.2, 0.3, 0.5, 0.3],  # mat
    ]

    # Draw connections with varying opacity based on strength
    for i in range(n_words):
        for j in range(n_words):
            if i != j:
                strength = connection_strength[i][j]
                # Adjust curve radius based on distance
                radius = 0.3 if abs(i - j) == 1 else 0.5
                ax2.annotate('', 
                            xy=(j + 0.4, 0.65), 
                            xytext=(i + 0.4, 0.65),
                            arrowprops=dict(arrowstyle='->', 
                                          color='blue', 
                                          alpha=strength,
                                          lw=1.5,
                                          connectionstyle=f"arc3,rad={radius}"))

    ax2.text(2.5, -0.5, 'Every word directly attends to\nEVERY other word', 
             ha='center', fontsize=9, color='blue', fontweight='bold')
    ax2.text(2.5, -0.9, 'No information bottleneck!\nDirect long-range connections', 
             ha='center', fontsize=8, style='italic', color='blue')

    ax2.set_xlim(-0.5, n_words)
    ax2.set_ylim(-1.2, 1.5)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    print("🔑 Key Differences:")
    print("   RNN: Sequential processing → Information bottleneck → Slow")
    print("   Transformer: Parallel processing → Direct connections → Fast & effective")
    print("\nNotice: In the transformer, you can see connections going in ALL directions,")
    print("        showing that each word can attend to every other word simultaneously!")


def create_causal_mask(seq_len):
    """
    Create a causal mask for autoregressive attention.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        torch.Tensor: Lower triangular mask where 1 = allowed, 0 = blocked.
                      Shape: (seq_len, seq_len)
    
    Example:
        >>> mask = create_causal_mask(3)
        >>> print(mask)
        tensor([[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]])
    
    Usage:
        Position i can only attend to positions <= i (past and present, not future).
        This is used in autoregressive models like GPT for causal/masked attention.
    """
    # torch.tril creates a lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


def visualize_causal_masking(attn_weights, masked_attn, seq_len):
    """
    Visualize the effect of causal masking on attention weights.
    
    Creates a side-by-side comparison of attention patterns with and without
    causal masking to demonstrate how autoregressive models prevent attending
    to future positions.
    
    Args:
        attn_weights: Original attention weights (batch_size, seq_len, seq_len)
        masked_attn: Masked attention weights (batch_size, seq_len, seq_len)
        seq_len: Sequence length
        
    Returns:
        None (displays the plot)
    """
    print("\nInterpretation:")
    print("  - Row 0 (position 0): Can only see position 0 (itself)")
    print("  - Row 1 (position 1): Can see positions 0-1")
    print("  - Row 2 (position 2): Can see positions 0-2")
    print("  - etc. → Each position can only see past and present, not future!")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Original attention (no mask)
    sns.heatmap(
        attn_weights[0].detach().numpy(),
        annot=True,
        fmt='.3f',
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Weight'},
        ax=ax1
    )
    ax1.set_title('Without Causal Mask', fontweight='bold')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')

    # Masked attention
    sns.heatmap(
        masked_attn[0].detach().numpy(),
        annot=True,
        fmt='.3f',
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Weight'},
        ax=ax2
    )
    ax2.set_title('With Causal Mask (Autoregressive)', fontweight='bold')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')

    plt.tight_layout()
    plt.show()

    print("\n💡 Notice: With causal masking, upper triangle is all zeros!")
    print("   This is how GPT-style models generate text one token at a time.")

def visualize_attention_heads(attentions, layer_idx, heads_to_show, tokens):
    """
    Visualize attention patterns from different heads in a specific layer.
    
    Args:
        attentions: Tuple of attention tensors from model output
        layer_idx: Layer index to visualize (0-indexed)
        heads_to_show: List of head indices to display
        tokens: List of token strings for axis labels
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, head_num in enumerate(heads_to_show):
        # Get attention weights for this specific head
        # Shape: (seq_len, seq_len)
        attn = attentions[layer_idx][0, head_num].detach().numpy()
        
        # Create heatmap
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            ax=axes[idx],
            cbar_kws={'label': 'Attention Weight'},
            square=True,
            vmin=0,
            vmax=1
        )
        
        axes[idx].set_title(f'Layer {layer_idx + 1}, Head {head_num + 1}', 
                            fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Keys (attending to)', fontsize=10)
        axes[idx].set_ylabel('Queries (attending from)', fontsize=10)
        
        # Rotate labels for better readability
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

    print("\n🔍 Notice how different heads focus on different patterns:")
    print("   - Some heads may focus on nearby words (local context)")
    print("   - Others may focus on specific tokens like [CLS] or punctuation")
    print("   - Some may show coreference patterns (connecting 'it' to 'mat' or 'cat')")
    print("   - Different heads learn complementary patterns automatically!")

def plot_transformer_architectures():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    # Common styling
    box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black', linewidth=2)
    attn_style = dict(boxstyle='round,pad=0.3', facecolor='lightcoral', edgecolor='black', linewidth=2)
    norm_style = dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='black', linewidth=2)
    ff_style = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='black', linewidth=2)

    def draw_block(ax, x, y, width, height, text, style, fontsize=10):
        """Draw a rectangular block with text"""
        rect = FancyBboxPatch((x, y), width, height, **style)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2):
        """Draw an arrow between two points"""
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)

    # ============================================================================
    # 1. ENCODER-ONLY (BERT)
    # ============================================================================
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Encoder-Only (BERT)', fontsize=14, fontweight='bold', pad=20)

    # Input
    draw_block(ax1, 2, 0.5, 6, 0.8, 'Input Tokens + Positional Encoding', box_style, 9)
    draw_arrow(ax1, 5, 1.3, 5, 2)

    # Encoder block (repeated)
    y_start = 2
    for i in range(2):
        # Multi-head attention
        draw_block(ax1, 2, y_start, 6, 1, 'Multi-Head\nSelf-Attention', attn_style)
        draw_arrow(ax1, 5, y_start + 1, 5, y_start + 1.3)
        
        # Add & Norm
        draw_block(ax1, 2, y_start + 1.3, 6, 0.6, 'Add & Norm', norm_style, 8)
        draw_arrow(ax1, 5, y_start + 1.9, 5, y_start + 2.2)
        
        # Feed Forward
        draw_block(ax1, 2, y_start + 2.2, 6, 1, 'Feed Forward', ff_style)
        draw_arrow(ax1, 5, y_start + 3.2, 5, y_start + 3.5)
        
        # Add & Norm
        draw_block(ax1, 2, y_start + 3.5, 6, 0.6, 'Add & Norm', norm_style, 8)
        
        if i == 0:
            draw_arrow(ax1, 5, y_start + 4.1, 5, y_start + 4.5)
            y_start += 4.5
        
    # Output
    draw_arrow(ax1, 5, 10.6, 5, 11)
    draw_block(ax1, 2, 11, 6, 0.8, 'Contextual Embeddings', box_style, 9)

    # Annotation
    ax1.text(5, -0.5, '✓ Bidirectional\n✓ Classification/QA', 
            ha='center', fontsize=9, style='italic')

    # ============================================================================
    # 2. DECODER-ONLY (GPT)
    # ============================================================================
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Decoder-Only (GPT)', fontsize=14, fontweight='bold', pad=20)

    # Input
    draw_block(ax2, 2, 0.5, 6, 0.8, 'Input Tokens + Positional Encoding', box_style, 9)
    draw_arrow(ax2, 5, 1.3, 5, 2)

    # Decoder block (repeated)
    y_start = 2
    for i in range(2):
        # Masked Multi-head attention
        draw_block(ax2, 2, y_start, 6, 1, 'Masked Multi-Head\nSelf-Attention', attn_style)
        draw_arrow(ax2, 5, y_start + 1, 5, y_start + 1.3)
        
        # Add & Norm
        draw_block(ax2, 2, y_start + 1.3, 6, 0.6, 'Add & Norm', norm_style, 8)
        draw_arrow(ax2, 5, y_start + 1.9, 5, y_start + 2.2)
        
        # Feed Forward
        draw_block(ax2, 2, y_start + 2.2, 6, 1, 'Feed Forward', ff_style)
        draw_arrow(ax2, 5, y_start + 3.2, 5, y_start + 3.5)
        
        # Add & Norm
        draw_block(ax2, 2, y_start + 3.5, 6, 0.6, 'Add & Norm', norm_style, 8)
        
        if i == 0:
            draw_arrow(ax2, 5, y_start + 4.1, 5, y_start + 4.5)
            y_start += 4.5

    # Output
    draw_arrow(ax2, 5, 10.6, 5, 11)
    draw_block(ax2, 2, 11, 6, 0.8, 'Next Token Prediction', box_style, 9)

    # Annotation
    ax2.text(5, -0.5, '✓ Causal Masking\n✓ Text Generation', 
            ha='center', fontsize=9, style='italic')

    # ============================================================================
    # 3. ENCODER-DECODER (T5)
    # ============================================================================
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 12)
    ax3.axis('off')
    ax3.set_title('Encoder-Decoder (T5)', fontsize=14, fontweight='bold', pad=20)

    # Input
    draw_block(ax3, 0.5, 0.5, 4, 0.8, 'Source Input', box_style, 9)
    draw_arrow(ax3, 2.5, 1.3, 2.5, 2)

    # Encoder (simplified)
    draw_block(ax3, 0.5, 2, 4, 2.5, 'ENCODER\n\nMulti-Head\nSelf-Attention\n+\nFeed Forward', attn_style)
    encoder_out_y = 4.5
    draw_arrow(ax3, 2.5, encoder_out_y, 2.5, encoder_out_y + 0.5)
    draw_block(ax3, 0.5, encoder_out_y + 0.5, 4, 0.6, 'Encoder Output', box_style, 8)

    # Decoder input
    draw_block(ax3, 5.5, 0.5, 4, 0.8, 'Target Input', box_style, 9)
    draw_arrow(ax3, 7.5, 1.3, 7.5, 2)

    # Decoder (simplified)
    draw_block(ax3, 5.5, 2, 4, 1.2, 'Masked\nSelf-Attention', attn_style, 9)
    draw_arrow(ax3, 7.5, 3.2, 7.5, 3.5)

    # Cross attention (with connection from encoder)
    draw_block(ax3, 5.5, 3.5, 4, 1.2, 'Cross-Attention\nto Encoder', 
            dict(boxstyle='round,pad=0.3', facecolor='orange', edgecolor='black', linewidth=2), 9)
    # Arrow from encoder to cross-attention
    draw_arrow(ax3, 4.5, 5.1, 5.5, 4.1)

    draw_arrow(ax3, 7.5, 4.7, 7.5, 5.0)

    # Feed forward
    draw_block(ax3, 5.5, 5.0, 4, 1.2, 'Feed Forward', ff_style, 9)
    draw_arrow(ax3, 7.5, 6.2, 7.5, 6.5)

    # Output
    draw_block(ax3, 5.5, 6.5, 4, 0.8, 'Generated Output', box_style, 9)

    # Annotation
    ax3.text(5, -0.5, '✓ Encoder + Decoder\n✓ Translation/Summarization', 
            ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.show()

    print("\nKey Differences:")
    print("="*60)
    print("• ENCODER-ONLY: Processes all tokens bidirectionally")
    print("                → Good for understanding tasks")
    print("\n• DECODER-ONLY: Masked attention, sees only past tokens")
    print("                → Good for generation tasks")
    print("\n• ENCODER-DECODER: Encoder understands, decoder generates")
    print("                   → Good for transformation tasks")
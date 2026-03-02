"""
Visual helpers for Vision Transformers & CLIP notebook (07).

This module contains visualization functions and utility functions to keep the notebook
clean and focused on concepts rather than implementation details.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import torch


def plot_patching_process(image_tensor: torch.Tensor, patch_size: int = 4) -> None:
    """
    Visualize how an image is split into a grid of patches (the first step of ViT).

    Args:
        image_tensor: Image tensor of shape (C, H, W) with values in [0, 1].
        patch_size: Size of each square patch in pixels.
    """
    C, H, W = image_tensor.shape
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    n_patches = n_patches_h * n_patches_w

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Left: original image ---
    img_np = image_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # --- Middle: image with grid overlay ---
    axes[1].imshow(img_np)
    for i in range(1, n_patches_h):
        axes[1].axhline(y=i * patch_size - 0.5, color="red", linewidth=1.5)
    for j in range(1, n_patches_w):
        axes[1].axvline(x=j * patch_size - 0.5, color="red", linewidth=1.5)
    axes[1].set_title(
        f"Patch Grid ({n_patches_h}×{n_patches_w} = {n_patches} patches)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].axis("off")

    # --- Right: patches laid out as a sequence ---
    max_show = min(n_patches, 16)  # show up to 16 patches
    cols = min(8, max_show)
    rows = int(np.ceil(max_show / cols))

    # Create a grid image of individual patches
    grid_img = np.ones((rows * (patch_size + 2), cols * (patch_size + 2), 3))
    patch_idx = 0
    for pi in range(n_patches_h):
        for pj in range(n_patches_w):
            if patch_idx >= max_show:
                break
            patch = image_tensor[
                :, pi * patch_size : (pi + 1) * patch_size, pj * patch_size : (pj + 1) * patch_size
            ]
            patch_np = patch.permute(1, 2, 0).numpy()
            patch_np = np.clip(patch_np, 0, 1)

            r = patch_idx // cols
            c = patch_idx % cols
            y0 = r * (patch_size + 2) + 1
            x0 = c * (patch_size + 2) + 1
            grid_img[y0 : y0 + patch_size, x0 : x0 + patch_size, :] = patch_np
            patch_idx += 1
        if patch_idx >= max_show:
            break

    axes[2].imshow(grid_img)
    axes[2].set_title(
        f"Patches as Token Sequence (first {max_show})",
        fontsize=13,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"🔑 Each {patch_size}×{patch_size} patch becomes a token (vector of length {patch_size * patch_size * C}).")
    print(f"   Total sequence length: {n_patches} patches + 1 [CLS] token = {n_patches + 1} tokens.")


def plot_vit_architecture() -> None:
    """
    Draw a simplified ViT architecture diagram:
    Image → Patches → Linear Projection → [CLS] + Positional Embedding
    → Transformer Encoder → Classification Head.
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black", linewidth=2)
    attn_style = dict(boxstyle="round,pad=0.3", facecolor="lightcoral", edgecolor="black", linewidth=2)
    embed_style = dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="black", linewidth=2)
    cls_style = dict(boxstyle="round,pad=0.3", facecolor="lightgreen", edgecolor="black", linewidth=2)

    # Step 1: Image
    ax.text(1.5, 3.5, "Image\n(3×H×W)", ha="center", va="center", fontsize=11, fontweight="bold", bbox=box_style)

    # Arrow
    ax.annotate("", xy=(3.0, 3.5), xytext=(2.4, 3.5), arrowprops=dict(arrowstyle="->", lw=2))

    # Step 2: Split into patches
    ax.text(4.0, 3.5, "Split into\nPatches", ha="center", va="center", fontsize=11, fontweight="bold", bbox=embed_style)

    # Arrow
    ax.annotate("", xy=(5.5, 3.5), xytext=(4.9, 3.5), arrowprops=dict(arrowstyle="->", lw=2))

    # Step 3: Linear Projection
    ax.text(6.7, 3.5, "Linear\nProjection", ha="center", va="center", fontsize=11, fontweight="bold", bbox=embed_style)

    # Arrow
    ax.annotate("", xy=(8.1, 3.5), xytext=(7.6, 3.5), arrowprops=dict(arrowstyle="->", lw=2))

    # Step 4: [CLS] + Positional Embeddings
    ax.text(9.2, 3.5, "[CLS] +\nPos Embed", ha="center", va="center", fontsize=11, fontweight="bold", bbox=cls_style)

    # Arrow
    ax.annotate("", xy=(10.6, 3.5), xytext=(10.1, 3.5), arrowprops=dict(arrowstyle="->", lw=2))

    # Step 5: Transformer Encoder
    ax.text(11.8, 3.5, "Transformer\nEncoder\n(×L layers)", ha="center", va="center", fontsize=11, fontweight="bold", bbox=attn_style)

    # Arrow
    ax.annotate("", xy=(13.2, 3.5), xytext=(12.7, 3.5), arrowprops=dict(arrowstyle="->", lw=2))

    # Step 6: Classification Head
    ax.text(14.2, 3.5, "MLP\nHead", ha="center", va="center", fontsize=11, fontweight="bold", bbox=cls_style)

    # Annotations below
    annotations = [
        (1.5, 1.5, "e.g. 3×224×224"),
        (4.0, 1.5, "16×16 pixel\npatches → N tokens"),
        (6.7, 1.5, "Each patch →\nD-dim vector"),
        (9.2, 1.5, "Learnable class\ntoken prepended"),
        (11.8, 1.5, "Multi-Head\nSelf-Attention\n+ FFN"),
        (14.2, 1.5, "Predict\nclass label"),
    ]
    for x, y, text in annotations:
        ax.text(x, y, text, ha="center", va="center", fontsize=9, style="italic", alpha=0.7)

    ax.set_title("Vision Transformer (ViT) Architecture", fontsize=15, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.show()

    print("🔑 Key insight: ViT treats an image like a sentence of patch-tokens.")
    print("   The [CLS] token aggregates information from all patches via attention.")
    print("   This is the SAME transformer architecture used for text (notebook 05)!")


def plot_cls_attention_flow() -> None:
    """
    Diagram showing how the [CLS] token learns an image representation
    by attending to all patch tokens through multiple transformer layers.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Styles
    patch_style = dict(boxstyle="round,pad=0.2", facecolor="#AED6F1", edgecolor="black", lw=1.5)
    cls_style   = dict(boxstyle="round,pad=0.2", facecolor="#ABEBC6", edgecolor="black", lw=1.5)
    out_style   = dict(boxstyle="round,pad=0.2", facecolor="#D7BDE2", edgecolor="black", lw=1.5)
    head_style  = dict(boxstyle="round,pad=0.3", facecolor="#F1948A", edgecolor="black", lw=1.5)

    # ── Row 1: Input tokens ──
    ax.text(8, 8.3, "Input Sequence", ha="center", fontsize=13, fontweight="bold")
    ax.text(2, 7.3, "[CLS]\n(random)", ha="center", va="center", fontsize=9, fontweight="bold", bbox=cls_style)
    for i, label in enumerate(["Patch 1", "Patch 2", "Patch 3", "...", "Patch 196"]):
        x = 4.5 + i * 2.2
        style = patch_style if label != "..." else dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="white")
        ax.text(x, 7.3, label, ha="center", va="center", fontsize=9,
                fontweight="bold" if label != "..." else "normal", bbox=style)

    # ── Arrows down to attention ──
    arrow_kw = dict(arrowstyle="-|>", lw=1.5, color="#555555")
    for x in [2, 4.5, 6.7, 8.9, 13.3]:
        ax.annotate("", xy=(x, 6.2), xytext=(x, 6.8), arrowprops=arrow_kw)

    # ── Row 2: Self-Attention block ──
    ax.add_patch(plt.Rectangle((0.8, 5.0), 13.5, 1.2, fill=True, facecolor="#FEF9E7",
                                 edgecolor="#F39C12", lw=2, zorder=0, clip_on=False))
    ax.text(7.5, 5.6, "Self-Attention:  every token attends to every other token",
            ha="center", va="center", fontsize=11, fontweight="bold", style="italic")

    # Draw attention arrows from CLS to all patches (highlight!)
    for x_end in [4.5, 6.7, 8.9, 13.3]:
        ax.annotate("", xy=(x_end, 5.15), xytext=(2, 5.15),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#E74C3C",
                                    connectionstyle="arc3,rad=0.15", alpha=0.6))

    # Label the CLS→patch arrows
    ax.text(7.5, 4.5, "[CLS] reads from ALL patches via attention weights",
            ha="center", va="center", fontsize=10, color="#C0392B", fontweight="bold")

    # ── Arrows down ──
    for x in [2, 4.5, 6.7, 8.9, 13.3]:
        ax.annotate("", xy=(x, 3.5), xytext=(x, 4.2), arrowprops=arrow_kw)

    # ── Row 3: Output tokens ──
    ax.text(8, 3.9, "x 12 layers", ha="center", fontsize=10, style="italic", alpha=0.6)
    ax.text(2, 2.8, "[CLS]\n(now image\naware!)", ha="center", va="center", fontsize=9,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#2ECC71",
                                          edgecolor="black", lw=1.5))
    for i, label in enumerate(["Patch 1'", "Patch 2'", "Patch 3'", "...", "Patch 196'"]):
        x = 4.5 + i * 2.2
        style = out_style if label != "..." else dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="white")
        ax.text(x, 2.8, label, ha="center", va="center", fontsize=9,
                fontweight="bold" if label != "..." else "normal", bbox=style)

    # ── Arrow from CLS to classification head ──
    ax.annotate("", xy=(2, 1.3), xytext=(2, 2.2), arrowprops=dict(arrowstyle="-|>", lw=2, color="#2ECC71"))
    ax.text(2, 0.8, "Classification\nHead (MLP)", ha="center", va="center", fontsize=10,
            fontweight="bold", bbox=head_style)
    ax.text(5.5, 0.8, "<-- Only the [CLS] output is used for classification",
            ha="left", va="center", fontsize=10, style="italic", color="#7D3C98")

    ax.set_title("How [CLS] Learns an Image Representation Through Attention",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.show()

    print("🔑 The [CLS] token starts with NO image information.")
    print("   Through 12 layers of self-attention, it attends to all 196 patches,")
    print("   gradually building a global image representation.")
    print("   At the end, only the [CLS] output is passed to the classification head.")
    print("\n💡 This is why 'attention maps' are meaningful: they show which patches")
    print("   the [CLS] token relied on most to build its image understanding.")


def plot_attention_maps(
    image: torch.Tensor,
    attentions: tuple,
    patch_size: int = 16,
    image_size: int = 224,
    layer_idx: int = -1,
    heads_to_show: list[int] | None = None,
) -> None:
    """
    Visualize per-head attention maps from a ViT model overlaid on the input image.

    Args:
        image: Original image tensor (C, H, W) for display (values in [0,1]).
        attentions: Tuple of attention tensors from model output (one per layer).
                    Each has shape (batch, num_heads, seq_len, seq_len).
        patch_size: ViT patch size (e.g. 16).
        image_size: Input image size (e.g. 224).
        layer_idx: Which layer to visualize (-1 = last).
        heads_to_show: List of head indices to show (default: first 6).
    """
    attn = attentions[layer_idx]  # (1, num_heads, seq_len, seq_len)
    num_heads = attn.shape[1]
    n_patches_side = image_size // patch_size

    if heads_to_show is None:
        heads_to_show = list(range(min(6, num_heads)))

    # Get attention from [CLS] token (index 0) to all patch tokens
    cls_attn = attn[0, :, 0, 1:]  # (num_heads, n_patches)

    n_show = len(heads_to_show)
    cols = min(3, n_show)
    rows = int(np.ceil((n_show + 1) / cols))  # +1 for original image

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    img_np = image.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    # Original image in first cell
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Attention heatmaps
    for idx, head in enumerate(heads_to_show):
        r = (idx + 1) // cols
        c = (idx + 1) % cols

        head_attn = cls_attn[head].detach().cpu().numpy()
        head_attn = head_attn.reshape(n_patches_side, n_patches_side)
        # Normalize to [0, 1]
        head_attn = (head_attn - head_attn.min()) / (head_attn.max() - head_attn.min() + 1e-8)
        # Resize to image size
        from PIL import Image as PILImage

        attn_map = PILImage.fromarray((head_attn * 255).astype(np.uint8))
        attn_map = attn_map.resize((image_size, image_size), PILImage.BILINEAR)
        attn_map = np.array(attn_map).astype(np.float32) / 255.0

        axes[r, c].imshow(img_np)
        axes[r, c].imshow(attn_map, cmap="jet", alpha=0.5)
        axes[r, c].set_title(f"Head {head + 1}", fontsize=11, fontweight="bold")
        axes[r, c].axis("off")

    # Turn off unused axes
    for idx in range(n_show + 1, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    plt.suptitle(
        f"[CLS] Token Attention (Layer {len(attentions) + layer_idx + 1 if layer_idx < 0 else layer_idx + 1})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    print("🔍 Each heatmap shows what the [CLS] token attends to in that head.")
    print("   Warm colors (red/yellow) = high attention, cool colors (blue) = low.")
    print("   Notice how different heads focus on different regions!")


def compute_attention_rollout(attentions: tuple) -> np.ndarray:
    """
    Compute attention rollout by multiplying attention matrices across all layers.

    This gives a global view of what the [CLS] token attends to across the entire model,
    accounting for residual connections.

    Args:
        attentions: Tuple of attention tensors, each (1, num_heads, seq_len, seq_len).

    Returns:
        np.ndarray: Rolled-out attention map from [CLS] to all patches, shape (n_patches,).
    """
    # Average attention across heads for each layer
    result = None
    for attn in attentions:
        # attn shape: (1, num_heads, seq_len, seq_len)
        attn_heads_avg = attn[0].mean(dim=0).detach().numpy()  # (seq_len, seq_len)
        # Add identity (residual connection)
        attn_heads_avg = 0.5 * attn_heads_avg + 0.5 * np.eye(attn_heads_avg.shape[0])
        # Re-normalize
        attn_heads_avg = attn_heads_avg / attn_heads_avg.sum(axis=-1, keepdims=True)

        if result is None:
            result = attn_heads_avg
        else:
            result = result @ attn_heads_avg

    # CLS token attention to patch tokens (skip CLS-to-CLS at index 0)
    cls_rollout = result[0, 1:]
    return cls_rollout


def plot_attention_rollout(
    image: torch.Tensor,
    attentions: tuple,
    patch_size: int = 16,
    image_size: int = 224,
) -> None:
    """
    Visualize attention rollout — the aggregated attention from [CLS] across all layers.

    Args:
        image: Original image tensor (C, H, W) for display.
        attentions: Tuple of attention tensors from model output.
        patch_size: ViT patch size.
        image_size: Input image size.
    """
    n_patches_side = image_size // patch_size
    rollout = compute_attention_rollout(attentions)
    rollout = rollout.reshape(n_patches_side, n_patches_side)

    # Normalize
    rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-8)

    # Resize to image size
    from PIL import Image as PILImage

    rollout_img = PILImage.fromarray((rollout * 255).astype(np.uint8))
    rollout_img = rollout_img.resize((image_size, image_size), PILImage.BILINEAR)
    rollout_np = np.array(rollout_img).astype(np.float32) / 255.0

    img_np = image.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(rollout_np, cmap="jet")
    axes[1].set_title("Attention Rollout", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(rollout_np, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(
        "Attention Rollout: What Does the [CLS] Token See?",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    print("🔑 Attention rollout aggregates attention across ALL layers.")
    print("   It accounts for residual connections, giving a more global view")
    print("   than looking at a single layer's attention.")
    print("   This is what the model 'looks at' to make its classification decision.")


def plot_cnn_vs_clip_comparison(cnn_acc: float, clip_acc: float) -> None:
    """
    Side-by-side bar chart comparing CNN and CLIP test accuracy on CIFAR-10.

    Args:
        cnn_acc: CNN test accuracy (0-1 scale).
        clip_acc: CLIP test accuracy (0-1 scale).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["CNN\n(from notebook 06)", "CLIP ViT\n(fine-tuned)"]
    accs = [cnn_acc * 100, clip_acc * 100]
    colors = ["#4ECDC4", "#FF6B6B"]

    bars = ax.bar(models, accs, color=colors, edgecolor="black", linewidth=2, width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("CIFAR-10: CNN vs Fine-Tuned CLIP", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    diff = abs(clip_acc - cnn_acc) * 100
    if clip_acc > cnn_acc:
        print(f"📊 CLIP outperforms CNN by {diff:.1f} percentage points on CIFAR-10!")
        print("   This advantage comes from CLIP's contrastive pretraining on 400M image-text pairs.")
        print("   The CNN was trained from scratch on CIFAR-10 only.")
    else:
        print(f"📊 CNN and CLIP perform similarly ({diff:.1f}pp difference).")
        print("   With more epochs or unfreezing the backbone, CLIP typically pulls ahead.")

    print("\n💡 Key takeaway: Pretrained CLIP delivers strong results with minimal fine-tuning,")
    print("   but CNNs remain efficient for small-data, low-latency, and edge deployment scenarios.")


def plot_clip_architecture() -> None:
    """
    Draw a simplified CLIP architecture diagram showing the dual-encoder structure:
    Image Encoder (ViT) + Text Encoder → shared embedding space with contrastive loss.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    img_style = dict(boxstyle="round,pad=0.3", facecolor="#AED6F1", edgecolor="black", linewidth=2)
    txt_style = dict(boxstyle="round,pad=0.3", facecolor="#ABEBC6", edgecolor="black", linewidth=2)
    enc_style = dict(boxstyle="round,pad=0.3", facecolor="#F9E79F", edgecolor="black", linewidth=2)
    emb_style = dict(boxstyle="round,pad=0.3", facecolor="#D7BDE2", edgecolor="black", linewidth=2)
    loss_style = dict(boxstyle="round,pad=0.4", facecolor="#F1948A", edgecolor="black", linewidth=2)

    # --- Image path (top) ---
    ax.text(2.0, 6.0, "Image", ha="center", va="center", fontsize=12, fontweight="bold", bbox=img_style)
    ax.annotate("", xy=(3.5, 6.0), xytext=(2.8, 6.0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(4.8, 6.0, "Patch + Embed", ha="center", va="center", fontsize=11, fontweight="bold", bbox=img_style)
    ax.annotate("", xy=(6.3, 6.0), xytext=(5.8, 6.0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(7.5, 6.0, "ViT Encoder\n(12 layers)", ha="center", va="center", fontsize=11, fontweight="bold", bbox=enc_style)
    ax.annotate("", xy=(9.1, 6.0), xytext=(8.6, 6.0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(10.3, 6.0, "Image\nEmbedding", ha="center", va="center", fontsize=11, fontweight="bold", bbox=emb_style)

    # --- Text path (bottom) ---
    ax.text(2.0, 2.5, "Text", ha="center", va="center", fontsize=12, fontweight="bold", bbox=txt_style)
    ax.annotate("", xy=(3.5, 2.5), xytext=(2.8, 2.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(4.8, 2.5, "BPE Tokenize", ha="center", va="center", fontsize=11, fontweight="bold", bbox=txt_style)
    ax.annotate("", xy=(6.3, 2.5), xytext=(5.8, 2.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(7.5, 2.5, "Text Encoder\n(12 layers)", ha="center", va="center", fontsize=11, fontweight="bold", bbox=enc_style)
    ax.annotate("", xy=(9.1, 2.5), xytext=(8.6, 2.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(10.3, 2.5, "Text\nEmbedding", ha="center", va="center", fontsize=11, fontweight="bold", bbox=emb_style)

    # --- Contrastive loss (right) ---
    ax.annotate("", xy=(12.5, 4.8), xytext=(11.1, 5.7), arrowprops=dict(arrowstyle="->", lw=2, color="#8E44AD"))
    ax.annotate("", xy=(12.5, 3.7), xytext=(11.1, 2.8), arrowprops=dict(arrowstyle="->", lw=2, color="#8E44AD"))
    ax.text(13.5, 4.25, "Contrastive\nLoss\n(cosine sim)", ha="center", va="center", fontsize=12, fontweight="bold", bbox=loss_style)

    # --- Labels ---
    ax.text(8.0, 7.5, "CLIP Architecture: Contrastive Language-Image Pre-training",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(7.5, 6.9, "Image Encoder (ViT)", ha="center", va="center", fontsize=10,
            style="italic", color="#2471A3")
    ax.text(7.5, 1.6, "Text Encoder (Transformer)", ha="center", va="center", fontsize=10,
            style="italic", color="#1E8449")
    ax.text(13.5, 2.2, "Match: image ↔ caption\nMismatch: push apart",
            ha="center", va="center", fontsize=9, style="italic", alpha=0.7)

    plt.tight_layout()
    plt.show()

    print("🔑 CLIP trains two encoders jointly on 400M image-text pairs from the internet.")
    print("   The image encoder is a ViT — the SAME architecture we just studied!")
    print("   At inference, we can compare ANY image with ANY text (zero-shot classification).")


def plot_tokenization_comparison() -> None:
    """
    Draw a comparison diagram of three tokenization approaches:
    1. Text tokenization (BPE/WordPiece) — discrete vocabulary lookup
    2. ViT patch embeddings — continuous linear projection
    3. VQ-VAE — discrete codebook lookup
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    discrete_style = dict(boxstyle="round,pad=0.25", facecolor="#AED6F1", edgecolor="black", linewidth=1.5)
    continuous_style = dict(boxstyle="round,pad=0.25", facecolor="#ABEBC6", edgecolor="black", linewidth=1.5)
    input_style = dict(boxstyle="round,pad=0.25", facecolor="#F9E79F", edgecolor="black", linewidth=1.5)
    output_style = dict(boxstyle="round,pad=0.25", facecolor="#D7BDE2", edgecolor="black", linewidth=1.5)

    # --- Panel 1: Text BPE ---
    ax = axes[0]
    ax.set_title("Text: BPE Tokenization", fontsize=13, fontweight="bold", pad=10)
    ax.text(5, 9, '"a photo of a cat"', ha="center", va="center", fontsize=10, bbox=input_style)
    ax.annotate("", xy=(5, 7.8), xytext=(5, 8.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 7.3, "BPE Tokenizer", ha="center", va="center", fontsize=10, bbox=discrete_style)
    ax.annotate("", xy=(5, 6.1), xytext=(5, 6.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 5.5, "[320, 1125, 539, 320, 2368]", ha="center", va="center", fontsize=9, bbox=discrete_style)
    ax.annotate("", xy=(5, 4.3), xytext=(5, 5.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 3.8, "Embedding Lookup\n(vocab table)", ha="center", va="center", fontsize=10, bbox=discrete_style)
    ax.annotate("", xy=(5, 2.5), xytext=(5, 3.2), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 1.8, "Dense vectors\n[v₁, v₂, v₃, v₄, v₅]", ha="center", va="center", fontsize=10, bbox=output_style)
    ax.text(5, 0.5, "DISCRETE IDs → lookup", ha="center", va="center", fontsize=9, style="italic", color="red")

    # --- Panel 2: ViT Patches (Continuous) ---
    ax = axes[1]
    ax.set_title("ViT: Patch Embeddings", fontsize=13, fontweight="bold", pad=10)
    ax.text(5, 9, "Image (3×224×224)", ha="center", va="center", fontsize=10, bbox=input_style)
    ax.annotate("", xy=(5, 7.8), xytext=(5, 8.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 7.3, "Split into 16×16\npatches (196 total)", ha="center", va="center", fontsize=10, bbox=continuous_style)
    ax.annotate("", xy=(5, 5.8), xytext=(5, 6.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 5.2, "Linear Projection\n(Conv2d / matmul)", ha="center", va="center", fontsize=10, bbox=continuous_style)
    ax.annotate("", xy=(5, 3.8), xytext=(5, 4.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 3.2, "Raw pixels → dense vectors\nDIRECTLY", ha="center", va="center", fontsize=10, bbox=continuous_style)
    ax.annotate("", xy=(5, 2.0), xytext=(5, 2.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 1.4, "Dense vectors\n[v₁, v₂, ..., v₁₉₆]", ha="center", va="center", fontsize=10, bbox=output_style)
    ax.text(5, 0.3, "CONTINUOUS — no vocab table", ha="center", va="center", fontsize=9, style="italic", color="green")

    # --- Panel 3: VQ-VAE (Discrete) ---
    ax = axes[2]
    ax.set_title("VQ-VAE: Vector Quantization", fontsize=13, fontweight="bold", pad=10)
    ax.text(5, 9, "Image (3×H×W)", ha="center", va="center", fontsize=10, bbox=input_style)
    ax.annotate("", xy=(5, 7.8), xytext=(5, 8.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 7.3, "CNN Encoder →\nlatent vectors", ha="center", va="center", fontsize=10, bbox=discrete_style)
    ax.annotate("", xy=(5, 5.8), xytext=(5, 6.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 5.2, "Nearest-Neighbor\nCodebook Lookup", ha="center", va="center", fontsize=10, bbox=discrete_style)
    ax.annotate("", xy=(5, 3.8), xytext=(5, 4.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 3.2, "Discrete code IDs\n[42, 7, 128, 3, ...]", ha="center", va="center", fontsize=10, bbox=discrete_style)
    ax.annotate("", xy=(5, 2.0), xytext=(5, 2.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 1.4, "Codebook vectors\n[c₄₂, c₇, c₁₂₈, c₃, ...]", ha="center", va="center", fontsize=10, bbox=output_style)
    ax.text(5, 0.3, "DISCRETE IDs → codebook lookup", ha="center", va="center", fontsize=9, style="italic", color="red")

    plt.tight_layout()
    plt.show()

    print("🔑 Text tokenization is DISCRETE: words → integer IDs → embedding table lookup.")
    print("   ViT patch embeddings are CONTINUOUS: raw pixels → linear projection → dense vectors.")
    print("   VQ-VAE is DISCRETE like text: encode → find nearest codebook entry → discrete ID.")
    print("\n💡 This is why ViT is perfect for multimodal LLMs: it produces the same kind of")
    print("   dense vectors as text embeddings — just concatenate them and feed to a transformer!")

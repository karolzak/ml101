# Core libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def model_weights_visualization():
    word_embeddings = {
        "The": np.array([0.2, 0.1, 0.8, 0.3]),
        "cat": np.array([0.9, 0.2, 0.1, 0.7]),
        "sat": np.array([0.3, 0.8, 0.2, 0.4]),
    }
    # Possible next words with their embeddings
    candidates = {
        "on": np.array([0.4, 0.7, 0.3, 0.5]),
        "under": np.array([0.5, 0.6, 0.4, 0.3]),
        "near": np.array([0.6, 0.5, 0.5, 0.2]),
        "quickly": np.array([0.1, 0.3, 0.9, 0.8]),
    }
    np.random.seed(42)
    W1 = np.array([
        [ 0.8, -0.2,  0.3,  0.5],
        [-0.3,  0.9,  0.1,  0.2],
        [ 0.4,  0.1,  0.7, -0.4],
        [ 0.2,  0.6, -0.1,  0.8]
    ])

    # Bias terms (more parameters!)
    b1 = np.array([0.1, -0.2, 0.3, 0.0])


    # Step 3: Mathematical Transformation
    # =====================================================
    # Combine the input embeddings (simplified: just average them)
    input_vector = np.mean([word_embeddings["The"], 
                            word_embeddings["cat"], 
                            word_embeddings["sat"]], axis=0)

    # Apply the transformation
    hidden = np.dot(W1, input_vector) + b1

    scores = {}
    for word, embedding in candidates.items():
        # Calculate similarity (dot product) between transformed input and candidate
        score = np.dot(hidden, embedding)
        scores[word] = score

    # Find the winner
    best_word = max(scores, key=scores.get)

    # Draw a simplified neural network diagram showing the flow
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Define positions
    input_x, hidden_x, output_x = 1.5, 5, 8.5
    input_y_positions = [8, 6.5, 5, 3.5]
    hidden_y_positions = [8, 6, 4, 2]
    output_y_positions = [7, 5.5, 4, 2.5]

    # Draw input layer
    for i, (word, y) in enumerate(zip(["The", "cat", "sat", "(combined)"], input_y_positions)):
        circle = plt.Circle((input_x, y), 0.3, color='lightblue', ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(input_x - 0.4, y, word, ha='right', va='center', fontsize=12, fontweight='bold')
        if i < 3:
            vec = word_embeddings[word]
            ax.text(input_x, y - 0.6, f'[{vec[0]:.1f},{vec[1]:.1f},{vec[2]:.1f},{vec[3]:.1f}]', 
                ha='center', va='top', fontsize=9, color='darkblue')

    # Draw hidden layer
    for i, y in enumerate(hidden_y_positions):
        circle = plt.Circle((hidden_x, y), 0.35, color='gold', ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(hidden_x, y, f'h{i+1}', ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw output layer
    for i, (word, y) in enumerate(zip(["on", "under", "near", "quickly"], output_y_positions)):
        color = 'lightgreen' if word == best_word else 'lightcoral'
        circle = plt.Circle((output_x, y), 0.3, color=color, ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(output_x + 0.4, y, f'{word}: {scores[word]:.2f}', 
            ha='left', va='center', fontsize=12, fontweight='bold')

    # Draw connections (weights) from input to hidden
    for in_y in [input_y_positions[3]]:  # Just from combined input
        for hid_y in hidden_y_positions:
            ax.plot([input_x + 0.3, hidden_x - 0.35], [in_y, hid_y], 
                'gray', alpha=0.4, linewidth=1, zorder=1)

    # Draw connections from hidden to output
    for hid_y in hidden_y_positions:
        for out_y in output_y_positions:
            ax.plot([hidden_x + 0.35, output_x - 0.3], [hid_y, out_y], 
                'gray', alpha=0.4, linewidth=1, zorder=1)

    # Add labels
    ax.text(input_x, 9.3, 'INPUT LAYER\n(Words as Numbers)', 
        ha='center', fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))

    ax.text(hidden_x, 9.3, 'HIDDEN LAYER\n(Weights × Input)', 
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.5))

    ax.text(output_x, 9.3, 'OUTPUT LAYER\n(Next Word Scores)', 
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))

    # Add weight annotations
    ax.annotate('WEIGHTS (W1)\n16 parameters\n\n↓\n\nW × input + bias',
                xy=(3.2, 5.5), xytext=(3.2, 5.5),
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', alpha=0.7, linewidth=2))

    ax.annotate('MORE WEIGHTS\n16 parameters\n\n↓\n\nW × hidden',
                xy=(6.7, 5.5), xytext=(6.7, 5.5),
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', alpha=0.7, linewidth=2))

    # Title
    ax.text(5, 0.5, 'A Neural Network = Mathematical Function with Learned Weights', 
        ha='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='black', linewidth=2))

    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("🧠 THE CORE CONCEPT:")
    print("="*70)
    print("This toy network has: 4 inputs × 4 hidden + 4 hidden × 4 outputs")
    print(f"                      = 16 + 16 + 8 bias = 40 parameters total")
    print("\nA real LLM like GPT-4:")
    print("  • Has 100+ layers (not just 2)")
    print("  • Each layer has millions of connections")
    print("  • Total: ~1,760,000,000,000 parameters")
    print("\nBut the PRINCIPLE is exactly the same:")
    print("  Text → Numbers → Weights × Numbers → More Numbers → Prediction")
    print("\n🎯 The 'magic' is in the WEIGHTS, learned from massive data!")

def model_scale_comparison():
    # Model scale comparison
    models = {
        'GPT-2\n(2019)': 1.5,
        'GPT-3\n(2020)': 175,
        'GPT-3.5\n(2022)': 175,
        'GPT-4\n(2023)': 1700,  # Estimated
        'Claude 3.5\nSonnet\n(2024)': 200,  # Estimated
        'Llama 3.1\n405B\n(2024)': 405,
        'Human Brain\nSynapses': 100000  # ~100 trillion synapses
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#3498db'] * 6 + ['#e74c3c']
    bars = ax.barh(list(models.keys()), list(models.values()), color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for i, (model, params) in enumerate(models.items()):
        if params >= 1000:
            label = f'{params/1000:.0f}T'
        else:
            label = f'{params:.0f}B'
        ax.text(params + max(models.values())*0.02, i, label, va='center', fontweight='bold', fontsize=12)

    ax.set_xlabel('Parameters / Synapses', fontsize=13, fontweight='bold')
    ax.set_title('The Scale of "Large" in LLMs\n(B = Billion, T = Trillion)', fontsize=15, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3, which='both')

    # Add annotation
    ax.annotate('Each parameter is a learned weight\nthat helps predict the next word',
                xy=(175, 1), xytext=(8000, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.show()

    print("\n💡 Key Insight:")
    print("   - Modern LLMs have 100+ billion parameters")
    print("   - Each parameter is a number the model learned during training")
    print("   - More parameters = more capacity to learn patterns (but diminishing returns)")
    print("   - GPT-4 estimated to have ~1.7 trillion parameters (not confirmed)")
    print("   - Still orders of magnitude less than human brain synapses!")

def calc_memory_and_costs():
    # Calculate memory requirements for different models
    # Each parameter is typically stored as a float (4 bytes for FP32, 2 bytes for FP16)

    models_memory = {
        'GPT-3 175B': {'params': 175, 'precision': 'FP16'},
        'GPT-4 1.7T': {'params': 1700, 'precision': 'FP16'},
        'Llama 3.1 405B': {'params': 405, 'precision': 'FP16'},
    }

    # Calculate memory in GB
    memory_data = {}
    for model, info in models_memory.items():
        params_billions = info['params']
        bytes_per_param = 2 if info['precision'] == 'FP16' else 4  # FP16 = 2 bytes, FP32 = 4 bytes
        
        # Memory = params * bytes_per_param
        memory_gb = (params_billions * 1e9 * bytes_per_param) / (1024**3)
        memory_data[model] = {
            'memory_gb': memory_gb,
            'params': params_billions
        }

    # Add GPU specifications
    gpu_specs = {
        'NVIDIA A100': {'memory_gb': 80, 'cost_per_hour': 3.67},
        'NVIDIA H100': {'memory_gb': 80, 'cost_per_hour': 8.00},
        'NVIDIA H200': {'memory_gb': 141, 'cost_per_hour': 12.00}  # Estimated
    }

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 30))

    # 1. Model memory requirements
    model_names = list(memory_data.keys())
    memory_values = [memory_data[m]['memory_gb'] for m in model_names]

    colors_mem = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax1.barh(model_names, memory_values, color=colors_mem, alpha=0.8, edgecolor='black', linewidth=2)

    for i, (bar, mem) in enumerate(zip(bars, memory_values)):
        ax1.text(mem + 20, bar.get_y() + bar.get_height()/2, 
                f'{mem:.0f} GB', 
                va='center', fontweight='bold', fontsize=12)

    ax1.set_xlabel('Memory Required (GB)', fontsize=13, fontweight='bold')
    ax1.set_title('Model Memory Requirements\n(FP16 Precision)', fontsize=15, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3)

    # 2. GPUs needed per model
    gpu_counts = {}
    for model in model_names:
        mem_needed = memory_data[model]['memory_gb']
        gpu_counts[model] = {}
        for gpu, specs in gpu_specs.items():
            # Account for model parallelism overhead (~20%)
            gpus_needed = int(np.ceil(mem_needed / (specs['memory_gb'] * 0.8)))
            gpu_counts[model][gpu] = gpus_needed

    # Create grouped bar chart
    x = np.arange(len(model_names))
    width = 0.25
    gpu_names = list(gpu_specs.keys())

    for i, gpu in enumerate(gpu_names):
        counts = [gpu_counts[m][gpu] for m in model_names]
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, counts, width, label=gpu, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of GPUs Required', fontsize=13, fontweight='bold')
    ax2.set_title('GPU Count Needed for Inference\n(Minimum Configuration)', fontsize=15, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha='right')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Cost comparison for running models (per hour)
    costs_per_hour = {}
    for model in model_names:
        costs_per_hour[model] = {}
        for gpu in gpu_names:
            cost = gpu_counts[model][gpu] * gpu_specs[gpu]['cost_per_hour']
            costs_per_hour[model][gpu] = cost

    # Create grouped bar chart for costs
    x = np.arange(len(model_names))
    width = 0.25
    colors_cost = ['#3498db', '#e67e22', '#9b59b6']

    for i, gpu in enumerate(gpu_names):
        costs = [costs_per_hour[m][gpu] for m in model_names]
        offset = (i - 1) * width
        bars = ax3.bar(x + offset, costs, width, label=gpu, alpha=0.8, edgecolor='black', 
                linewidth=1.5, color=colors_cost[i])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:.0f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax3.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Cost per Hour (USD)', fontsize=13, fontweight='bold')
    ax3.set_title('Inference Cost Comparison\n(Cloud GPU Pricing)', fontsize=15, fontweight='bold', pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=15, ha='right')
    ax3.legend(loc='upper left', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Training cluster size comparison
    training_gpus = {
        'GPT-3': 10000,
        'GPT-4\n(estimated)': 25000,
        'Llama 3.1 405B': 16000,
        'Meta Llama 2': 2000
    }

    colors_train = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax4.barh(list(training_gpus.keys()), list(training_gpus.values()), 
                    color=colors_train, alpha=0.8, edgecolor='black', linewidth=2)

    for i, (bar, gpus) in enumerate(zip(bars, training_gpus.values())):
        ax4.text(gpus + 500, bar.get_y() + bar.get_height()/2, 
                f'{gpus:,} GPUs', 
                va='center', fontweight='bold', fontsize=11)

    ax4.set_xlabel('Number of GPUs', fontsize=13, fontweight='bold')
    ax4.set_title('Training Cluster Size\n(Peak GPU Count During Training)', fontsize=15, fontweight='bold', pad=20)
    ax4.grid(axis='x', alpha=0.3)

    # Add cost annotation
    training_cost_estimate = 25000 * 8 * 24 * 90  # 25k H100s * $8/hr * 24hrs * 90 days
    ax4.annotate(f'GPT-4 training cost\nestimated at ${training_cost_estimate/1e6:.0f}M+', 
                xy=(25000, 1), xytext=(15000, 2.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.show()

    print("\n💰 Key Cost Insights:")
    print(f"   - GPT-3 (175B): ~{memory_data['GPT-3 175B']['memory_gb']:.0f} GB memory")
    print(f"   - Minimum {gpu_counts['GPT-3 175B']['NVIDIA H100']} x H100 GPUs needed for inference")
    print(f"   - Inference cost: ~${costs_per_hour['GPT-3 175B']['NVIDIA H100']:.0f}/hour on H100s")
    print(f"\n   - GPT-4 (1.7T): ~{memory_data['GPT-4 1.7T']['memory_gb']:.0f} GB memory")
    print(f"   - Minimum {gpu_counts['GPT-4 1.7T']['NVIDIA H100']} x H100 GPUs needed")
    print(f"   - Training required ~25,000 GPUs for months")
    print(f"\n⚡ Why This Matters:")
    print(f"   - Each API call runs on multi-GPU clusters")
    print(f"   - Operating at scale requires data center infrastructure")
    print(f"   - Cloud costs can reach millions per month for popular services")
    print(f"   - Smaller models (7B-70B) can run on single GPUs → edge deployment possible")

def visualize_real_next_word_predictions(prompt: str = None, temperature: float = 2.0):
    # Visualize next-word predictions using real Azure OpenAI data
    from openai import AzureOpenAI
    from dotenv import load_dotenv
    import os
    import math

    # Load environment variables from .env file
    load_dotenv()

    # Initialize Azure OpenAI client
    # Credentials loaded from .env file in repository root
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Request logprobs (log probabilities) from the API
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Complete the user's sentence with just one word."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1,
        temperature=temperature,  # Higher temperature for more equalized probabilities
        logprobs=True,
        top_logprobs=5,  # Get top 5 alternatives
        n=1
    )

    # Extract the actual probabilities from chat completion
    logprobs_data = response.choices[0].logprobs.content[0].top_logprobs
    # logprobs_data = response.choices[0].logprobs.content[0].top_logprobs

    # Convert log probabilities to actual probabilities
    tokens = [item.token for item in logprobs_data]
    logprobs = [item.logprob for item in logprobs_data]
    # Convert log probabilities to actual probabilities using softmax
    logprobs_array = np.array(logprobs)
    exp_logprobs = np.exp(logprobs_array - np.max(logprobs_array))  # Subtract max for numerical stability
    probs = exp_logprobs / np.sum(exp_logprobs)  # Softmax normalization

    using_real_data = True
    print("✅ Successfully connected to Azure OpenAI!")
        
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Bar chart of top predictions
    colors_bars = plt.cm.viridis(np.linspace(0.3, 0.9, len(tokens)))
    bars = ax1.barh(range(len(tokens)), probs, color=colors_bars, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(len(tokens)))
    ax1.set_yticklabels([f'"{t}"' for t in tokens], fontsize=13)
    ax1.set_xlabel('Probability', fontsize=13, fontweight='bold')
    ax1.set_title(f'Next-Word Predictions for: "{prompt} ____"\n{"[REAL Azure OpenAI Data]" if using_real_data else "[Simulated Data]"}', 
                fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', 
                va='center', fontweight='bold', fontsize=14)

    # Highlight the winner
    ax1.patches[0].set_edgecolor('red')
    ax1.patches[0].set_linewidth(3)

    # Pie chart showing top 3 vs rest
    top3_prob = sum(probs[:3])
    rest_prob = sum(probs[3:])
    pie_labels = [f'Top 3 words\n({top3_prob*100:.1f}%)', f'Other options\n({rest_prob*100:.1f}%)']
    pie_sizes = [top3_prob, rest_prob]
    pie_colors = ['#3498db', '#95a5a6']

    wedges, texts, autotexts = ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%',
                                        startangle=90, colors=pie_colors,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax2.set_title('Concentration of Probability\n(Top predictions dominate)', 
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()

    print(f"\n🎯 Next-Word Prediction Results:")
    print(f"   Prompt: '{prompt} ____'")
    print(f"   Most likely: '{tokens[0]}' ({probs[0]*100:.1f}%)")
    print(f"   Runner-up: '{tokens[1]}' ({probs[1]*100:.1f}%)")
    print(f"   Third: '{tokens[2]}' ({probs[2]*100:.1f}%)")
    print(f"\n💡 Key Insights:")
    print(f"   - Top 3 predictions account for {top3_prob*100:.1f}% of probability")
    print(f"   - LLM assigns different probabilities based on training data patterns")
    print(f"   - Model is {'confident' if probs[0] > 0.3 else 'uncertain'} (highest prob: {probs[0]*100:.1f}%)")
    print(f"   - Temperature controls randomness in final selection")


def visualize_real_tokenization(text_example: str = None):
    # Visualize real tokenization using tiktoken
    import tiktoken

    # Use actual GPT tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    # Get actual tokens and token IDs
    token_ids = encoding.encode(text_example)
    tokens = [encoding.decode([token_id]) for token_id in token_ids]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Real GPT Tokenization: How GPT-4 Breaks Down Text', 
                    ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)

    # Original text
    ax.text(0.5, 0.85, f'Original Text: "{text_example}"', 
                    ha='center', fontsize=13, transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.5))

    # Arrow
    ax.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.8),
                            arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                            transform=ax.transAxes)

    x_pos = 0.00
    x_step = 0.03
    y_pos = 0.6

    # Display tokens
    for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
            # Clean up token display (replace newlines/spaces with visible characters)
            display_token = token.replace('\n', '\\n').replace(' ', '␣')
            x_pos = x_pos + min((x_step * len(display_token)), 0.06)
            
            # Token box
            ax.text(x_pos, y_pos, f'{display_token}', ha='center', fontsize=10, fontweight='bold',
                            transform=ax.transAxes,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', edgecolor='black', linewidth=2))
            # Token ID
            ax.text(x_pos, y_pos - 0.1, str(token_id), ha='center', fontsize=10,
                            transform=ax.transAxes, style='italic', color='gray')

    # Arrow
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.45),
                            arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                            transform=ax.transAxes)

    # Numbers representation
    ax.text(0.5, 0.25, f'Token IDs: {token_ids}', 
                    ha='center', fontsize=11, transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.5))

    ax.text(0.5, 0.15, f'Total tokens: {len(token_ids)} | Original characters: {len(text_example)}', 
                    ha='center', fontsize=10, fontweight='bold', transform=ax.transAxes)

    ax.text(0.5, 0.05, '→ These exact numbers are fed into GPT-4', 
                    ha='center', fontsize=10, style='italic', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

    print(f"\n🔍 Real GPT-4 Tokenization Results:")
    print(f"   Original text: '{text_example}'")
    print(f"   Number of tokens: {len(token_ids)}")
    print(f"   Tokens: {tokens}")
    print(f"   Token IDs: {token_ids}")

def compare_vocab_sizes():
    # Compare vocabulary sizes across different models and tokenization strategies

    # Model vocabulary sizes
    vocab_sizes = {
        'GPT-2\n(2019)': 50257,
        'GPT-3/3.5\n(2020-2022)': 50257,
        'GPT-4\n(cl100k_base)': 100277,
        'Claude\n(2023)': 100000,
        'Llama 2\n(2023)': 32000,
        'Llama 3\n(2024)': 128256
    }
    # Visualize real tokenization using tiktoken
    import tiktoken

    # Use actual GPT tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    # Hypothetical comparison: Character-level vs Subword tokenization
    sentence = "The quick brown fox jumps over the lazy dog"
    char_tokens = len(sentence)  # Each character = 1 token
    gpt4_tokens = len(encoding.encode(sentence))  # Subword tokens

    # Calculate implications
    avg_chars_per_token = char_tokens / gpt4_tokens

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 30))

    # 1. Vocabulary sizes comparison
    models = list(vocab_sizes.keys())
    sizes = list(vocab_sizes.values())

    colors_vocab = ['#3498db', '#3498db', '#e74c3c', '#9b59b6', '#2ecc71', '#f39c12']
    bars = ax1.barh(models, sizes, color=colors_vocab, alpha=0.8, edgecolor='black', linewidth=2)

    for i, (bar, size) in enumerate(zip(bars, sizes)):
        ax1.text(size + 2000, bar.get_y() + bar.get_height()/2, 
                f'{size:,}', 
                va='center', fontweight='bold', fontsize=11)

    ax1.set_xlabel('Vocabulary Size (Number of Tokens)', fontsize=13, fontweight='bold')
    ax1.set_title('Vocabulary Sizes Across LLMs\n(Fixed Dictionary Size)', fontsize=15, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3)

    # Add character-level reference
    char_vocab_size = 256  # UTF-8 basic characters
    ax1.axvline(x=char_vocab_size, color='red', linestyle='--', alpha=0.7, linewidth=3)
    ax1.text(char_vocab_size + 1000, len(models) - 1, 'Character-level\n(~256 chars)', 
            fontsize=10, color='red', fontweight='bold')

    # 2. Why not character-level? - Sequence length explosion
    context_windows = [2048, 4096, 8192, 32768, 128000]
    context_labels = ['2K', '4K', '8K', '32K', '128K']

    # Calculate tokens vs characters for different context windows
    text_capacity_words_subword = [int(ctx * avg_chars_per_token / 5) for ctx in context_windows]  # ~5 chars/word
    text_capacity_words_char = [int(ctx / 5) for ctx in context_windows]  # Direct char-to-word

    x = np.arange(len(context_labels))
    width = 0.35

    bars1 = ax2.bar(x - width/2, text_capacity_words_subword, width, label='Subword Tokens (GPT-4)', 
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, text_capacity_words_char, width, label='Character Tokens', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Context Window Size', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Approximate Words Capacity', fontsize=13, fontweight='bold')
    ax2.set_title('Text Capacity: Subword vs Character Tokenization\n(Why Character-Level is Inefficient)', 
                fontsize=15, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(context_labels)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Computational cost comparison
    sequence_lengths = [100, 500, 1000, 2000, 5000]

    # Attention is O(n²) where n is sequence length
    # Character-level would be ~4x longer for same text
    char_seq_lengths = [s * 4 for s in sequence_lengths]
    subword_costs = [s**2 for s in sequence_lengths]
    char_costs = [s**2 for s in char_seq_lengths]

    ax3.plot(sequence_lengths, subword_costs, marker='o', linewidth=3, markersize=10, 
            color='#2ecc71', label='Subword Tokens', alpha=0.8)
    ax3.plot(sequence_lengths, char_costs, marker='s', linewidth=3, markersize=10, 
            color='#e74c3c', label='Character Tokens', alpha=0.8)

    ax3.set_xlabel('Text Length (Subword Tokens)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Computational Cost (Attention O(n²))', fontsize=13, fontweight='bold')
    ax3.set_title('Why Character-Level is Computationally Expensive\n(Attention Complexity)', 
                fontsize=15, fontweight='bold', pad=20)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(alpha=0.3)
    ax3.set_yscale('log')

    # Add annotation
    ax3.annotate('Character-level: 16x more expensive\nfor same text at 2000 tokens!', 
                xy=(2000, char_costs[3]), xytext=(1000, char_costs[3] * 2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    # 4. Vocabulary construction - BPE (Byte Pair Encoding) example
    # Simulate how BPE builds vocabulary
    initial_text = "aaabdaaabac"
    steps = [
        ("Initial", list(initial_text), "Each char is a token"),
        ("Merge 'aa'", initial_text.replace('aa', 'Z'), "Most frequent pair merged"),
        ("Merge 'ab'", initial_text.replace('aa', 'Z').replace('ab', 'Y'), "Next frequent pair"),
        ("Final", "ZYdZYac", "Efficient representation")
    ]

    y_positions = [0.75, 0.55, 0.35, 0.15]
    ax4.axis('off')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    ax4.text(0.5, 0.95, 'How Vocabulary is Built: Byte Pair Encoding (BPE)', 
            ha='center', fontsize=16, fontweight='bold')
    ax4.text(0.5, 0.88, 'Iteratively merge most frequent character pairs', 
            ha='center', fontsize=12, style='italic', color='gray')

    for i, (step_name, tokens, description) in enumerate(steps):
        y = y_positions[i]
        
        # Step name
        ax4.text(0.05, y + 0.05, step_name, fontsize=11, fontweight='bold', color='#2c3e50')
        
        # Tokens visualization
        if isinstance(tokens, list):
            token_str = ' | '.join(tokens[:15])  # Show first 15
        else:
            token_str = ' | '.join(list(tokens)[:15])
        
        ax4.text(0.15, y, token_str, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
        
        # Description
        ax4.text(0.75, y, description, fontsize=9, style='italic', color='#7f8c8d')

    # Add final insights box
    insights_text = """Key Insights:
    • Subword tokenization balances vocabulary size & sequence length
    • Character-level: Simple but inefficient (4-5x longer sequences)
    • Word-level: Efficient but huge vocabulary (millions of words)
    • BPE/SentencePiece: Best of both worlds (~50K-130K vocab)"""

    ax4.text(0.5, 0.05, insights_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.7, edgecolor='black', linewidth=2))

    plt.tight_layout()
    plt.show()

    print("\n📊 Vocabulary Analysis:")
    print(f"   Example: '{sentence}'")
    print(f"   - Character tokens: {char_tokens}")
    print(f"   - GPT-4 subword tokens: {gpt4_tokens}")
    print(f"   - Compression ratio: {char_tokens/gpt4_tokens:.1f}x")
    print(f"\n🎯 Why Not Character-Level Tokenization?")
    print(f"   1. Sequence Length: 4-5x longer for same text")
    print(f"   2. Computational Cost: O(n²) attention → 16x more expensive")
    print(f"   3. Context Window: Wastes precious context on characters, not meaning")
    print(f"   4. Training Time: Longer sequences = slower training")
    print(f"\n💡 Why Subword Tokenization Wins:")
    print(f"   ✓ Balanced vocabulary size (50K-130K tokens)")
    print(f"   ✓ Efficient sequence length (~4x shorter than characters)")
    print(f"   ✓ Handles rare words via subword units")
    print(f"   ✓ Language-agnostic (works across all languages)")
    print(f"   ✓ Optimal for attention mechanisms")
    print(f"\n🔧 Popular Tokenization Methods:")
    print(f"   - BPE (Byte Pair Encoding): GPT models")
    print(f"   - SentencePiece: Llama, PaLM models")
    print(f"   - WordPiece: BERT")

def visualize_training_data():
    # Visualize training data sources and scale over time
    # Training data sources and scale
    data_sources = {
        'Web Pages\n(CommonCrawl)': 60,
        'Books': 16,
        'Wikipedia': 3,
        'Scientific\nPapers': 8,
        'Code\nRepositories': 10,
        'Social Media\n& Forums': 3
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

    # Pie chart of data sources
    colors_pie = plt.cm.Set3(range(len(data_sources)))
    wedges, texts, autotexts = ax1.pie(data_sources.values(), labels=data_sources.keys(), 
                                        autopct='%1.0f%%', startangle=90, colors=colors_pie,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Training Data Sources\n(Approximate Distribution)', fontsize=14, fontweight='bold', pad=20)

    # Scale over time
    models_timeline = ['GPT-2\n(2019)', 'GPT-3\n(2020)', 'PaLM\n(2022)', 'GPT-4\n(2023)', 'Llama 3\n(2024)']
    tokens_trained = [40, 300, 780, 13000, 15000]  # Billions of tokens

    ax2.plot(models_timeline, tokens_trained, marker='o', linewidth=3, markersize=12, color='#2E86AB')
    ax2.fill_between(range(len(models_timeline)), tokens_trained, alpha=0.3, color='#2E86AB')
    ax2.set_ylabel('Training Tokens (Billions)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Data Scale Over Time\n(Exponential Growth)', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')

    # Add annotations
    ax2.annotate('GPT-4 trained on\n~13 trillion tokens!', 
                xy=(3, 13000), xytext=(1.5, 5000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.show()

    print("\n📊 Training Data Facts:")
    print("   - GPT-3: ~500 billion tokens (~1 million books equivalent)")
    print("   - GPT-4: ~13 trillion tokens (estimated)")
    print("   - Cost: $10M - $100M+ just for compute")
    print("   - Time: Months of training on thousands of GPUs")
    print("\n⚠️  Critical Insight:")
    print("   - Quality of training data = Quality of model behavior")
    print("   - Biases in data → Biases in model")
    print("   - Cutoff date means no knowledge of events after training")

def visualize_training_pipeline():
    # Visualize the 3-phase training pipeline

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Phase 1: Training Pipeline Flowchart
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    phases = [
        {
            'name': 'Phase 1: Pre-training',
            'y': 0.75,
            'goal': 'Learn language patterns',
            'data': 'Trillions of tokens from internet',
            'cost': '$10M-$100M+',
            'time': '2-4 months',
            'result': 'Base model (can predict next word)',
            'color': '#3498db'
        },
        {
            'name': 'Phase 2: Supervised Fine-tuning (SFT)',
            'y': 0.50,
            'goal': 'Follow instructions',
            'data': 'Thousands of human-written examples',
            'cost': '$10K-$100K',
            'time': 'Days to weeks',
            'result': 'Instruction-following model',
            'color': '#2ecc71'
        },
        {
            'name': 'Phase 3: RLHF (Alignment)',
            'y': 0.25,
            'goal': 'Align with human values',
            'data': 'Human preference rankings',
            'cost': '$100K-$1M',
            'time': 'Weeks',
            'result': 'Helpful, harmless, honest assistant',
            'color': '#e74c3c'
        }
    ]

    ax1.text(0.5, 0.95, 'The 3-Phase LLM Training Pipeline', 
            ha='center', fontsize=20, fontweight='bold')

    for phase in phases:
        y = phase['y']
        
        # Phase box - smaller and more compact
        rect = plt.Rectangle((0.1, y - 0.08), 0.8, 0.14, 
                            facecolor=phase['color'], alpha=0.2, 
                            edgecolor=phase['color'], linewidth=2)
        ax1.add_patch(rect)
        
        # Phase name
        ax1.text(0.5, y + 0.07, phase['name'], 
                ha='center', fontsize=16, fontweight='bold', color=phase['color'])
        
        # Details - split into two lines for better readability
        line1 = f"Goal: {phase['goal']} | Data: {phase['data']}"
        line2 = f"Cost: {phase['cost']} | Time: {phase['time']}"
        ax1.text(0.5, y + 0.02, line1, 
                ha='center', fontsize=13, color='#2c3e50')
        ax1.text(0.5, y - 0.025, line2, 
                ha='center', fontsize=13, color='#2c3e50')
        
        # Result
        ax1.text(0.5, y - 0.07, f"→ {phase['result']}", 
                ha='center', fontsize=13, fontweight='bold', color='#7f8c8d')
        
        # Arrow to next phase
        if y > 0.25:
            ax1.annotate('', xy=(0.5, y - 0.09), xytext=(0.5, y - 0.17),
                        arrowprops=dict(arrowstyle='<-', lw=2, color='black'))

    ax1.text(0.5, 0.05, '💡 Key: Each phase builds on the previous one', 
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.7))

    # Phase 2: Training loss over time (simulated)
    iterations = np.linspace(0, 100, 1000)

    # Pre-training: steep decline then plateau
    pretraining_loss = 5 * np.exp(-iterations/15) + 1.5 + 0.1 * np.random.randn(1000).cumsum() * 0.01

    # Fine-tuning: starts from lower loss, quick improvement
    finetuning_start = 1.8
    finetuning_loss = finetuning_start * np.exp(-iterations/10) + 0.8

    # RLHF: reward model score (inverse of loss)
    rlhf_reward = 1 - 0.7 * np.exp(-iterations/12)

    ax2.plot(iterations, pretraining_loss, linewidth=3, color='#3498db', 
            label='Pre-training Loss', alpha=0.8)
    ax2.plot(iterations, finetuning_loss, linewidth=3, color='#2ecc71', 
            label='Fine-tuning Loss', alpha=0.8)
    ax2.plot(iterations, rlhf_reward, linewidth=3, color='#e74c3c', 
            label='RLHF Reward Score', linestyle='--', alpha=0.8)

    ax2.set_xlabel('Training Iterations', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Loss / Reward', fontsize=13, fontweight='bold')
    ax2.set_title('Training Progress Across Phases\n(Conceptual View)', 
                fontsize=15, fontweight='bold', pad=20)
    ax2.legend(fontsize=11, loc='right')
    ax2.grid(alpha=0.3)

    # Add annotations
    ax2.annotate('Pre-training:\nLearn basic patterns', 
                xy=(20, 2.8), xytext=(60, 4.5),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2),
                fontsize=10, color='#3498db', fontweight='bold')

    ax2.annotate('Fine-tuning:\nLearn to follow instructions', 
                xy=(25, 1.4), xytext=(65, 2.2),
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
                fontsize=10, color='#2ecc71', fontweight='bold')

    ax2.annotate('RLHF:\nAlign with human values', 
                xy=(40, 0.7), xytext=(70, 0.9),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
                fontsize=10, color='#e74c3c', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("\n🎓 Training Pipeline Explained:")
    print("\n📘 Phase 1: Pre-training (The Foundation)")
    print("   - Model reads trillions of words from the internet")
    print("   - Learns to predict the next word in any context")
    print("   - Acquires world knowledge, grammar, facts, reasoning patterns")
    print("   - Result: A 'base model' that can complete text")
    print("   - Example: GPT-4 base (before instruction tuning)")
    print("\n📗 Phase 2: Supervised Fine-tuning (SFT)")
    print("   - Humans write examples of good assistant behavior")
    print("   - Model learns to respond to instructions")
    print("   - Thousands of high-quality examples (not billions)")
    print("   - Result: Model follows instructions reliably")
    print("   - Example: Can answer 'Write a poem about AI'")
    print("\n📕 Phase 3: RLHF (Reinforcement Learning from Human Feedback)")
    print("   - Humans rank multiple model outputs (A vs B)")
    print("   - Model learns what humans prefer")
    print("   - Teaches helpfulness, harmlessness, honesty")
    print("   - Result: ChatGPT-like behavior")
    print("   - Example: Politely declines harmful requests")
    print("\n💰 Cost Implications:")
    print("   - Pre-training: 95%+ of total cost")
    print("   - Fine-tuning: Much cheaper, can customize")
    print("   - RLHF: Expensive human labor but worth it")
    print("\n⚡ Why This Matters:")
    print("   - Base models are powerful but not user-friendly")
    print("   - Fine-tuning makes them practical")
    print("   - RLHF makes them safe and aligned")
    print("   - Companies can fine-tune open models (e.g., Llama) for specific needs")

# Visualize how temperature affects token selection

# Simulate probability distribution for next token
tokens_example = ['sunny', 'cloudy', 'rainy', 'clear', 'overcast', 'foggy', 'stormy', 'nice']
base_probs = np.array([0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.02, 0.01])

def apply_temperature(probs, temperature):
    """Apply temperature to probability distribution"""
    if temperature == 0:
        # Greedy: always pick highest
        result = np.zeros_like(probs)
        result[np.argmax(probs)] = 1.0
        return result
    # Apply temperature scaling
    logits = np.log(probs + 1e-10)
    scaled_logits = logits / temperature
    # Softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / exp_logits.sum()

def visualize_temperature_effects():
    temperatures = [0.0, 0.5, 1.0, 2.0]

    fig, axes = plt.subplots(4, 1, figsize=(12, 30))

    for ax, temp in zip(axes, temperatures):
        probs = apply_temperature(base_probs, temp)
        
        colors_bars = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(tokens_example)))
        bars = ax.bar(tokens_example, probs, color=colors_bars, alpha=0.8, 
                    edgecolor='black', linewidth=2)
        
        # Highlight the most likely token
        max_idx = np.argmax(probs)
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(4)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            if prob > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., prob + 0.02,
                    f'{prob*100:.1f}%',
                    ha='center', fontweight='bold', fontsize=13)
        
        ax.set_ylabel('Probability', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_title(f'Temperature = {temp}\n{"(Deterministic)" if temp == 0 else "(Creative)" if temp > 1 else "(Balanced)"}', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=13)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add description
        if temp == 0.0:
            desc = "Always picks 'sunny' (most likely)\nPerfect for factual tasks"
        elif temp == 0.5:
            desc = "Strongly favors 'sunny'\nbut allows some variation"
        elif temp == 1.0:
            desc = "Natural distribution\nBalanced creativity"
        else:
            desc = "More uniform distribution\nHighly creative/unpredictable"
        
        ax.text(0.98, 0.95, desc, transform=ax.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    print("\n🎲 Temperature Parameter Explained:")
    print("\n🔵 Temperature = 0 (Deterministic)")
    print("   - Always picks the most likely token")
    print("   - Same input → Same output every time")
    print("   - Use cases: Code generation, data extraction, factual Q&A")
    print("   - Example: '2+2=' → always outputs '4'")
    print("\n🟢 Temperature = 0.5-0.7 (Focused)")
    print("   - Slightly randomized but coherent")
    print("   - Good for most production use cases")
    print("   - Use cases: Customer support, summaries, analysis")
    print("\n🟡 Temperature = 1.0 (Balanced)")
    print("   - Natural probability distribution")
    print("   - Good balance of creativity and coherence")
    print("   - Use cases: General conversation, writing assistance")
    print("\n🔴 Temperature = 1.5-2.0 (Creative)")
    print("   - Highly randomized, unpredictable")
    print("   - Can generate surprising/unusual outputs")
    print("   - Use cases: Creative writing, brainstorming, poetry")
    print("   - Risk: May lose coherence")
    print("\n💡 Business Implications:")
    print("   - Lower temperature = More reliable, predictable (better for automation)")
    print("   - Higher temperature = More creative, diverse (better for ideation)")
    print("   - Different use cases need different settings")
    print("   - Most APIs default to 0.7-1.0")

def visualize_word_by_word():
    # Visualize word-by-word generation process

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 30))

    # 1. Step-by-step generation visualization
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax1.text(0.5, 0.95, 'Word-by-Word Generation Process', 
            ha='center', fontsize=20, fontweight='bold')

    prompt_text = "The future of AI is"
    generated_tokens = ["bright", "and", "full", "of", "potential"]

    steps = [
        ("Step 1", "The future of AI is", None, "Model predicts: 'bright'"),
        ("Step 2", "The future of AI is bright", "bright", "Model predicts: 'and'"),
        ("Step 3", "The future of AI is bright and", "and", "Model predicts: 'full'"),
        ("Step 4", "The future of AI is bright and full", "full", "Model predicts: 'of'"),
        ("Step 5", "The future of AI is bright and full of", "of", "Model predicts: 'potential'"),
    ]

    y_start = 0.82
    y_step = 0.15

    for i, (step_name, context, new_token, prediction) in enumerate(steps):
        y = y_start - (i * y_step)
        
        # Step label
        ax1.text(0.02, y + 0.05, step_name, fontsize=13, fontweight='bold', color='#2c3e50')
        
        # Context box
        context_color = '#e8f4f8' if new_token else '#fff9e6'
        ax1.text(0.15, y + 0.02, context, fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=context_color, 
                        edgecolor='#3498db', linewidth=2))
        
        # New token highlighted
        if new_token:
            ax1.text(0.68, y + 0.02, f'+ "{new_token}"', fontsize=12, fontweight='bold', color='#e74c3c',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', 
                            edgecolor='#e74c3c', linewidth=2))
        
        # Prediction
        ax1.text(0.15, y - 0.03, f'→ {prediction}', fontsize=11, style='italic', color='#7f8c8d')
        
        # Arrow between steps
        if i < len(steps) - 1:
            ax1.annotate('', xy=(0.08, y - 0.08), xytext=(0.08, y - 0.05),
                        arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

    ax1.text(0.5, 0.02, '⚡ Key: Each word depends on ALL previous words', 
            ha='center', fontsize=13, style='italic', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.7))

    # 2. Why it feels like "thinking" - time visualization
    words_output = ["The", "key", "to", "success", "in", "AI", "is", "understanding", "fundamentals"]
    generation_times = [0.05, 0.08, 0.06, 0.12, 0.05, 0.07, 0.06, 0.15, 0.13]  # seconds per word

    cumulative_time = np.cumsum([0] + generation_times)

    ax2.plot(range(len(words_output) + 1), cumulative_time, marker='o', linewidth=3, 
            markersize=10, color='#3498db', alpha=0.8)

    # Add word labels
    for i, (word, time) in enumerate(zip(words_output, cumulative_time[1:])):
        ax2.text(i + 1, time + 0.02, word, ha='center', fontsize=11, 
                fontweight='bold', rotation=45)

    ax2.set_xlabel('Word Number', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Time (seconds)', fontsize=13, fontweight='bold')
    ax2.set_title('Generation Feels Sequential\n(Why you see text appear word-by-word)', 
                fontsize=15, fontweight='bold', pad=20)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-0.5, len(words_output) + 0.5)
    ax2.tick_params(axis='both', labelsize=12)

    # Add annotation
    ax2.annotate('Each word takes time to compute\n→ Sequential appearance', 
                xy=(5, cumulative_time[5]), xytext=(7, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    # 3. Parallel vs Sequential comparison
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    ax3.text(0.5, 0.95, '🚫 Why Parallel Generation is Impossible', 
            ha='center', fontsize=18, fontweight='bold', color='#e74c3c')

    # Parallel (not possible)
    ax3.text(0.5, 0.82, '❌ Parallel Generation (NOT POSSIBLE)', 
            ha='center', fontsize=14, fontweight='bold', color='#e74c3c')

    parallel_words = ["Word 1", "Word 2", "Word 3", "Word 4"]
    y_parallel = 0.70
    for i, word in enumerate(parallel_words):
        x_pos = 0.2 + (i * 0.15)
        ax3.text(x_pos, y_parallel, word, ha='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', 
                        edgecolor='red', linewidth=2))
        # All arrows from input
        ax3.annotate('', xy=(x_pos, y_parallel - 0.02), xytext=(x_pos, 0.58),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='red', linestyle='--'))

    ax3.text(0.5, 0.60, 'Input Context', ha='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', edgecolor='blue', linewidth=2))

    ax3.text(0.5, 0.50, 'Problem: Word 2 needs Word 1 to exist first!', 
            ha='center', fontsize=12, color='red', style='italic')

    # Sequential (actual)
    ax3.text(0.5, 0.35, '✅ Sequential Generation (ACTUAL)', 
            ha='center', fontsize=14, fontweight='bold', color='#2ecc71')

    seq_words = ["Word 1", "Word 2", "Word 3", "Word 4"]
    y_seq = 0.23
    for i, word in enumerate(seq_words):
        x_pos = 0.15 + (i * 0.18)
        ax3.text(x_pos, y_seq, word, ha='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc', 
                        edgecolor='green', linewidth=2))
        # Arrow from previous
        if i > 0:
            ax3.annotate('', xy=(x_pos - 0.02, y_seq), xytext=(x_pos - 0.16, y_seq),
                        arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    ax3.text(0.5, 0.10, 'Each word becomes input for the next → Autoregressive', 
            ha='center', fontsize=12, color='green', style='italic', fontweight='bold')

    # 4. Inference cost visualization
    output_lengths = [10, 50, 100, 200, 500, 1000]
    relative_costs = output_lengths  # Linear relationship
    relative_times = output_lengths  # Linear relationship

    ax4_twin = ax4.twinx()

    # Cost bars
    bars = ax4.bar(range(len(output_lengths)), relative_costs, 
                color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5, label='Cost')

    # Time line
    line = ax4_twin.plot(range(len(output_lengths)), relative_times, 
                        marker='o', linewidth=3, markersize=10, 
                        color='#3498db', label='Generation Time')

    ax4.set_xlabel('Output Length (words)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Relative Cost ($)', fontsize=13, fontweight='bold', color='#e74c3c')
    ax4_twin.set_ylabel('Relative Time (seconds)', fontsize=13, fontweight='bold', color='#3498db')
    ax4.set_title('Cost & Time Scale Linearly with Output Length\n(Every word costs money and time)', 
                fontsize=15, fontweight='bold', pad=20)

    ax4.set_xticks(range(len(output_lengths)))
    ax4.set_xticklabels(output_lengths)
    ax4.grid(alpha=0.3, axis='y')
    ax4.tick_params(axis='both', labelsize=12)
    ax4_twin.tick_params(axis='y', labelsize=12)

    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars, relative_costs)):
        ax4.text(i, cost + 20, f'{cost}x', ha='center', fontweight='bold', fontsize=12, color='#e74c3c')

    # Legend
    ax4.legend(loc='upper left', fontsize=12)
    ax4_twin.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

    print("\n🔄 Word-by-Word Generation Explained:")
    print("\n📝 Autoregressive Process:")
    print("   1. Model reads your prompt")
    print("   2. Predicts most likely next word")
    print("   3. Adds that word to the context")
    print("   4. Predicts next word (considering ALL previous words)")
    print("   5. Repeats until done (stop word or max length)")
    print("\n⏱️ Why It Feels Like 'Thinking':")
    print("   - Each word takes ~50-200ms to generate")
    print("   - Appears sequentially, not all at once")
    print("   - Longer responses take proportionally longer")
    print("   - Gives impression of 'reasoning' or 'writing'")
    print("   - Reality: Pure prediction, no actual thinking")
    print("\n🚫 Why Parallel Generation is Impossible:")
    print("   - Word N depends on words 1 through N-1")
    print("   - Can't predict Word 3 without knowing Word 2")
    print("   - Must process sequentially (autoregressive)")
    print("   - This is fundamental to transformer architecture")
    print("\n💰 Cost Implications:")
    print("   - Every output word costs money")
    print("   - 1000-word output ≈ 100x more expensive than 10-word")
    print("   - Longer outputs = higher latency")
    print("   - Optimize: Be specific to get concise answers")
    print("\n⚡ Performance Impact:")
    print("   - Input words (prompt): Processed in parallel (fast)")
    print("   - Output words: Generated sequentially (slow)")
    print("   - Bottleneck: Output generation, not input processing")
    print("   - Strategy: Minimize output length when possible")
    print("\n💡 Business Implications:")
    print("   ✓ Verbose responses cost more (words + time)")
    print("   ✓ Streaming gives better UX (shows progress)")
    print("   ✓ Set max_words to control costs")
    print("   ✓ Cache prompts when possible (some APIs support this)")
    print("   ✗ Don't expect instant long-form generation")

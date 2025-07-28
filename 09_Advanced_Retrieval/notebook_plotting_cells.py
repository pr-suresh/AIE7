# Cell 1: Import plotting libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Cell 2: Simple Bar Chart (Most Common)
def plot_metrics_bar(metrics_dict, title="Retrieval Metrics Performance"):
    """Create a simple bar chart of metrics"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Create bars
    bars = ax.bar(metrics, values, color='skyblue', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid and rotate labels
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

# Use your evaluation_scores from Ragas
plot_metrics_bar(evaluation_scores, "Naive Retrieval Chain Performance")

# Cell 3: Horizontal Bar Chart
def plot_metrics_horizontal(metrics_dict, title="Retrieval Metrics Performance"):
    """Create a horizontal bar chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Create horizontal bars
    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, values, color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Score', fontsize=14)
    ax.set_ylabel('Metrics', fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center', fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_metrics_horizontal(evaluation_scores, "Naive Retrieval Chain Performance")

# Cell 4: Radar Chart
def plot_metrics_radar(metrics_dict, title="Retrieval Metrics Radar Chart"):
    """Create a radar chart of metrics"""
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    values += values[:1]
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label='Performance')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')
    
    # Fix axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add legend and set range
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.set_ylim(0, 1)
    plt.title(title, size=16, y=1.1, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

plot_metrics_radar(evaluation_scores, "Naive Retrieval Chain Performance")

# Cell 5: Comparison Chart (if you have multiple retrievers)
def plot_retriever_comparison(retrievers_metrics, title="Retriever Comparison"):
    """Compare multiple retrievers"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = list(list(retrievers_metrics.values())[0].keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(retrievers_metrics)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (retriever, metrics_dict) in enumerate(retrievers_metrics.items()):
        values = [metrics_dict[metric] for metric in metrics]
        bars = ax.bar(x + i * width, values, width, label=retriever, 
                     color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_xticks(x + width * (len(retrievers_metrics) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.show()

# Example usage for comparison (uncomment if you have multiple retrievers)
# comparison_data = {
#     'naive': evaluation_scores,  # Your naive retriever scores
#     'bm25': bm25_scores,         # Your BM25 retriever scores
#     'ensemble': ensemble_scores  # Your ensemble retriever scores
# }
# plot_retriever_comparison(comparison_data, "Retriever Performance Comparison")

# Cell 6: Summary Statistics
def plot_metrics_summary(metrics_dict, title="Metrics Summary"):
    """Create a summary statistics chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    values = list(metrics_dict.values())
    
    # Box plot
    ax1.boxplot(values, labels=['All Metrics'])
    ax1.set_title('Distribution of Scores', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Score Distribution', fontweight='bold')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
    Mean: {np.mean(values):.3f}
    Median: {np.median(values):.3f}
    Std: {np.std(values):.3f}
    Min: {np.min(values):.3f}
    Max: {np.max(values):.3f}
    """
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_metrics_summary(evaluation_scores, "Naive Retrieval Chain Metrics Summary")

# Cell 7: Save plots (optional)
def save_plots(metrics_dict, filename_prefix="naive_retrieval"):
    """Save all plots as PNG files"""
    
    # Bar chart
    plot_metrics_bar(metrics_dict)
    plt.savefig(f"{filename_prefix}_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Horizontal bar chart
    plot_metrics_horizontal(metrics_dict)
    plt.savefig(f"{filename_prefix}_horizontal_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Radar chart
    plot_metrics_radar(metrics_dict)
    plt.savefig(f"{filename_prefix}_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    plot_metrics_summary(metrics_dict)
    plt.savefig(f"{filename_prefix}_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved with prefix: {filename_prefix}")

# Uncomment to save plots
# save_plots(evaluation_scores, "naive_retrieval_metrics") 
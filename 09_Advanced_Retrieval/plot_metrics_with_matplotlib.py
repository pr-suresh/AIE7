import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# Example metrics dictionary (replace with your actual data)
metrics_dict = {
    'faithfulness': 0.85,
    'answer_relevancy': 0.78,
    'context_relevancy': 0.82,
    'context_recall': 0.75,
    'answer_correctness': 0.88,
    'answer_similarity': 0.79
}

# Set up the plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Bar Chart - Most common for metrics comparison
def plot_bar_chart(metrics_dict, title="Retrieval Metrics Performance"):
    """Create a bar chart of all metrics"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Create bars with custom colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#4A90E2']
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize the chart
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

# 2. Horizontal Bar Chart
def plot_horizontal_bar(metrics_dict, title="Retrieval Metrics Performance"):
    """Create a horizontal bar chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Create horizontal bars
    y_pos = np.arange(len(metrics))
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize the chart
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Score', fontsize=14)
    ax.set_ylabel('Metrics', fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center', fontweight='bold')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()

# 3. Radar/Spider Chart
def plot_radar_chart(metrics_dict, title="Retrieval Metrics Radar Chart"):
    """Create a radar chart of metrics"""
    # Number of variables
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Values for each axis
    values += values[:1]
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label='Performance')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set the range for the radial axis
    ax.set_ylim(0, 1)
    
    # Add title
    plt.title(title, size=16, y=1.1, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# 4. Gauge Chart
def plot_gauge_chart(metrics_dict, title="Metrics Gauge Chart"):
    """Create gauge charts for each metric"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, value) in enumerate(metrics_dict.items()):
        ax = axes[i]
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=3)
        
        # Value arc
        value_angle = value * np.pi
        theta_value = np.linspace(0, value_angle, 100)
        ax.plot(r * np.cos(theta_value), r * np.sin(theta_value), 'r-', linewidth=5)
        
        # Add text
        ax.text(0, 0.5, f'{value:.3f}', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0, -0.3, metric.replace('_', ' ').title(), ha='center', va='center', fontsize=10)
        
        # Set limits and remove axes
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 5. Heatmap-style visualization
def plot_heatmap_style(metrics_dict, title="Metrics Heatmap"):
    """Create a heatmap-style visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a single row of data
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Create a 2D array for heatmap
    data = np.array(values).reshape(1, -1)
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Customize the plot
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score', rotation=270, labelpad=15)
    
    # Add text annotations
    for i, value in enumerate(values):
        color = 'white' if value < 0.5 else 'black'
        ax.text(i, 0, f'{value:.3f}', ha='center', va='center', 
                color=color, fontweight='bold', fontsize=12)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

# 6. Comparison Chart (if you have multiple retrievers)
def plot_comparison_chart(retrievers_metrics, title="Retriever Comparison"):
    """Compare multiple retrievers"""
    # Example data structure for multiple retrievers
    # retrievers_metrics = {
    #     'naive': {'faithfulness': 0.85, 'answer_relevancy': 0.78, ...},
    #     'bm25': {'faithfulness': 0.82, 'answer_relevancy': 0.81, ...},
    #     'ensemble': {'faithfulness': 0.88, 'answer_relevancy': 0.85, ...}
    # }
    
    if not retrievers_metrics:
        print("No comparison data provided")
        return
    
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

# 7. Summary Statistics Chart
def plot_summary_statistics(metrics_dict, title="Metrics Summary"):
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

# Example usage
if __name__ == "__main__":
    # Plot all chart types
    print("1. Bar Chart:")
    plot_bar_chart(metrics_dict)
    
    print("2. Horizontal Bar Chart:")
    plot_horizontal_bar(metrics_dict)
    
    print("3. Radar Chart:")
    plot_radar_chart(metrics_dict)
    
    print("4. Gauge Chart:")
    plot_gauge_chart(metrics_dict)
    
    print("5. Heatmap Style:")
    plot_heatmap_style(metrics_dict)
    
    print("6. Summary Statistics:")
    plot_summary_statistics(metrics_dict)
    
    # Example for comparison chart (uncomment if you have multiple retrievers)
    # example_comparison = {
    #     'naive': {'faithfulness': 0.85, 'answer_relevancy': 0.78, 'context_relevancy': 0.82},
    #     'bm25': {'faithfulness': 0.82, 'answer_relevancy': 0.81, 'context_relevancy': 0.79},
    #     'ensemble': {'faithfulness': 0.88, 'answer_relevancy': 0.85, 'context_relevancy': 0.86}
    # }
    # plot_comparison_chart(example_comparison) 
"""
Visualization utilities for tropical pruning experiments.

This module provides functions for creating publication-quality visualizations:
- Winner frequency histograms
- Sparsity vs accuracy curves with error bars
- Fine-tuning recovery curves
- Layer-wise importance distributions
- Compression/speedup comparisons
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def setup_publication_style():
    """Set up matplotlib style for publication-quality figures."""
    _check_matplotlib()

    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    if HAS_SEABORN:
        sns.set_palette("colorblind")


def plot_winner_frequency_histogram(
    statistics: Dict[str, Any],
    layer_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
    bins: int = 50,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot histogram of winner frequencies for each layer.

    Args:
        statistics: Dictionary mapping layer names to WinnerStatistics.
        layer_name: Specific layer to plot. If None, plots all layers.
        figsize: Figure size.
        bins: Number of histogram bins.
        save_path: Path to save figure. If None, displays interactively.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()

    if layer_name:
        layers = [layer_name]
    else:
        layers = list(statistics.keys())

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, name in zip(axes, layers):
        stats = statistics[name]
        freq = stats.winner_frequency.cpu().numpy()

        ax.hist(freq, bins=bins, edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(freq), color='red', linestyle='--',
                   label=f'Mean: {np.mean(freq):.3f}')
        ax.axvline(x=0.01, color='orange', linestyle=':',
                   label='1% threshold')

        ax.set_xlabel('Winner Frequency')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}')
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def plot_sparsity_accuracy_curve(
    results: Dict[str, Dict[float, Any]],
    title: str = "Sparsity vs Accuracy",
    xlabel: str = "Sparsity (%)",
    ylabel: str = "Accuracy (%)",
    figsize: Tuple[int, int] = (8, 5),
    show_error_bars: bool = True,
    baseline_acc: Optional[float] = None,
    save_path: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
) -> "plt.Figure":
    """
    Plot sparsity vs accuracy curves with error bars.

    Args:
        results: Dictionary mapping method names to {sparsity: ResultStats}.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size.
        show_error_bars: Whether to show error bars.
        baseline_acc: Baseline accuracy to show as horizontal line.
        save_path: Path to save figure.
        colors: Optional dict mapping method names to colors.
        markers: Optional dict mapping method names to markers.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()
    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    default_markers = ['o', 's', '^', 'D', 'v']

    for i, (method, sparsity_results) in enumerate(results.items()):
        sparsities = sorted(sparsity_results.keys())
        means = []
        stds = []

        for s in sparsities:
            stats = sparsity_results[s]
            if hasattr(stats, 'mean'):
                means.append(stats.mean)
                stds.append(stats.std if hasattr(stats, 'std') else 0)
            elif isinstance(stats, (list, np.ndarray)):
                means.append(np.mean(stats))
                stds.append(np.std(stats))
            else:
                means.append(float(stats))
                stds.append(0)

        color = colors.get(method, default_colors[i % len(default_colors)]) if colors else default_colors[i % len(default_colors)]
        marker = markers.get(method, default_markers[i % len(default_markers)]) if markers else default_markers[i % len(default_markers)]

        x = [s * 100 for s in sparsities]

        if show_error_bars and any(s > 0 for s in stds):
            ax.errorbar(x, means, yerr=stds, label=method,
                       marker=marker, capsize=3, capthick=1,
                       color=color, linewidth=1.5, markersize=6)
        else:
            ax.plot(x, means, label=method, marker=marker,
                   color=color, linewidth=1.5, markersize=6)

    if baseline_acc is not None:
        ax.axhline(y=baseline_acc, color='gray', linestyle='--',
                   label=f'Baseline ({baseline_acc:.1f}%)', alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def plot_finetuning_curve(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Fine-tuning Progress",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot fine-tuning loss and accuracy curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_accs: Optional training accuracy per epoch.
        val_accs: Optional validation accuracy per epoch.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()

    has_acc = train_accs is not None and val_accs is not None
    n_plots = 2 if has_acc else 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=4)
    axes[0].plot(epochs, val_losses, label='Val Loss', marker='s', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if has_acc:
        axes[1].plot(epochs, train_accs, label='Train Acc', marker='o', markersize=4)
        axes[1].plot(epochs, val_accs, label='Val Acc', marker='s', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def plot_layer_importance(
    statistics: Dict[str, Any],
    top_k: int = 20,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot importance scores per layer showing most and least important neurons.

    Args:
        statistics: Dictionary mapping layer names to WinnerStatistics.
        top_k: Number of top/bottom neurons to highlight.
        figsize: Figure size.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()

    layers = list(statistics.keys())
    n_layers = len(layers)

    fig, axes = plt.subplots(n_layers, 1, figsize=figsize)
    if n_layers == 1:
        axes = [axes]

    for ax, name in zip(axes, layers):
        stats = statistics[name]
        freq = stats.winner_frequency.cpu().numpy()

        # Sort by importance
        sorted_indices = np.argsort(freq)[::-1]
        sorted_freq = freq[sorted_indices]

        # Plot all
        x = range(len(sorted_freq))
        ax.bar(x, sorted_freq, color='steelblue', alpha=0.6)

        # Highlight top-k
        ax.bar(x[:top_k], sorted_freq[:top_k], color='green', alpha=0.8,
               label=f'Top {top_k}')

        # Highlight bottom-k
        ax.bar(x[-top_k:], sorted_freq[-top_k:], color='red', alpha=0.8,
               label=f'Bottom {top_k}')

        ax.set_xlabel('Neuron (sorted by importance)')
        ax.set_ylabel('Winner Frequency')
        ax.set_title(f'{name} ({len(freq)} neurons)')
        ax.legend(loc='upper right')
        ax.set_xlim(-1, len(freq))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def plot_calibration_ablation(
    results: Dict[int, Any],
    title: str = "Calibration Sample Ablation",
    xlabel: str = "Calibration Samples",
    ylabel: str = "Accuracy (%)",
    figsize: Tuple[int, int] = (7, 5),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot accuracy vs number of calibration samples.

    Args:
        results: Dictionary mapping sample counts to ResultStats or accuracy values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()
    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    sample_sizes = sorted(results.keys())
    means = []
    stds = []

    for n in sample_sizes:
        stats = results[n]
        if hasattr(stats, 'mean'):
            means.append(stats.mean)
            stds.append(stats.std if hasattr(stats, 'std') else 0)
        elif isinstance(stats, (list, np.ndarray)):
            means.append(np.mean(stats))
            stds.append(np.std(stats))
        else:
            means.append(float(stats))
            stds.append(0)

    ax.errorbar(sample_sizes, means, yerr=stds,
                marker='o', capsize=4, capthick=1,
                linewidth=2, markersize=8, color='#1f77b4')

    # Fill between for confidence
    means = np.array(means)
    stds = np.array(stds)
    ax.fill_between(sample_sizes, means - stds, means + stds,
                    alpha=0.2, color='#1f77b4')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def plot_compression_comparison(
    methods: List[str],
    compression_ratios: List[float],
    accuracies: List[float],
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Create a dual-axis bar chart comparing compression and accuracy.

    Args:
        methods: Method names.
        compression_ratios: Compression ratio for each method.
        accuracies: Accuracy for each method.
        figsize: Figure size.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()
    setup_publication_style()

    fig, ax1 = plt.subplots(figsize=figsize)

    x = np.arange(len(methods))
    width = 0.35

    # Compression bars
    bars1 = ax1.bar(x - width/2, compression_ratios, width,
                    label='Compression Ratio', color='steelblue')
    ax1.set_ylabel('Compression Ratio (x)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Accuracy bars on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, accuracies, width,
                    label='Accuracy', color='coral')
    ax2.set_ylabel('Accuracy (%)', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    ax1.set_xlabel('Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def plot_method_comparison_heatmap(
    results_dict: Dict[str, Dict[str, float]],
    title: str = "Method Comparison",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "RdYlGn",
    annot: bool = True,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Create a heatmap comparing methods across different metrics/sparsities.

    Args:
        results_dict: Nested dict {method: {metric/sparsity: value}}.
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap name.
        annot: Whether to annotate cells with values.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    _check_matplotlib()

    # Convert to 2D array
    methods = list(results_dict.keys())
    metrics = list(results_dict[methods[0]].keys())

    data = np.array([
        [results_dict[m].get(metric, np.nan) for metric in metrics]
        for m in methods
    ])

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(data, annot=annot, fmt='.2f', cmap=cmap,
                    xticklabels=metrics, yticklabels=methods,
                    ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    else:
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(methods)
        plt.colorbar(im, ax=ax, label='Accuracy (%)')

        if annot:
            for i in range(len(methods)):
                for j in range(len(metrics)):
                    ax.text(j, i, f'{data[i, j]:.2f}',
                           ha='center', va='center', color='black')

    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def generate_paper_figures(
    results: Dict,
    output_dir: str,
    prefix: str = "tropical_pruning",
):
    """
    Generate all figures needed for the paper.

    Args:
        results: Complete results dictionary.
        output_dir: Directory to save figures.
        prefix: Filename prefix.
    """
    _check_matplotlib()
    import os
    os.makedirs(output_dir, exist_ok=True)

    setup_publication_style()

    # Generate sparsity-accuracy curve
    if "method_results" in results:
        plot_sparsity_accuracy_curve(
            results["method_results"],
            title="Accuracy vs Sparsity",
            baseline_acc=results.get("baseline_accuracy"),
            save_path=os.path.join(output_dir, f"{prefix}_sparsity_accuracy.pdf"),
        )

    # Generate calibration ablation
    if "calibration_ablation" in results:
        plot_calibration_ablation(
            results["calibration_ablation"],
            save_path=os.path.join(output_dir, f"{prefix}_calibration_ablation.pdf"),
        )

    print(f"Figures saved to {output_dir}")

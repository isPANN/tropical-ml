"""
MNIST Tropical Pruning Experiment with Baseline Comparisons

This script compares winner-based tropical pruning against standard baselines:
1. Tropical Winner-based Pruning (ours)
2. L1 Magnitude Pruning
3. L2 Magnitude Pruning
4. Random Pruning (lower bound)
5. Activation Sparsity Pruning
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent to path for local development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_pruning import (
    WinnerCounter,
    TropicalPruner,
    WinnerFrequencyCriterion,
    MagnitudeStructuredPruner,
    RandomStructuredPruner,
    ActivationSparsityPruner,
)


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST classification."""

    def __init__(self, hidden_sizes: Tuple[int, ...] = (256, 128)):
        super().__init__()
        layers = []
        in_features = 784  # 28x28

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, 10))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        return self.layers(x)


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get MNIST train, calibration, and test dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    # Split train into train and calibration
    train_size = 55000
    cal_size = 5000
    train_subset = Subset(train_dataset, range(train_size))
    cal_subset = Subset(train_dataset, range(train_size, train_size + cal_size))

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    cal_loader = DataLoader(
        cal_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, cal_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Train the model on MNIST."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        print(f"Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}, "
              f"Acc={100. * correct / total:.2f}%")

    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluate model accuracy."""
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_comparison_experiment(
    model: nn.Module,
    cal_loader: DataLoader,
    test_loader: DataLoader,
    sparsity_levels: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Run comparison experiment with all pruning methods.

    Returns dictionary with results for each method and sparsity level.
    """
    model = model.to(device)

    print("\n" + "=" * 70)
    print("PRUNING METHODS COMPARISON EXPERIMENT")
    print("=" * 70)

    # Step 1: Evaluate baseline
    print("\nStep 1: Evaluating baseline model...")
    baseline_acc = evaluate_model(model, test_loader, device)
    baseline_params = count_parameters(model)
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")
    print(f"  Baseline parameters: {baseline_params:,}")

    # Step 2: Collect statistics for tropical and activation-based pruning
    print("\nStep 2: Collecting statistics for tropical pruning...")
    counter = WinnerCounter(model, track_margin=True)
    stats = counter.collect(cal_loader, show_progress=True)

    # Analyze winner statistics
    pruner = TropicalPruner(model, stats, criterion=WinnerFrequencyCriterion())
    analysis = pruner.analyze_winners()
    for layer_name, layer_analysis in analysis.items():
        print(f"  {layer_name}: {layer_analysis['never_win']} neurons never win "
              f"({100*layer_analysis['never_win']/layer_analysis['num_neurons']:.1f}%)")

    print("\nStep 3: Collecting activations for activation-based pruning...")
    activation_pruner = ActivationSparsityPruner(model)
    activation_pruner.collect_activations(cal_loader)

    # Initialize results
    results = {
        "baseline_accuracy": baseline_acc,
        "baseline_parameters": baseline_params,
        "methods": {
            "tropical": {},
            "magnitude_l1": {},
            "magnitude_l2": {},
            "activation": {},
            "random": {},
        }
    }

    # Step 4: Run pruning experiments
    print("\nStep 4: Running pruning experiments...")
    print("\n" + "-" * 70)
    print(f"{'Sparsity':<10} {'Tropical':<12} {'Mag-L1':<12} {'Mag-L2':<12} {'Activation':<12} {'Random':<12}")
    print("-" * 70)

    for sparsity in sparsity_levels:
        row = f"{sparsity*100:.0f}%{'':<7}"

        # 1. Tropical winner-based pruning
        tropical_pruner = TropicalPruner(model, stats, criterion=WinnerFrequencyCriterion())
        tropical_pruned = tropical_pruner.prune(sparsity=sparsity)
        tropical_acc = evaluate_model(tropical_pruned, test_loader, device)
        tropical_params = count_parameters(tropical_pruned)
        results["methods"]["tropical"][sparsity] = {
            "accuracy": tropical_acc,
            "parameters": tropical_params,
        }
        row += f"{tropical_acc:.2f}%{'':<6}"

        # 2. L1 Magnitude pruning
        mag_l1_pruner = MagnitudeStructuredPruner(model, norm="l1")
        mag_l1_pruned = mag_l1_pruner.prune(sparsity=sparsity)
        mag_l1_acc = evaluate_model(mag_l1_pruned, test_loader, device)
        mag_l1_params = count_parameters(mag_l1_pruned)
        results["methods"]["magnitude_l1"][sparsity] = {
            "accuracy": mag_l1_acc,
            "parameters": mag_l1_params,
        }
        row += f"{mag_l1_acc:.2f}%{'':<6}"

        # 3. L2 Magnitude pruning
        mag_l2_pruner = MagnitudeStructuredPruner(model, norm="l2")
        mag_l2_pruned = mag_l2_pruner.prune(sparsity=sparsity)
        mag_l2_acc = evaluate_model(mag_l2_pruned, test_loader, device)
        mag_l2_params = count_parameters(mag_l2_pruned)
        results["methods"]["magnitude_l2"][sparsity] = {
            "accuracy": mag_l2_acc,
            "parameters": mag_l2_params,
        }
        row += f"{mag_l2_acc:.2f}%{'':<6}"

        # 4. Activation sparsity pruning
        act_pruned = activation_pruner.prune(sparsity=sparsity)
        act_acc = evaluate_model(act_pruned, test_loader, device)
        act_params = count_parameters(act_pruned)
        results["methods"]["activation"][sparsity] = {
            "accuracy": act_acc,
            "parameters": act_params,
        }
        row += f"{act_acc:.2f}%{'':<6}"

        # 5. Random pruning (average of 3 runs)
        random_accs = []
        for seed in [42, 123, 456]:
            random_pruner = RandomStructuredPruner(model, seed=seed)
            random_pruned = random_pruner.prune(sparsity=sparsity)
            random_accs.append(evaluate_model(random_pruned, test_loader, device))
        random_acc = sum(random_accs) / len(random_accs)
        random_params = count_parameters(random_pruned)
        results["methods"]["random"][sparsity] = {
            "accuracy": random_acc,
            "parameters": random_params,
        }
        row += f"{random_acc:.2f}%"

        print(row)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Accuracy Drop from Baseline")
    print("=" * 70)
    print(f"\nBaseline: {baseline_acc:.2f}% accuracy")
    print("\n" + "-" * 70)
    print(f"{'Sparsity':<10} {'Tropical':<12} {'Mag-L1':<12} {'Mag-L2':<12} {'Activation':<12} {'Random':<12}")
    print("-" * 70)

    for sparsity in sparsity_levels:
        row = f"{sparsity*100:.0f}%{'':<7}"
        for method in ["tropical", "magnitude_l1", "magnitude_l2", "activation", "random"]:
            acc = results["methods"][method][sparsity]["accuracy"]
            drop = baseline_acc - acc
            row += f"{drop:+.2f}%{'':<6}"
        print(row)

    # Find best method at each sparsity
    print("\n" + "=" * 70)
    print("BEST METHOD AT EACH SPARSITY")
    print("=" * 70)
    for sparsity in sparsity_levels:
        best_method = None
        best_acc = 0
        for method in ["tropical", "magnitude_l1", "magnitude_l2", "activation", "random"]:
            acc = results["methods"][method][sparsity]["accuracy"]
            if acc > best_acc:
                best_acc = acc
                best_method = method
        print(f"  {sparsity*100:.0f}% sparsity: {best_method} ({best_acc:.2f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="MNIST Pruning Comparison Experiment")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--skip-training", action="store_true", help="Load pretrained model")
    parser.add_argument("--model-path", type=str, default="mnist_mlp.pt")
    parser.add_argument("--save-results", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Get data
    train_loader, cal_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    # Create or load model
    model = SimpleMLP(hidden_sizes=tuple(args.hidden_sizes))

    if args.skip_training and Path(args.model_path).exists():
        print(f"Loading pretrained model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    else:
        print("Training model...")
        model = train_model(model, train_loader, num_epochs=args.epochs, lr=args.lr, device=device)
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    # Run comparison experiment
    results = run_comparison_experiment(
        model, cal_loader, test_loader, device=device
    )

    # Save results
    if args.save_results:
        import json
        # Convert tensor values to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return float(obj)

        with open(args.save_results, 'w') as f:
            json.dump(convert(results), f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()

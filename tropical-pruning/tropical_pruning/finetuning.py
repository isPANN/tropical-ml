"""
Fine-tuning support for pruned models.

This module provides utilities for fine-tuning pruned neural networks to recover
accuracy lost during pruning. Supports various learning rate schedules and
training configurations.

Key Functions:
    - finetune_pruned_model: Main fine-tuning function
    - create_optimizer: Optimizer factory with common configurations
    - create_scheduler: LR scheduler factory
"""

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
    OneCycleLR,
    LambdaLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning."""

    epochs: int = 10
    lr: float = 0.001
    lr_schedule: str = "cosine"  # "cosine", "step", "multistep", "onecycle", "constant"
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "sgd"  # "sgd", "adam", "adamw"

    # Step scheduler params
    step_size: int = 3
    gamma: float = 0.1

    # MultiStep params
    milestones: List[int] = None

    # OneCycle params
    max_lr: float = 0.01
    pct_start: float = 0.3

    # Training params
    grad_clip: Optional[float] = None
    label_smoothing: float = 0.0

    # Early stopping
    patience: int = 0  # 0 = disabled

    def __post_init__(self):
        if self.milestones is None:
            self.milestones = [5, 8]


@dataclass
class FinetuneResult:
    """Results from fine-tuning."""

    final_train_loss: float
    final_train_acc: float
    final_val_loss: float
    final_val_acc: float
    best_val_acc: float
    train_losses: List[float]
    train_accs: List[float]
    val_losses: List[float]
    val_accs: List[float]
    epochs_trained: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_train_loss": self.final_train_loss,
            "final_train_acc": self.final_train_acc,
            "final_val_loss": self.final_val_loss,
            "final_val_acc": self.final_val_acc,
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
            "epochs_trained": self.epochs_trained,
        }


def finetune_pruned_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[FinetuneConfig] = None,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
    return_best: bool = True,
) -> Tuple[nn.Module, FinetuneResult]:
    """
    Fine-tune a pruned model to recover accuracy.

    Args:
        model: Pruned model to fine-tune.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Fine-tuning configuration. Uses defaults if None.
        device: Computation device. Auto-detects if None.
        show_progress: Whether to show progress bars.
        return_best: If True, return model with best validation accuracy.

    Returns:
        Tuple of (fine-tuned model, FinetuneResult).

    Example:
        >>> config = FinetuneConfig(epochs=10, lr=0.001, lr_schedule="cosine")
        >>> model, result = finetune_pruned_model(
        ...     pruned_model, train_loader, val_loader, config
        ... )
        >>> print(f"Best val accuracy: {result.best_val_acc:.2f}%")
    """
    if config is None:
        config = FinetuneConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Create scheduler
    scheduler = create_scheduler(optimizer, config, len(train_loader))

    # Loss function
    if config.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Tracking
    train_losses: List[float] = []
    train_accs: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    # Training loop
    epoch_pbar = tqdm(range(config.epochs), desc="Fine-tuning", disable=not show_progress)

    for epoch in epoch_pbar:
        # Train
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, config, show_progress and epoch == 0
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = _validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update progress
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.2f}%",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.2f}%",
        })

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if return_best:
                best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if config.patience > 0 and patience_counter >= config.patience:
            break

        # Step scheduler (for non-batch schedulers)
        if config.lr_schedule in ["step", "multistep", "cosine"]:
            scheduler.step()

    # Load best model if requested
    if return_best and best_model_state is not None:
        model.load_state_dict(best_model_state)

    result = FinetuneResult(
        final_train_loss=train_losses[-1],
        final_train_acc=train_accs[-1],
        final_val_loss=val_losses[-1],
        final_val_acc=val_accs[-1],
        best_val_acc=best_val_acc,
        train_losses=train_losses,
        train_accs=train_accs,
        val_losses=val_losses,
        val_accs=val_accs,
        epochs_trained=len(train_losses),
    )

    return model, result


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: FinetuneConfig,
    show_batch_progress: bool = False,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    iterator = train_loader
    if show_batch_progress:
        iterator = tqdm(train_loader, desc="Training", leave=False)

    for batch in iterator:
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        else:
            inputs, targets = batch, None

        inputs = inputs.to(device)
        if targets is not None:
            targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if targets is not None:
            loss = criterion(outputs, targets)
            loss.backward()

            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Step scheduler for batch-level schedulers
            if config.lr_schedule == "onecycle":
                scheduler.step()

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)

            outputs = model(inputs)

            if targets is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def create_optimizer(
    model: nn.Module,
    config: FinetuneConfig,
) -> optim.Optimizer:
    """
    Create optimizer based on configuration.

    Args:
        model: Model to optimize.
        config: Fine-tuning configuration.

    Returns:
        Configured optimizer.
    """
    params = model.parameters()

    if config.optimizer.lower() == "sgd":
        return optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "adam":
        return optim.Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "adamw":
        return optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_scheduler(
    optimizer: optim.Optimizer,
    config: FinetuneConfig,
    steps_per_epoch: int,
) -> Any:
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: Optimizer to schedule.
        config: Fine-tuning configuration.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        Configured scheduler.
    """
    if config.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.lr_schedule == "step":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.lr_schedule == "multistep":
        return MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.lr_schedule == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.pct_start,
        )
    elif config.lr_schedule == "constant":
        return LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")


def quick_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Quick fine-tuning with sensible defaults.

    Simplified interface for common fine-tuning scenarios.

    Args:
        model: Model to fine-tune.
        train_loader: Training data.
        val_loader: Validation data.
        epochs: Number of epochs.
        lr: Learning rate.
        device: Computation device.

    Returns:
        Tuple of (model, metrics dict).
    """
    config = FinetuneConfig(
        epochs=epochs,
        lr=lr,
        lr_schedule="cosine",
        optimizer="sgd",
    )

    model, result = finetune_pruned_model(
        model, train_loader, val_loader, config, device
    )

    return model, result.to_dict()


def knowledge_distillation_finetune(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[FinetuneConfig] = None,
    alpha: float = 0.5,
    temperature: float = 4.0,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Tuple[nn.Module, FinetuneResult]:
    """
    Fine-tune a pruned model using knowledge distillation from the original model.

    Uses a combination of hard labels (cross-entropy) and soft labels from teacher.

    Args:
        student_model: Pruned model to fine-tune.
        teacher_model: Original unpruned model (teacher).
        train_loader: Training data.
        val_loader: Validation data.
        config: Fine-tuning configuration.
        alpha: Weight for distillation loss (vs hard label loss).
        temperature: Softmax temperature for distillation.
        device: Computation device.
        show_progress: Show progress bars.

    Returns:
        Tuple of (fine-tuned student model, FinetuneResult).
    """
    if config is None:
        config = FinetuneConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Teacher always in eval mode

    optimizer = create_optimizer(student_model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))

    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None

    epoch_pbar = tqdm(range(config.epochs), desc="KD Fine-tuning", disable=not show_progress)

    for epoch in epoch_pbar:
        student_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            # Student predictions
            student_outputs = student_model(inputs)

            # Teacher predictions (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # Hard label loss
            hard_loss = ce_criterion(student_outputs, targets)

            # Soft label loss (distillation)
            soft_student = torch.log_softmax(student_outputs / temperature, dim=1)
            soft_teacher = torch.softmax(teacher_outputs / temperature, dim=1)
            soft_loss = kl_criterion(soft_student, soft_teacher) * (temperature ** 2)

            # Combined loss
            loss = (1 - alpha) * hard_loss + alpha * soft_loss
            loss.backward()

            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.grad_clip)

            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if config.lr_schedule == "onecycle":
                scheduler.step()

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = _validate(student_model, val_loader, ce_criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(student_model.state_dict())

        epoch_pbar.set_postfix({
            "train_acc": f"{train_acc:.2f}%",
            "val_acc": f"{val_acc:.2f}%",
        })

        if config.lr_schedule in ["step", "multistep", "cosine"]:
            scheduler.step()

    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)

    result = FinetuneResult(
        final_train_loss=train_losses[-1],
        final_train_acc=train_accs[-1],
        final_val_loss=val_losses[-1],
        final_val_acc=val_accs[-1],
        best_val_acc=best_val_acc,
        train_losses=train_losses,
        train_accs=train_accs,
        val_losses=val_losses,
        val_accs=val_accs,
        epochs_trained=len(train_losses),
    )

    return student_model, result

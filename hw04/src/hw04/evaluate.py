"""Evaluation entry point for hw04."""

from pathlib import Path
from flax import nnx
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import structlog

from .config import load_settings, PlottingSettings

from .data import Data
from .logging import configure_logging
from .model import Classifier
from .training import train

log = structlog.get_logger()


def optax_learning_rate_schedule(
    total_steps: int,  # decay steps + warmup_steps = total_steps
    base_learning_rate: float,
    warmup_percentage: float = 0.10,  # allocating 10% of steps to warmup
) -> optax.Schedule:
    warmup_steps = int(total_steps * warmup_percentage)
    decay_steps = total_steps - warmup_steps
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_learning_rate,
        warmup_steps=warmup_steps,  # length of linear warmup (https://optax.readthedocs.io/en/latest/api/generated/optax.schedules.warmup_cosine_decay_schedule.html)
        decay_steps=decay_steps,
        end_value=1e-5,
    )


def run_single_experiment(  # we need to do this everytime to clear model weights and such
    name: str,
    settings,
    residual: bool,
    augment: bool,
    use_schedule: bool,
    seed: int = 31415,
) -> dict:
    log.info(  # w logger
        f"Ablation: {name}",
        residual=residual,
        augment=augment,
        use_schedule=use_schedule,
    )

    np_rng = np.random.default_rng(seed)
    model_key = jax.random.key(seed)

    data = Data(batch_size=settings.training.batch_size, augmentation=augment)

    model = Classifier(
        input_channels=3,
        layer_channels=[32, 64, 128],
        strides=[1, 2, 2],
        num_classes=10,
        num_groups=8,
        residual=residual,
        rngs=nnx.Rngs(params=model_key),
    )

    if use_schedule:
        schedule = optax_learning_rate_schedule(
            total_steps=settings.training.num_iters,
            base_learning_rate=settings.training.learning_rate,
        )
        optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    else:
        optimizer = optax.adam(learning_rate=settings.training.learning_rate)

    optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    history = train(  # contains all the info we want to plot
        model,
        optimizer,
        data,
        settings.training,
        np_rng,
    )
    return history


def plot_and_save_artifacts(  # avoided plotting.py since this oinly runs on just eval
    histories: dict[str, dict],
    settings: PlottingSettings,
) -> None:
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    styles = {
        "Baseline (Full ResNet)": {
            "color": "#1f77b4",
            "lw": 2.2,
            "zorder": 4,
        },
        "Ablation: Plain CNN (No Skip)": {
            "color": "#ff7f0e",
            "lw": 1.6,
            "ls": "--",
        },
        "Ablation: No Augmentation": {
            "color": "#2ca02c",
            "lw": 1.6,
            "ls": "-.",
        },
        "Ablation: Constant Learning Rate": {
            "color": "#d62728",
            "lw": 1.6,
            "ls": ":",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    # training curves
    ax_loss = axes[0]
    for name, hist in histories.items():
        ax_loss.plot(
            hist["step"][::10],
            hist["loss"][::10],
            label=name,
        )
    ax_loss.set_title("Training Loss Comparison", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8, loc="upper right")

    # validation curves
    ax_acc = axes[1]
    for name, hist in histories.items():
        acc_pct = [acc * 100 for acc in hist["eval_accuracy"]]
        ax_acc.plot(
            hist["eval_step"],
            acc_pct,
            label=name,
            marker="o",
            markersize=3,
        )
    ax_acc.set_title("Test Accuracy Comparison", fontsize=11, fontweight="bold")
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plot_file = out_dir / "ablation_learning_curves.svg"
    plt.savefig(plot_file)
    plt.close()

    # 3. table
    table_rows = []
    for name, hist in histories.items():
        final_loss = hist["loss"][-1]
        final_acc = hist["eval_accuracy"][-1] * 100
        table_rows.append(
            {
                "Model Variant": name,
                "Final Loss": f"{final_loss:.4f}",
                "Final Test Accuracy": f"{final_acc:.2f}%",
            }
        )

    df = pd.DataFrame(table_rows)
    md_content = df.to_markdown(index=False)

    table_file = out_dir / "ablation_table.md"
    with open(table_file, "w") as f:
        f.write(md_content)

    log.info(
        "Ablation artifacts generated successfully",
        plot_path=str(plot_file),
        table_path=str(table_file),
    )
    print("\n" + md_content + "\n")


def main() -> None:
    """CLI entry point for homework evaluation."""
    configure_logging()
    settings = load_settings()

    experiments = [
        ("Baseline", True, True, True),
        ("Ablation: Plain CNN (No Skip)", False, True, True),
        ("Ablation: No Augmentation", True, False, True),
        ("Ablation: Constant Learning Rate", True, True, False),
    ]

    histories = {}
    for name, res, aug, sched in experiments:
        histories[name] = run_single_experiment(
            name=name,
            settings=settings,
            residual=res,
            augment=aug,
            use_schedule=sched,
            seed=settings.random_seed,
        )

    plot_and_save_artifacts(histories, settings.plotting)


if __name__ == "__main__":
    main()

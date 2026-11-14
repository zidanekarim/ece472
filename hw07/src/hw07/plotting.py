from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_sae_results(
    latent_dims: list[int],
    results: dict[int, dict[str, float]],
    save_path: Path | str,
) -> None:
    """Plot R^2 (variance explained) and dead latent fraction across dictionary dimensions."""
    expansions = [f"{m}\n({m // 384}x)" for m in latent_dims]
    r2_vals = [results[m]["r2"] for m in latent_dims]
    dead_pcts = [results[m]["dead_fraction"] * 100.0 for m in latent_dims]
    l0_vals = [results[m]["l0"] for m in latent_dims]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(latent_dims))

    # Panel 1: Variance Explained (R^2)
    bars1 = ax1.bar(x, r2_vals, width=0.5, color="#2563eb")
    ax1.set_xticks(x)
    ax1.set_xticklabels(expansions)
    ax1.set_ylabel("Variance Explained ($R^2$)")
    ax1.set_title(f"Reconstruction Fidelity ($L_0 \\approx {np.mean(l0_vals):.1f}$)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars1, r2_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel 2: Dead Latent Fraction (%)
    bars2 = ax2.bar(x, dead_pcts, width=0.5, color="#dc2626")
    ax2.set_xticks(x)
    ax2.set_xticklabels(expansions)
    ax2.set_ylabel("Dead Latents (%)")
    ax2.set_title("Dead Latent Fraction vs. Dictionary Size")
    ax2.set_ylim(0, max(max(dead_pcts) * 1.25, 10.0))
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars2, dead_pcts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)